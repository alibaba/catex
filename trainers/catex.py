# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import random
import json
import os
from glob import glob
from tqdm import tqdm
from contextlib import nullcontext
import shutil

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from synthesis.feature_sample import IDFeatPool
from ood.posthoc import applyReAct, applyBATS, applyASH

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
perturb_methods = ['neg', 'zero', 'randn', 'randn_add', 'swap']  # 


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        ctx_len = prompts.shape[1]  # TODO: compatible for dynamic context length
        x = prompts + self.positional_embedding.type(self.dtype)[:ctx_len]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CATEX.N_CTX
        ctx_init = cfg.TRAINER.CATEX.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.adjust_cls_promt = False
        self.cfg = cfg

        ctx_common = None
        if ctx_init and 'ensemble' not in ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.CATEX.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_cm = nn.Parameter(ctx_common) if ctx_common is not None else None
        if cfg.TRAINER.OOD_PROMPT:
            if cfg.TRAINER.OOD_PROMPT_NUM > 1:
                self.ctx_ood = []
                for _ in range(cfg.TRAINER.OOD_PROMPT_NUM):
                    ctx_ood = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_ood, std=0.02)
                    self.ctx_ood.append(nn.Parameter(ctx_ood))
                self.ctx_ood = nn.ParameterList(self.ctx_ood)
            else:  ## TODO: compatible for pre-trained weights
                ctx_ood = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_ood, std=0.02)
                self.ctx_ood = nn.Parameter(ctx_ood)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.adjust_cls_promt:
            self.token_suffix = nn.Parameter(embedding[:, 1 + n_ctx :, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.CATEX.CLASS_TOKEN_POSITION

    def forward(self, perturb='none', ood_prompt=False, ood_prompt_idx=None):
        # ctx = self.ctx
        if ood_prompt:
            assert perturb == 'none', perturb
            if ood_prompt_idx is None:
                assert self.cfg.TRAINER.OOD_PROMPT_NUM == 1
                ctx = self.ctx_ood
            else:
                ctx = self.ctx_ood[ood_prompt_idx]
        else:
            ctx = self.perturb_prompt(perturb)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if self.ctx_cm is not None:
            ctx_cm = self.ctx_cm.expand(self.n_cls, -1, -1)
            ctx = torch.cat((ctx_cm, ctx), dim=1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts

    def perturb_prompt(self, method='none'):
        if method == 'none':
            return self.ctx
        
        coef_dict = {
            'neg': [-1., 0.], 'zero': [0., 0.], 'randn': [0., 1.], 'randn_add': [1., 1.], 'swap': [0., 1.]
        }
        assert method in coef_dict
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        ncls, nctx, ndim = ctx.shape
        assert nctx > 1

        for i in range(self.cfg.TRAINER.ID_PERTUR_NUM):
            # perturb one prompt for each class
            ctx_ind = torch.randint(0, nctx, size=(ncls,))
            cls_ind = torch.arange(ncls)

            src_mask = torch.ones((ncls, nctx, 1)).type_as(ctx)
            src_mask[cls_ind, ctx_ind] = 0.
            src_ctx = ctx[cls_ind, ctx_ind].detach()
            
            if method == 'swap':
                ori_ind = torch.arange(ncls)
                while True:
                    rand_ind = torch.randperm(ncls)
                    if (ori_ind != rand_ind).all():
                        noise = src_ctx[rand_ind]
                        break
            else:
                noise = torch.randn_like(ctx[:, 0, :])
            src_coef, noise_coef = coef_dict[method]

            perturb = torch.zeros_like(ctx)
            perturb[cls_ind, ctx_ind] = src_coef * src_ctx + noise_coef * noise

            ctx = ctx * src_mask + perturb * (1. - src_mask)

        return ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.token_embedding = clip_model.token_embedding

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale if not cfg.TRAINER.OOD_TEST else torch.zeros_like(clip_model.logit_scale)
        self.dtype = clip_model.dtype
        self.feat_dim = clip_model.text_projection.data.shape[1]

        self.text_feature_ensemble = self.prompt_ensemble() if cfg.TRAINER.CATEX.CTX_INIT == 'ensemble' else None

    @torch.no_grad()
    def prompt_ensemble(self, learned_text_features=None):
        if learned_text_features is None:
            imagenet_templates = [  # for NPOS
                'a photo of a {}.',
                'a blurry photo of a {}.',
                'a black and white photo of a {}.',
                'a low contrast photo of a {}.',
                'a high contrast photo of a {}.',
                'a bad photo of a {}.',
                'a good photo of a {}.',
                'a photo of a small {}.',
                'a photo of a big {}.',
                'a photo of the {}.',
                'a blurry photo of the {}.',
                'a black and white photo of the {}.',
                'a low contrast photo of the {}.',
                'a high contrast photo of the {}.',
                'a bad photo of the {}.',
                'a good photo of the {}.',
                'a photo of the small {}.',
                'a photo of the big {}.',
            ]
        else:
            imagenet_templates = [  # for MCM
                'a photo of a {}.',
                'a blurry photo of a {}.',
                'a photo of many {}.',
                'a black and white photo of a {}.',
                'a photo of the large {}.',
                'a photo of the small {}.',
            ]
            lambd = 0.5

        dtype = self.text_encoder.dtype
        self.text_encoder = self.text_encoder.cuda()
        self.token_embedding = self.token_embedding.cuda()

        text_feature = []
        for ci, classname in enumerate(self.classnames):
            texts = [template.format(classname) for template in imagenet_templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            embedding = self.token_embedding(texts).type(dtype)
            class_embeddings = self.text_encoder(embedding, texts) # embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if learned_text_features is not None:
                class_embeddings = torch.cat((class_embeddings, lambd * learned_text_features[ci:ci+1]))

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_feature.append(class_embedding)
        text_feature = torch.stack(text_feature, dim=0).type(dtype)

        return text_feature

    def get_text_features(self, perturb='none', ood_prompt=False, ood_prompt_idx=None, return_norm=True):
        if self.text_feature_ensemble is not None and ood_prompt is False and perturb == 'none':
            assert return_norm
            text_features = self.text_feature_ensemble
        else:
            prompts = self.prompt_learner(perturb, ood_prompt=ood_prompt, ood_prompt_idx=ood_prompt_idx)
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            if return_norm:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def get_all_ood_text_features(self):
        x = []
        for ood_prompt_idx in range(self.prompt_learner.cfg.TRAINER.OOD_PROMPT_NUM):
            x.append(self.get_text_features(ood_prompt=True, ood_prompt_idx=ood_prompt_idx))
        return torch.stack(x, dim=1)  # shape(1000,5,512)

    def get_logits(self, image_features, text_features, logit_scale=None):
        if logit_scale is None:
            logit_scale = self.logit_scale.exp()
        
        if text_features.dim() == 2:
            logits = image_features.float() @ text_features.float().t()
        else:
            n = text_features.size(0)
            logits = torch.bmm(image_features.unsqueeze(0).repeat(n,1,1), text_features.transpose(1,2)).max(dim=0)[0] #.mean(dim=0)
        
        return logits * logit_scale

    def forward(self, image, perturb='none', ood_prompt=False, 
                return_feat=False, return_norm=True, posthoc=None):
        if len(image.shape) == 2:
            image_features = image
        else:
            assert len(image.shape) == 4
            image_features = self.image_encoder(image.type(self.dtype))
        if return_feat and not return_norm:
            ret_feat = image_features.detach().clone()
        
        if posthoc == 'apply_react':
            image_features = applyReAct(image_features)
        elif posthoc == 'apply_bats':
            image_features = applyBATS(image_features)
        elif posthoc == 'apply_ash':
            image_features = applyASH(image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if return_feat and return_norm:
            ret_feat = image_features.detach().clone()

        text_features = self.get_text_features(perturb, ood_prompt=ood_prompt)

        logits = self.get_logits(image_features, text_features)

        if return_feat:
            return ret_feat, text_features, logits
        else:
            return logits

    def calc_prompt_loss(self, clean_feat=None, perturb_feat=None):
        l_intra, l_inter = 0., 0.

        # # 1. class-specific prompts should not be similar to class-name prompts
        # csc_prompts = F.normalize(self.prompt_learner.ctx, p=2, dim=-1)
        # cls_prompts = F.normalize(self.prompt_learner.token_suffix[:, :max(self.prompt_learner.name_lens), :], p=2, dim=-1)
        # prompts = torch.cat((csc_prompts, cls_prompts), dim=1)   # shape(ncls, nctx, ndim)
        # similarity = torch.bmm(prompts, prompts.transpose(1,2))  # shape(ncls, nctx, nctx)
        # diag = torch.arange(similarity.shape[1])
        # similarity[:, diag, diag] = -1.

        # l_intra += similarity.max(dim=-1)[0].relu().mean()

        # 2. prompts should obviously affect the text-feature
        if clean_feat is None:
            clean_feat = self.get_text_features()
        if perturb_feat is None:
            # with torch.no_grad():
            perturb_feat = self.get_text_features(random.choice(perturb_methods))
        similarity = (perturb_feat * clean_feat).sum(dim=1)

        l_inter += (similarity - 0.8).relu().mean()

        return l_intra + l_inter

    def calc_ood_prompt_loss(self, image, logits, label):
        perturb_logits = self.forward(image, perturb=random.choice(perturb_methods))

        bi = torch.arange(image.shape[0])
        intra_loss = (perturb_logits[bi, label] - logits[bi, label]).relu().mean()
        inter_loss = -(perturb_logits.mean(1) - torch.logsumexp(perturb_logits, dim=1)).mean()

        return intra_loss + inter_loss
    

@TRAINER_REGISTRY.register()
class CATEX(TrainerX):
    """Context Optimization (CATEX).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.TRAINER.OOD_TRAIN or self.is_large_ID():
            if self.is_large_ID():
                nsample = 1200 # 1200
                self.id_pool = IDFeatPool(self.model.prompt_learner.n_cls, nsample, self.model.feat_dim, mode='npos', device='cuda:0')
                if cfg.TRAINER.ID_FEAT_PRELOAD != '':
                    queue = torch.load(cfg.TRAINER.ID_FEAT_PRELOAD).to(self.id_pool.queue.device)
                    self.id_pool.queue = queue[:, :nsample, :]
                    self.id_pool.class_ptr += nsample
            else:
                from torch.utils.data import DataLoader, Subset
                from ood.datasets import TinyImages, InfiniteDataLoader

                assert 'cifar' in self.dm.dataset.dataset_name
                data_root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
                ood_set = TinyImages(data_root, transform=self.train_loader_x.dataset.transform)
                self.ood_loader = InfiniteDataLoader(ood_set, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, 
                                                    shuffle=False, num_workers=self.train_loader_x.num_workers,
                                                    pin_memory=True)  # drop_last=True, 
            
            from ood.losses import LogitNormLoss
            self.ce_criterion = LogitNormLoss() if cfg.TRAINER.LOGIT_NORM else nn.CrossEntropyLoss()

    def is_large_ID(self):
        # return True
        return 'imagenet' in self.dm.dataset.dataset_name

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CATEX.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.CATEX.PREC == "fp32" or cfg.TRAINER.CATEX.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner/image_encoder to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CATEX.PREC == "amp" else None

        # # Note that multi-gpu training could be slow because CLIP's size is
        # # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """

        super().build_data_loader()

        if self.cfg.TRAINER.FEAT_AS_INPUT and not self.cfg.TRAINER.OOD_TEST:
            from ood.datasets import CLIPFeatDataset

            self.load_shuffle = False
            self.train_loader_x = torch.utils.data.DataLoader(
                CLIPFeatDataset(self.dm.dataset.dataset_dir+'/clip_feat', self.start_epoch), 
                batch_size=self.dm.train_loader_x.batch_size, shuffle=self.load_shuffle,
                num_workers=self.dm.train_loader_x.num_workers, pin_memory=True, drop_last=False, 
            )

    def calc_loss(self, logits, label, image=None, image_features=None, text_features=None, return_norm=False):
        nb, ncls = logits.shape

        # 1. classification
        if self.cfg.TRAINER.OOD_PROMPT and self.epoch >= self.cfg.TRAINER.START_EPOCH \
            and self.is_large_ID() and self.id_pool.ready():

            if not return_norm:
                image_features = F.normalize(image_features, p=2, dim=1)
            ood_text_features = self.model.get_text_features(ood_prompt=True)

            if self.cfg.TRAINER.OOD_PROMPT_CE_LOSS:
                logits = self.model.get_logits(image_features, torch.cat((text_features, ood_text_features)))
                logits[torch.arange(nb), label+ncls] = -10.  # generally -inf

        loss = 2. * self.ce_criterion(logits, label)

        # 2. prompt perturbation
        perturbed_text_features = None
        if self.cfg.TRAINER.OOD_PROMPT and self.cfg.TRAINER.ID_PERTURB_LOSS and self.epoch >= self.cfg.TRAINER.START_EPOCH:
            with torch.no_grad():
                perturbed_text_features = self.model.get_text_features(perturb=random.choice(perturb_methods))
            loss += 0.1 * self.model.calc_prompt_loss(text_features, perturbed_text_features)

        # 3. outlier exposure
        assert text_features is not None
        if self.is_large_ID():
            if self.id_pool.ready() and self.cfg.TRAINER.OOD_PROMPT and self.epoch >= self.cfg.TRAINER.START_EPOCH:
                if logits.size(0) < self.id_pool.queue.size(0):
                    cls_mask = torch.unique(label).cpu()
                else:
                    cls_mask = None

                if self.cfg.TRAINER.OOD_ANCHOR:
                    if self.cfg.TRAINER.ID_PERTURB_LOSS and False:
                        perturbed_text_features = self.model.get_text_features(perturb=random.choice(perturb_methods))

                        logit_scale = self.model.logit_scale.exp()
                        id_pos_sim = (image_features * text_features[label]).sum(dim=-1) * logit_scale
                        id_neg_sim = (image_features * perturbed_text_features[label]).sum(dim=-1) * logit_scale
                        loss += F.cross_entropy(torch.stack((id_pos_sim, id_neg_sim), dim=1), 
                                                torch.zeros((len(id_pos_sim),), dtype=torch.long, device=self.device)) * 0.5
                    elif perturbed_text_features is None:
                        with torch.no_grad():
                            perturbed_text_features = self.model.get_text_features(perturb=random.choice(perturb_methods))
                    text_anchors = torch.stack((text_features, perturbed_text_features), dim=1).detach()
                else:
                    text_anchors = None
                ood_features, ood_labels = self.id_pool.gen_ood(anchors=text_anchors, device=self.device, cls_mask=cls_mask)

                if self.cfg.TRAINER.OOD_OE_LOSS:
                    ood_logits = self.model.get_logits(ood_features, text_features, logit_scale=1.)
                    loss += 0.5 * -(ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1)).mean()

                if self.cfg.TRAINER.OOD_PROMPT:
                    # ood_text_features = self.model.get_text_features(ood_prompt=True)

                    if self.cfg.TRAINER.OOD_PROMPT_ORTH:
                        assert self.cfg.TRAINER.OOD_PROMPT_NUM > 1
                        all_ood_text_features = self.model.get_all_ood_text_features()
                        # (1000,5,512) x (1000,512,5) -> (1000,5,5)
                        ood_sim_matrix = torch.bmm(all_ood_text_features, all_ood_text_features.transpose(1,2))
                        ood_text_num = ood_sim_matrix.shape[-1]
                        zrange = torch.arange(ood_text_num)
                        ood_sim_matrix[:, zrange, zrange] = 0.
                        loss += 0.1 * ood_sim_matrix.mean()

                    if self.cfg.TRAINER.OOD_PROMPT_CE_LOSS:
                        ood_logits = self.model.get_logits(ood_features,
                                                        torch.cat((ood_text_features, text_features)))
                        ood_logits[torch.arange(ood_logits.shape[0]), ood_labels+ncls] = -10.  # generally -inf
                        loss += 0.5 * self.ce_criterion(ood_logits, ood_labels)
                        
                    if self.cfg.TRAINER.OOD_PROMPT_MARGIN_LOSS:
                        if self.cfg.TRAINER.OOD_PROMPT_MARGIN_SOFT_LOSS:
                            logit_scale = self.model.logit_scale.exp()
                        else:
                            logit_scale = 1.

                        id_pos_sim = (image_features * text_features[label]).sum(dim=-1) * logit_scale
                        id_neg_sim = (image_features * ood_text_features[label]).sum(dim=-1) * logit_scale
                        ood_pos_sim = (ood_features * ood_text_features[ood_labels]).sum(dim=-1) * logit_scale
                        ood_neg_sim = (ood_features * text_features[ood_labels]).sum(dim=-1) * logit_scale

                        # id_pos_sim = (image_features @ text_features.T).max(dim=-1)[0] * logit_scale
                        # id_neg_sim = (image_features @ ood_text_features.T).max(dim=-1)[0] * logit_scale
                        # ood_pos_sim = (ood_features @ ood_text_features.T).max(dim=-1)[0] * logit_scale
                        # ood_neg_sim = (ood_features @ text_features.T).max(dim=-1)[0] * logit_scale

                        if self.cfg.TRAINER.OOD_PROMPT_MARGIN_SOFT_LOSS:
                            loss += F.cross_entropy(torch.stack((id_pos_sim, id_neg_sim), dim=1), 
                                                    torch.zeros((len(id_pos_sim),), dtype=torch.long, device=self.device)) + \
                                    F.cross_entropy(torch.stack((ood_pos_sim, ood_neg_sim), dim=1), 
                                                    torch.zeros((len(ood_pos_sim),), dtype=torch.long, device=self.device))
                        else:
                            loss += (id_neg_sim - id_pos_sim).relu().mean() + (ood_neg_sim - ood_pos_sim).relu().mean()

        else:
            ood_data, _ = next(self.ood_loader.__iter__())
            ood_data = ood_data.to(self.device)
            ood_logits = self.model(ood_data) #/ self.model.logit_scale.exp()
            loss += 0.1 * -(ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1)).mean()

        return loss

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.CATEX.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def forward_backward_ood(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.CATEX.PREC
        return_norm = False
        if prec == "amp":
            with autocast():
                img_feat, text_feat, output = \
                    self.model(image, return_feat=True, return_norm=return_norm)
                self.id_pool.update(img_feat.detach(), label)
                loss = self.calc_loss(output, label, image, img_feat, text_feat, return_norm=return_norm)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            img_feat, text_feat, output = \
                self.model(image, return_feat=True, return_norm=return_norm)
            self.id_pool.update(img_feat.detach(), label)
            loss = self.calc_loss(output, label, image, img_feat, text_feat, return_norm=return_norm)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            model_dict = self._models[name].state_dict()
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            assert all(k in model_dict for k in state_dict)

            print("Loading weights to {} {} " 'from "{}" (epoch = {})'.format(name, list(state_dict.keys()), model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

        if self.cfg.TRAINER.CATEX.CTX_INIT:
            assert self.cfg.TRAINER.CATEX.CTX_INIT == 'ensemble_learned'
            text_feature = self.model.get_text_features()
            self.model.text_feature_ensemble = self.model.prompt_ensemble(text_feature)

    def load_model_vanilla(self, directory, name, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = osp.join(directory, name, model_file)

        if not osp.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        print("Loading vanilla weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
        # set strict=False
        model = self.model.module if torch.cuda.device_count() > 1 else self.model    
        getattr(model, name).load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        if self.cfg.TRAINER.FEAT_AS_INPUT:
            if not self.load_shuffle:
                self.train_loader_x.dataset.load_data(self.epoch)

    @torch.no_grad()
    def test_ood(self, split=None, model_directory=''):
        """A generic OOD testing pipeline."""
        from tqdm import tqdm
        import os
        import os.path as osp
        from torch.utils.data import DataLoader
        import numpy as np

        from ood.datasets import CLIPFeatDataset
        from ood.datasets import SCOODDataset, LargeOODDataset, SemanticOODDataset, ClassOODDataset
        from ood.metrics import get_msp_scores, get_measures

        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if self.cfg.TRAINER.FEAT_AS_INPUT:
            feat_data_dir = self.dm.dataset.dataset_dir+'/clip_feat'
            if not osp.exists(feat_data_dir):
                self.cache_feat(split=split, is_ood=False)

            data_loader = DataLoader(
                    CLIPFeatDataset(feat_data_dir, self.start_epoch, split='test'), 
                    batch_size=self.test_loader.batch_size, shuffle=False,
                    num_workers=self.test_loader.num_workers, pin_memory=True, drop_last=False, 
                )
        else:
            if split == "val" and self.val_loader is not None:
                data_loader = self.val_loader
            else:
                split = "test"  # in case val_loader is None
                data_loader = self.test_loader
        lab2cname = self.dm.dataset.lab2cname

        ood_cfg = {
            'SCOOD': ['texture', 'svhn', 'cifar', 'tin', 'lsun', 'places365'],
            'LargeOOD': ['inaturalist', 'sun', 'places', 'texture'],
        }
        data_root = osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT))
        ood_type = 'SCOOD' if 'cifar' in self.dm.dataset.dataset_name else 'LargeOOD' # LargeOOD, ClassOOD
        if 'apply_' in self.cfg.TRAINER.OOD_INFER_OPTION:
            posthoc = self.cfg.TRAINER.OOD_INFER_OPTION
        else:
            posthoc = None

        if self.cfg.TRAINER.OOD_PROMPT:
            if self.cfg.TRAINER.OOD_PROMPT_NUM > 1:
                ood_text_features = torch.stack([self.model.get_text_features(ood_prompt=True, ood_prompt_idx=i) for i in range(self.cfg.TRAINER.OOD_PROMPT_NUM)])
            else:
                ood_text_features = self.model.get_text_features(ood_prompt=True)
                if self.cfg.TRAINER.CATEX.CTX_INIT:
                    assert self.cfg.TRAINER.CATEX.CTX_INIT == 'ensemble_learned'
                    ood_text_features = self.model.prompt_ensemble(ood_text_features)
            min_thresh = 0.51 if any(flag in model_directory for flag in ['/imagenet/', '/imagenet100-MCM-SCTX8-Orth/']) else 0.5

        self.model.text_feature_ensemble = self.model.get_text_features()

        print(f"Evaluate on the *{split}* set")

        if self.cfg.TRAINER.OOD_INFER_OPTION == 'save_res':
            save_dir = f'{model_directory}/restore'
            os.makedirs(save_dir, exist_ok=True)
            with open(f'{save_dir}/lab2cname.json', 'w+') as f:
                json.dump(lab2cname, f, indent=4)

            text_features = self.model.get_text_features()
            torch.save(text_features.cpu(), f'{save_dir}/in_text_features.pt')
            if self.cfg.TRAINER.OOD_PROMPT:
                torch.save(ood_text_features.cpu(), f'{save_dir}/ood_text_features.pt')

            im_feats, im_labels = [], []

        if self.cfg.TRAINER.OOD_INFER_OPTION == 'resume_res':
            resume_dir = 'weights/imagenet100-MCM/CATEX/vit_b16_ep50_-1shots/nctx16_cscTrue_ctpend/seed1/restore'
            resume_image_features = torch.load(f'{resume_dir}/in_image_features.pt').to(self.device)
            resume_image_labels = torch.load(f'{resume_dir}/in_labels.pt').to(self.device)
            resume_text_features = torch.load(f'{resume_dir}/in_text_features.pt').to(self.device)
            if self.cfg.TRAINER.OOD_PROMPT:
                resume_ood_text_features = torch.load(f'{resume_dir}/ood_text_features.pt').to(self.device)
            with open(f'{resume_dir}/lab2cname.json', 'r') as f:
                resume_lab2cname = json.load(f)
            resume_lab2cname = {int(k): v for k, v in resume_lab2cname.items()}

            label_offset = resume_image_labels.max().item() + 1
            resume_image_labels += label_offset

            text_features = self.model.get_text_features()
            merged_text_features = torch.cat((text_features, resume_text_features), dim=0)
            if self.cfg.TRAINER.OOD_PROMPT:  # TODO: not implemented
                merged_ood_text_features = torch.cat((ood_text_features, resume_ood_text_features), dim=0)

        score_list = []
        base_acc, novel_acc = [], []
        near_ood_flag = []
        all_logits = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            image_features, _, output = \
                self.model(input, return_feat=True, return_norm=True, posthoc=posthoc)
            
            if self.cfg.TRAINER.OOD_PROMPT:
                ood_logits = self.model.get_logits(image_features, ood_text_features, logit_scale=1.)
                # all_logits.append(torch.stack((output, ood_logits), dim=1))

                if self.cfg.TRAINER.OOD_INFER_INTEGRATE:
                    id_score = F.softmax(torch.stack((output, ood_logits), dim=1), dim=1)[:, 0, :]
                    output *= id_score.clamp(min=min_thresh)
            else:
                ood_logits = None

            if self.cfg.TRAINER.OOD_INFER_OPTION == 'save_res':
                im_feats.append(image_features.cpu())
                im_labels.append(label.cpu())

            if self.cfg.TRAINER.OOD_INFER_OPTION == 'resume_res':
                start = data_loader.batch_size * batch_idx
                end = start + input.shape[0]
                merged_image_features = torch.cat((image_features, resume_image_features[start:end]), dim=0)
                label = torch.cat((label, resume_image_labels[start:end]), dim=0)

                output = merged_image_features @ merged_text_features.t()
                acc = output.argmax(dim=1) == label
                base_acc.append(acc[input.shape[0]:].cpu())
                novel_acc.append(acc[:input.shape[0]].cpu())

            if hasattr(self.dm.dataset, 'valid_classes'):
                # if self.cfg.TRAINER.OOD_PROMPT:
                #     raise NotImplementedError
                output[:, ~self.dm.dataset.valid_classes] = -1.
                scores = get_msp_scores(output[:, self.dm.dataset.valid_classes])
            else:
                scores, ood_flag = get_msp_scores(output, ood_logits, self.cfg.TRAINER.OOD_INFER, ret_near_ood=True)
                near_ood_flag.append(ood_flag)
            score_list.append(scores.detach().cpu().numpy())
            self.evaluator.process(output, label)

        in_scores = np.concatenate(score_list, axis=0)
        results = self.evaluator.evaluate()
        if self.cfg.TRAINER.OOD_PROMPT and len(near_ood_flag) and near_ood_flag[0] is not None:
            print('NearOOD FPR:', torch.cat(near_ood_flag).sum().item() / len(in_scores))

        if self.cfg.TRAINER.OOD_INFER_OPTION == 'save_res':
            torch.save(torch.cat(im_feats), f'{save_dir}/in_image_features.pt')
            torch.save(torch.cat(im_labels), f'{save_dir}/in_labels.pt')
            if len(all_logits):
                torch.save(torch.cat(all_logits), f'{save_dir}/in_logits_all.pt')
        
        if self.cfg.TRAINER.OOD_INFER_OPTION == 'resume_res':
            print(f'Base: {torch.cat(base_acc).float().mean(): .4f}. Novel: {torch.cat(novel_acc).float().mean(): .4f}')

        auroc_list, aupr_list, fpr95_list = [], [], []
        ood_tpr_list = []
        save_lines = []
        for ood_name in ood_cfg[ood_type]:
            ood_set = eval(f'{ood_type}Dataset')(osp.join(data_root, ood_type), id_name=self.dm.dataset.dataset_name, 
                                    ood_name=ood_name, transform=self.test_loader.dataset.transform)
            if self.cfg.TRAINER.FEAT_AS_INPUT:
                feat_data_dir = f'{data_root}/{ood_type}/clip_feat/{ood_name}'
                if not osp.exists(feat_data_dir):
                    self.cache_feat(split='test', is_ood=True)

                ood_loader = DataLoader(
                        CLIPFeatDataset(feat_data_dir, epoch=None, split='test'), 
                        batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False,
                        num_workers=data_loader.num_workers, pin_memory=True, drop_last=False, 
                    )
            else:
                ood_loader = DataLoader(ood_set, batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, num_workers=data_loader.num_workers,
                                            drop_last=False, pin_memory=True)
            
            ood_score_list, sc_labels_list, ood_pred_list = [], [], []
            near_ood_flag = []
            all_logits = []
            for batch_idx, batch in enumerate(tqdm(ood_loader)):
                if self.cfg.TRAINER.FEAT_AS_INPUT:
                    images, sc_labels = self.parse_batch_test(batch)
                else:
                    images, sc_labels = batch
                    images = images.to(self.device)

                image_features, _, output = \
                    self.model(images, return_feat=True, return_norm=True, posthoc=posthoc)
            
                if self.cfg.TRAINER.OOD_PROMPT:
                    ood_logits = self.model.get_logits(image_features, ood_text_features, logit_scale=1.)
                    # all_logits.append(torch.stack((output, ood_logits), dim=1))

                    if self.cfg.TRAINER.OOD_INFER_INTEGRATE:
                        id_score = F.softmax(torch.stack((output, ood_logits), dim=1), dim=1)[:, 0, :]
                        output *= id_score.clamp(min=min_thresh)
                else:
                    ood_logits = None

                if self.cfg.TRAINER.OOD_INFER_OPTION == 'resume_res':
                    output = image_features @ merged_text_features.t()

                if hasattr(self.dm.dataset, 'valid_classes'):
                    output[:, ~self.dm.dataset.valid_classes] = -1.
                    scores = get_msp_scores(output[:, self.dm.dataset.valid_classes])
                else:
                    scores, ood_flag = get_msp_scores(output, ood_logits, self.cfg.TRAINER.OOD_INFER, ret_near_ood=True)
                    near_ood_flag.append(ood_flag)
                ood_score_list.append(scores.detach().cpu().numpy())
                sc_labels_list.append(sc_labels.cpu().numpy())
                ood_pred_list.append(output.argmax(dim=1).cpu().numpy())
            ood_scores = np.concatenate(ood_score_list, axis=0)
            sc_labels = np.concatenate(sc_labels_list, axis=0)
            ood_preds = np.concatenate(ood_pred_list, axis=0)
            fake_ood_scores = ood_scores[sc_labels>=0]
            real_ood_scores = ood_scores[sc_labels<0]
            real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)

            if 'cifar' in self.dm.dataset.dataset_name:  
                # compatible with SCOOD
                auroc, aupr, fpr95, thresh = get_measures(real_ood_scores, real_in_scores)
            else:  
                # compatible with NPOS
                auroc, aupr, fpr95, thresh = get_measures(-real_in_scores, -real_ood_scores)
            print('auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95))

            save_lines.append('%10s auroc: %.4f, aupr: %.4f, fpr95: %.4f\n' % (ood_name, auroc, aupr, fpr95))
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            fpr95_list.append(fpr95)
            if self.cfg.TRAINER.OOD_PROMPT and len(near_ood_flag) and near_ood_flag[0] is not None:
                ood_tpr = torch.cat(near_ood_flag).sum().item() / len(ood_scores)
                print('NearOOD TPR: %.4f' % ood_tpr)
                ood_tpr_list.append(ood_tpr)
            
            if self.cfg.TRAINER.OOD_INFER_OPTION == 'save_res' and len(all_logits):
                torch.save(torch.cat(all_logits), f'{save_dir}/ood_{ood_name}_logits_all.pt')

        print('\nAverage: auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr95_list)))
        save_lines.append('%10s auroc: %.4f, aupr: %.4f, fpr95: %.4f\n' % ('nAverage', np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr95_list)))
        if self.cfg.TRAINER.OOD_PROMPT and len(ood_tpr_list) > 1:
            print('Average: OOD-TPR: %.4f' % np.mean(ood_tpr_list))

        if model_directory != '':
            if 'ClassOOD' == ood_type:
                res_list = np.stack((auroc_list, aupr_list, fpr95_list), axis=1).reshape(-1,) * 100
                np.savetxt(f'{model_directory}/{ood_type}_results.csv', res_list, fmt='%.2f', delimiter=',')
            save_path = f'{model_directory}/{ood_type}_results.txt'
            with open(save_path, 'w+') as f:
                f.writelines(save_lines)

        return list(results.values())[0], auroc, aupr, fpr95
    
    @torch.no_grad()
    def cache_feat(self, split='train', is_ood=True):
        """A generic OOD testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()
        if split == 'train':
            data_loader = self.train_loader_x  
            max_epoch = self.max_epoch
        else:
            data_loader = self.test_loader
            max_epoch = 1

        if is_ood:
            from ood.datasets import LargeOODDataset
            from torch.utils.data import DataLoader
            data_root = osp.join(osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT)), 'LargeOOD')
            for ood_name in ['inaturalist', 'sun', 'places', 'texture']:
                ood_set = LargeOODDataset(data_root, id_name=self.dm.dataset.dataset_name, 
                                            ood_name=ood_name, transform=self.test_loader.dataset.transform)
                ood_loader = DataLoader(ood_set, batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, num_workers=self.test_loader.num_workers,
                                            drop_last=False, pin_memory=True)

                save_dir = f'{data_root}/clip_feat/{ood_name}'
                os.makedirs(save_dir, exist_ok=True)

                features, labels, paths = [], [], []
                cnt = 0
                for input, label in tqdm(ood_loader, desc='Caching image features'):
                    input = input.to(self.device)
                    label = label.to(self.device)

                    image_features = self.model.image_encoder(input.type(self.model.dtype)).detach()

                    features.append(image_features.cpu())
                    labels.append(label.cpu())
                    for i in range(len(input)):
                        paths.append(ood_set.samples[cnt+i][0])
                    cnt += len(input)

                torch.save(torch.cat(features).half(), f'{save_dir}/test_image_features.pt')
                torch.save(torch.cat(labels).half(), f'{save_dir}/test_labels.pt')
                with open(f'{save_dir}/test_paths.txt', 'w+') as f:
                    f.writelines([p + '\n' for p in paths])
        else:
            save_dir = f'{self.dm.dataset.dataset_dir}/clip_feat'
            os.makedirs(save_dir, exist_ok=True)

            for self.epoch in range(self.start_epoch, max_epoch):
                features, labels, paths = [], [], []
                for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Caching image features: {split} {self.epoch+1}/{max_epoch}: ")):
                    input = batch["img"].to(self.device)
                    label = batch["label"].to(self.device)

                    image_features = self.model.image_encoder(input.type(self.model.dtype)).detach()

                    features.append(image_features.cpu())
                    labels.append(label.cpu())
                    paths.extend(batch["impath"])

                torch.save(torch.cat(features).half(), f'{save_dir}/ep{self.epoch}_{split}_image_features.pt')
                torch.save(torch.cat(labels).half(), f'{save_dir}/ep{self.epoch}_{split}_labels.pt')
                with open(f'{save_dir}/ep{self.epoch}_{split}_paths.txt', 'w+') as f:
                    f.writelines([p + '\n' for p in paths])