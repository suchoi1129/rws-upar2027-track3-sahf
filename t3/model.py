"""ViTPose wrappers (HF transformers). Forward returns heatmaps [B,17,64,48] for the challenge slot order."""
import torch
from transformers import VitPoseForPoseEstimation


class PoseNet(torch.nn.Module):
    def __init__(self, name, dataset_index=0, in_hw=None):
        super().__init__()
        self.net = VitPoseForPoseEstimation.from_pretrained(name)
        self.moe = getattr(self.net.config.backbone_config, "num_experts", 1) > 1
        self.register_buffer("dataset_index", torch.tensor([dataset_index], dtype=torch.int64), persistent=False)
        if in_hw is not None:
            self.resize_input(*in_hw)

    def resize_input(self, h, w):
        """Interpolate the patch position grid to a new input size (e.g. 384x288)."""
        cfg = self.net.config.backbone_config
        old = tuple(cfg.image_size) if not isinstance(cfg.image_size, int) else (cfg.image_size, cfg.image_size)
        if (h, w) == old:
            return
        ps = cfg.patch_size if not isinstance(cfg.patch_size, int) else (cfg.patch_size, cfg.patch_size)
        emb = self.net.backbone.embeddings
        pe = emb.position_embeddings.data                                  # [1, 1+N, C]
        gh, gw = old[0] // ps[0], old[1] // ps[1]
        grid = pe[:, 1:].reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(grid, size=(h // ps[0], w // ps[1]), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, -1, pe.shape[-1])
        emb.position_embeddings = torch.nn.Parameter(torch.cat([pe[:, :1], grid], 1))
        pt = emb.patch_embeddings
        pt.image_size = (h, w); pt.num_patches = (h // ps[0]) * (w // ps[1])
        cfg.image_size = [h, w]

    def forward(self, x):
        if self.moe:
            return self.net(pixel_values=x, dataset_index=self.dataset_index.expand(x.shape[0])).heatmaps
        return self.net(pixel_values=x).heatmaps

    def forward_train(self, x):
        """Heatmaps + mean-pooled backbone tokens (for the training-only attribute head)."""
        kw = dict(dataset_index=self.dataset_index.expand(x.shape[0])) if self.moe else {}
        seq = self.net.backbone.forward_with_filtered_kwargs(x, **kw).feature_maps[-1]      # [B, N, C]
        cfg = self.net.config.backbone_config
        ph, pw = cfg.image_size[0] // cfg.patch_size[0], cfg.image_size[1] // cfg.patch_size[1]
        fmap = seq.permute(0, 2, 1).reshape(x.shape[0], -1, ph, pw).contiguous()
        return self.net.head(fmap), seq.mean(1)


class Ensemble(torch.nn.Module):
    """Average heatmaps of several PoseNets that share the input size (traced as one module)."""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        w = [1.0] * len(models) if weights is None else [float(v) for v in weights]
        self.weights = [v / sum(w) for v in w]                   # python floats -> constants when traced

    def forward(self, x):
        out = self.models[0](x) * self.weights[0]
        for m, w in zip(self.models[1:], self.weights[1:]):
            out = out + m(x) * w
        return out


def param_groups(model, lr, wd, layer_decay):
    """Layer-wise lr decay over ViT blocks; head gets full lr."""
    n_layers = model.net.config.backbone_config.num_hidden_layers
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".embeddings." in n:
            lid = 0
        elif ".encoder.layer." in n:
            lid = int(n.split(".encoder.layer.")[1].split(".")[0]) + 1
        elif "backbone" in n:
            lid = n_layers
        else:
            lid = n_layers + 1
        scale = layer_decay ** (n_layers + 1 - lid)
        no_wd = p.ndim == 1 or n.endswith(".bias") or "position_embeddings" in n or "cls_token" in n
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], lr=lr * scale, base_lr=lr * scale, weight_decay=0.0 if no_wd else wd)
        groups[key]["params"].append(p)
    return list(groups.values())
