"""Convert original ViTPose (mmpose) checkpoints to HF VitPoseForPoseEstimation folders.
python -m t3.convert_orig <orig.pth> <out_dir> [--verify usyd-community/vitpose-base]
Provenance: public-data/ViTPose models/vitpose-{b,l}.pth (official ViTPose COCO-only 256x192 releases)."""
import argparse, copy, torch
from transformers import VitPoseForPoseEstimation, VitPoseConfig


def convert(sd, num_layers):
    out = {}
    out["backbone.embeddings.position_embeddings"] = sd["backbone.pos_embed"]
    out["backbone.embeddings.patch_embeddings.projection.weight"] = sd["backbone.patch_embed.proj.weight"]
    out["backbone.embeddings.patch_embeddings.projection.bias"] = sd["backbone.patch_embed.proj.bias"]
    for i in range(num_layers):
        o, h = f"backbone.blocks.{i}.", f"backbone.encoder.layer.{i}."
        qw, kw, vw = sd[o + "attn.qkv.weight"].chunk(3, 0); qb, kb, vb = sd[o + "attn.qkv.bias"].chunk(3, 0)
        out.update({h + "attention.attention.query.weight": qw, h + "attention.attention.query.bias": qb,
                    h + "attention.attention.key.weight": kw, h + "attention.attention.key.bias": kb,
                    h + "attention.attention.value.weight": vw, h + "attention.attention.value.bias": vb,
                    h + "attention.output.dense.weight": sd[o + "attn.proj.weight"], h + "attention.output.dense.bias": sd[o + "attn.proj.bias"],
                    h + "mlp.fc1.weight": sd[o + "mlp.fc1.weight"], h + "mlp.fc1.bias": sd[o + "mlp.fc1.bias"],
                    h + "mlp.fc2.weight": sd[o + "mlp.fc2.weight"], h + "mlp.fc2.bias": sd[o + "mlp.fc2.bias"],
                    h + "layernorm_before.weight": sd[o + "norm1.weight"], h + "layernorm_before.bias": sd[o + "norm1.bias"],
                    h + "layernorm_after.weight": sd[o + "norm2.weight"], h + "layernorm_after.bias": sd[o + "norm2.bias"]})
    out["backbone.layernorm.weight"] = sd["backbone.last_norm.weight"]; out["backbone.layernorm.bias"] = sd["backbone.last_norm.bias"]
    hm = {"deconv_layers.0": "deconv1", "deconv_layers.1": "batchnorm1", "deconv_layers.3": "deconv2", "deconv_layers.4": "batchnorm2", "final_layer": "conv"}
    # (simple decoder checkpoints only have keypoint_head.final_layer -> head.conv)
    for k, v in sd.items():
        if k.startswith("keypoint_head."):
            rest = k[len("keypoint_head."):]
            for a, b in hm.items():
                if rest.startswith(a + "."):
                    out["head." + b + rest[len(a):]] = v
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("orig"); ap.add_argument("out"); ap.add_argument("--verify", default="")
    a = ap.parse_args()
    sd = torch.load(a.orig, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
    C = sd["backbone.pos_embed"].shape[-1]; L = len({k.split(".")[2] for k in sd if ".blocks." in k})
    simple = "keypoint_head.deconv_layers.0.weight" not in sd
    base = VitPoseConfig.from_pretrained("usyd-community/vitpose-base-simple" if simple else "usyd-community/vitpose-base")
    cfg = copy.deepcopy(base); old = cfg.backbone_config.to_dict()
    for k in ["out_features", "out_indices", "stage_names", "_out_features", "_out_indices", "model_type", "transformers_version"]:
        old.pop(k, None)
    old.update(hidden_size=C, num_hidden_layers=L, num_attention_heads={768: 12, 1024: 16, 1280: 16}[C], mlp_ratio=4, out_indices=[L])
    cfg.backbone_config = type(cfg.backbone_config)(**old)
    model = VitPoseForPoseEstimation(cfg)
    hf = convert(sd, L)
    missing, unexpected = model.load_state_dict(hf, strict=False)
    print(f"hidden={C} layers={L} missing={missing} unexpected={unexpected}")
    assert not missing and not unexpected
    if a.verify:
        ref = VitPoseForPoseEstimation.from_pretrained(a.verify).state_dict(); mine = model.state_dict()
        diff = max((ref[k].float() - mine[k].float()).abs().max().item() for k in ref)
        print(f"verify vs {a.verify}: max|diff| = {diff}")
    model.save_pretrained(a.out); print("saved", a.out)


if __name__ == "__main__":
    main()
