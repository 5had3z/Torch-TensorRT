"""TensorRT: FP8 Q/DQ on the attention output fused into the P.V matmul gives wrong results for some
windows of a chunked prefill (Qwen3-VL attention block with a dynamic key length).

What it does
------------
One attention block (input RMSNorm, q/k/v projections, per-head q/k RMSNorm, rotary, causal GQA SDPA,
``o_proj``, residual) with ModelOpt FP8 quantizers on the four linears, taken from layer 14 of the
Alpamayo 1.5 FP8 decoder. It is compiled by Torch-TensorRT with ``static_cache_v2`` (KV cache inputs,
``start_idx``/``end_idx`` scalars, dynamic key length) with a dynamic batch (1..6) and sequence (1..64),
and run as chunked prefill at batch 1, 64 tokens per call, on the 3086-token prompt the layer saw. Every chunk is compared with the eager (fake-quant) block.

Expected: every chunk matches (relative error ~1e-2 or less per token, the FP8 grid noise).
Observed:  chunk 2240..2303 has ~38% error per token (the other 48 chunks are fine). The engine is
wrong only with the FP8 quantizer on the ``o_proj`` input (the attention output): TensorRT then fuses
the Q into the epilogue of the P.V GEMM (``sm80_xmma_gemm_f16f16_f16f16_f16_nn_..._by_fusion``).
``--control`` disables that quantizer: the GEMM is not fused and all chunks match.

The result depends on the data and on the dynamic batch dimension: random weights or random
activations of the same shapes do not fail, and neither does a static batch, so the real tensors are
needed (``make-pack``, ~110 MB).

Usage
-----
    # 1. once, needs the alpamayo1_5 package, the FP8 checkpoint and the dataset:
    python repro_fp8_attn_out_fusion.py make-pack --model <fp8 checkpoint dir> --parquet <clips.parquet> \
        --out fusion_pack.pt
    # 2. the repro (needs torch, nvidia-modelopt, torch-tensorrt; no checkpoint):
    python repro_fp8_attn_out_fusion.py run --pack fusion_pack.pt            # fails
    python repro_fp8_attn_out_fusion.py run --pack fusion_pack.pt --control  # passes

``--torchtrt_tools`` is the ``tools/llm`` directory of a Torch-TensorRT checkout (``static_cache_v2``
and its SDPA converter live there; branch ``alpamayo-vlm-export``).
"""

import argparse
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TOOLS = os.path.expanduser("~/dev/Torch-TensorRT/tools/llm")
CACHE_LEN, CHUNK, LAYER, MAX_BATCH = 3456, 64, 14, 6


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------
def rms_norm(x, weight, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-b, a), dim=-1)


class Attention(nn.Module):
    """x + o_proj(sdpa(rope(q_norm(q)), rope(k_norm(k)), v)) on the pre-normed x."""

    def __init__(self, pack):
        super().__init__()
        self.heads, self.kv_heads, self.hd, self.eps = pack["heads"], pack["kv_heads"], pack["head_dim"], pack["eps"]
        dim = pack["q"]["weight"].shape[1]
        self.names = ("q", "k", "v", "o")
        for n in self.names:
            w = pack[n]["weight"]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight.data = w.clone()
            setattr(self, f"{n}_proj", lin)
        for n in ("input_norm", "q_norm", "k_norm"):
            self.register_buffer(n, pack[n].clone())
        assert dim == pack["x"].shape[-1]

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = rms_norm(x, self.input_norm, self.eps)
        q = rms_norm(self.q_proj(h).view(B, T, -1, self.hd), self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm(self.k_proj(h).view(B, T, -1, self.hd), self.k_norm, self.eps).transpose(1, 2)
        v = self.v_proj(h).view(B, T, -1, self.hd).transpose(1, 2)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return x + self.o_proj(a.transpose(1, 2).reshape(B, T, self.heads * self.hd))


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(args):
    sys.path.insert(0, args.torchtrt_tools)
    import modelopt.torch.quantization as mtq
    import torch_tensorrt
    from modelopt.torch.quantization.utils import export_torch_mode
    from torchtrt_ext import register_sdpa

    import static_cache_v2

    print("torch", torch.__version__, "| torch_tensorrt", torch_tensorrt.__version__, "| gpu", torch.cuda.get_device_name())
    pack = torch.load(args.pack)
    x = pack["x"].cuda().half()  # [1, L, H], the input of layer 14
    cos, sin = pack["cos"].cuda().half(), pack["sin"].cuda().half()  # [1, L, 128]
    L, H = x.shape[1], x.shape[2]

    m = Attention(pack).cuda().half().eval()
    mtq.quantize(m, mtq.FP8_DEFAULT_CFG, forward_loop=lambda mod: mod(x[:, :64], cos[:, :64], sin[:, :64]))
    for n in m.names:  # the calibrated scales of the real model
        lin = getattr(m, f"{n}_proj")
        lin.input_quantizer.amax = pack[n]["input_amax"].cuda().half()
        lin.weight_quantizer.amax = pack[n]["weight_amax"].cuda().half()
    if args.control:
        m.o_proj.input_quantizer.disable()
    print("o_proj input quantizer:", "DISABLED (control)" if args.control else "FP8 (amax %.4f)" % m.o_proj.input_quantizer.amax.item())
    with torch.no_grad():
        ref = m(x, cos, sin)

    register_sdpa.enable_sdpa_converter("default", None)
    static_cache_v2.KV_CACHE_LEN = CACHE_LEN
    # dynamic batch (1..6) and sequence (1..64), as in the decoder engine that serves 6 samples
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    seq = torch.export.Dim("seq", min=1, max=CHUNK)
    ex = tuple(t[:, :8].contiguous().expand(2, -1, -1).contiguous() for t in (x, cos, sin))
    dyn = {0: batch, 1: seq}
    with torch.no_grad(), export_torch_mode():
        ep = torch.export.export(m, ex, dynamic_shapes=(dyn, dyn, dyn), strict=False)
    inp = lambda d: torch_tensorrt.Input(  # noqa: E731
        min_shape=(1, 1, d), opt_shape=(MAX_BATCH, 1, d), max_shape=(MAX_BATCH, CHUNK, d), dtype=torch.half
    )
    with torch.no_grad():
        trt = torch_tensorrt.dynamo.compile(
            ep, inputs=[inp(H), inp(pack["head_dim"]), inp(pack["head_dim"])],
            enabled_precisions={torch.half}, use_explicit_typing=False, min_block_size=1, disable_tf32=True,
        )

    caches = [torch.zeros(1, pack["kv_heads"], CACHE_LEN, pack["head_dim"], device="cuda", dtype=torch.half) for _ in range(2)]
    errs = []
    with torch.no_grad():
        for s in range(0, L, CHUNK):
            e = min(s + CHUNK, L)
            out = trt(x[:, s:e], cos[:, s:e], sin[:, s:e], *caches, s, e)
            caches = list(out[1:])
            r = ref[:, s:e].float()
            errs.append(((out[0].float() - r).norm(dim=-1) / r.norm(dim=-1)).mean().item())
    bad = [(i * CHUNK, min((i + 1) * CHUNK, L), round(v, 3)) for i, v in enumerate(errs) if v > 0.05]
    print(f"chunks: {len(errs)}, median relerr {sorted(errs)[len(errs) // 2]:.4f}, worst {max(errs):.3f}")
    print("BAD chunks (start, end, mean per-token relerr):", bad if bad else "none")
    print("RESULT:", "FAIL (engine output is wrong for the chunks above)" if bad else "PASS")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------
# make-pack: needs the alpamayo1_5 package, the FP8 checkpoint and the dataset
# --------------------------------------------------------------------------------------
def make_pack(args):
    sys.path.insert(0, args.torchtrt_tools)
    import run_vla as rv  # Torch-TensorRT tools/llm/run_vla.py

    a = types.SimpleNamespace(clip_id=None, parquet=args.parquet, clip_index=0, t0_us=5_100_100)
    model = rv.load_model(args.model).to(rv.DEVICE)
    _, tok = rv.load_inputs(model, a)
    ids = tok["input_ids"].to(rv.DEVICE)
    text = model.vlm.model.language_model
    with torch.no_grad():
        pv = tok["pixel_values"].to(rv.DEVICE, torch.float16)
        grid = tok["image_grid_thw"].to(rv.DEVICE)
        img, ds = model.vlm.model.visual(pv, grid)
        mask = ids == model.vlm.config.image_token_id
        emb = text.embed_tokens(ids)
        x = emb.masked_scatter(mask[..., None].expand_as(emb), img.to(emb.dtype)).half()
        dst = torch.zeros(len(ds), 1, ids.shape[1], x.shape[-1], device=rv.DEVICE, dtype=torch.half)
        for i, d in enumerate(ds):
            dst[i, 0, mask[0]] = d.half()
        pos, _ = model.vlm.model.get_rope_index(ids, grid, None, tok["attention_mask"].to(rv.DEVICE))
    del model
    # the first LAYER layers with the exported (fake-quant, fp16) weights, as the engine sees them
    seg_args = types.SimpleNamespace(model=args.model, num_traj_samples=6, chunk_len=64, opt_seq_len=1,
                                     max_seq_len=CACHE_LEN, fp8_attention_output_quant=True)
    seg = rv.build_llm_segment(seg_args, (0, LAYER + 1))[0]
    with torch.no_grad():
        pe = seg.rotary_emb(x, pos)
        for i, layer in enumerate(seg.layers[:LAYER]):
            x = layer(x, position_embeddings=pe, attention_mask=None)
            if i < seg.num_deepstack:
                x = x + dst[i]
        layer = seg.layers[LAYER]
        at = layer.self_attn
        cos, sin = pe
    g = lambda mod: {  # noqa: E731
        "weight": mod.weight.detach().half().cpu(),
        "input_amax": mod.input_quantizer.amax.float().cpu(),
        "weight_amax": mod.weight_quantizer.amax.float().cpu(),
    }
    pack = {
        "x": x.half().cpu(), "cos": cos.half().cpu(), "sin": sin.half().cpu(),
        "input_norm": layer.input_layernorm.weight.detach().half().cpu(),
        "q_norm": at.q_norm.weight.detach().half().cpu(), "k_norm": at.k_norm.weight.detach().half().cpu(),
        "eps": layer.input_layernorm.variance_epsilon,
        "q": g(at.q_proj), "k": g(at.k_proj), "v": g(at.v_proj), "o": g(at.o_proj),
        "heads": at.config.num_attention_heads, "kv_heads": at.config.num_key_value_heads, "head_dim": at.head_dim,
    }
    torch.save(pack, args.out)
    print("wrote", args.out, f"({os.path.getsize(args.out) / 1e6:.0f} MB), input {tuple(pack['x'].shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torchtrt_tools", default=os.environ.get("TORCHTRT_TOOLS_LLM", DEFAULT_TOOLS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pack", required=True)
    r.add_argument("--control", action="store_true", help="disable the o_proj input quantizer (passes)")
    mk = sub.add_parser("make-pack")
    mk.add_argument("--model", required=True, help="ModelOpt FP8 Alpamayo 1.5 checkpoint dir")
    mk.add_argument("--parquet", required=True)
    mk.add_argument("--out", default="fusion_pack.pt")
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else make_pack(a) or 0)
