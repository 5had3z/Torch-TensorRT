# FP8 Alpamayo 1.5 with `run_vla.py`: two TensorRT problems and their workarounds

`run_vla.py` accepts a ModelOpt FP8 checkpoint (a directory with a `modelopt_state.pth`). Bringing the FP8
decoder up exposed two problems in how TensorRT builds engines from the Q/DQ graph that ModelOpt and
Torch-TensorRT produce. Both are worked around in `run_vla.py`; neither is fixed in Torch-TensorRT, and both
look like TensorRT issues (the lowered Torch-TensorRT graph is identical between a failing and a working
build). Measured on an RTX 5090 (sm120), TensorRT 11.4.0.106, one clip (3086-token prompt), engines built in
FP16 with FP8 Q/DQ (`enabled_precisions={torch.float16}`, `use_explicit_typing=False`).

## How the FP8 model is exported

The checkpoint stores weights as `float8_e4m3fn` and dequantizes inside the weight quantizer, which cannot be
traced. `dequantize_real_quant_weights` re-materializes each weight as FP16 (it is already on the FP8 grid) and
switches its quantizer to fake-quant; the model is exported under ModelOpt's `export_torch_mode()`, which
turns every quantizer into a `tensorrt::quantize_op` Q/DQ node. Torch-TensorRT converts these to TensorRT
`IQuantizeLayer`/`IDequantizeLayer` pairs and TensorRT folds the weight Q/DQ into FP8 weights.

The vision tower and the expert use hand-written attention, so they apply the q/k/v quantizers themselves
(`bmm_quantize`; for the expert after the K/V concat, as the Hugging Face layer does). The decoder runs the
unmodified Hugging Face layers, which apply them on their own.

## Problem 1: sibling FP8 GEMMs with identical input scales give an all-NaN engine

**Symptom.** An engine of three or more decoder layers returns NaN for every output. Every FP8 linear works
alone; it takes two linears that read the same activation (`q/k/v_proj`, `gate/up_proj`). This is fixed
in `TensorRT 11.5`.

**Cause.** Sibling linears read the same activation, so ModelOpt calibrates the same input amax for them,
while their weight amax differ. With identical input Q/DQ and different weight scales, TensorRT merges the two
GEMMs into one (`[up]+[gate]`, output `[1, 64, 24576]`, followed by a `MulAdd` kernel on the merged output)
and the engine outputs NaN. With input scales that differ, the GEMMs stay separate and the result is correct;
equal weight scales also work. The Torch-TensorRT graph is the same in all cases (six `quantize_op` nodes).

**Reproduction** (self-contained: random weights, one MLP):

```python
import modelopt.torch.quantization as mtq, torch, torch.nn as nn, torch.nn.functional as F, torch_tensorrt
from modelopt.torch.quantization.utils import export_torch_mode

H, FF, L = 4096, 12288, 64
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.up, self.down = (nn.Linear(H, FF, bias=False), nn.Linear(H, FF, bias=False), nn.Linear(FF, H, bias=False))
    def forward(self, x):
        return x + self.down(F.silu(self.gate(x)) * self.up(x))

torch.manual_seed(0)
m = MLP().cuda().half().eval()
with torch.no_grad():
    m.up.weight.mul_(0.6)                       # different weight amax than gate
x = torch.randn(1, L, H, device="cuda", dtype=torch.half)
mtq.quantize(m, mtq.FP8_DEFAULT_CFG, forward_loop=lambda mod: mod(x))   # same input amax for gate and up
# m.up.input_quantizer.amax = m.up.input_quantizer.amax * 1.001           # <- makes the engine correct
with torch.no_grad(), export_torch_mode():
    ep = torch.export.export(m, (x,), strict=False)
trt = torch_tensorrt.dynamo.compile(ep, inputs=[x], enabled_precisions={torch.half}, use_explicit_typing=False, min_block_size=1)
print(torch.isfinite(trt(x)).all())            # False
```

**Workaround.** For TensorRT versions below `11.5`, `maybe_separate_sibling_input_scales` nudges the input 
amax of the second and third sibling by 0.2% (far below the FP8 grid spacing of about 6%). Applying the same
nudge to the eager  FP8 model changes minADE by 0.01 m or less, so it is not a source of error.

## Problem 2: FP8 Q/DQ on the attention output is fused into the P.V GEMM and is wrong for some chunks

**Symptom.** With the first problem worked around, the FP8 decoder engine was still worse than the eager FP8
model: minADE 2.63 / 2.14 / 2.68 m (seeds 1, 2, 3) against 2.04 / 1.70 / 2.00 m for the eager FP8 model and
2.00 / 1.66 / 1.92 m for FP16. Against the eager layers the error grows with depth (mean per-token relative
error 0.7% after 3 layers, 5.5% after 12, 15.6% after 18; the FP16 engine has 0.1% to 0.3%), and is
concentrated in a few tokens.

**Where it comes from.** Each layer in isolation, fed the exact eager input, was compared with the eager layer:

* only the quantizer on the `o_proj` input (the attention output) matters: disabling it in layer 14 takes the
  local error from 1.3% to 0.4% (99th percentile 0.37 to 0.009); the q/k/v, softmax, weight and other input
  quantizers make no difference;
* a standalone TensorRT Q/DQ on the real attention output matches ModelOpt to 0.02%, so the quantize operation is
  fine;
* in layer 14, 48 of the 49 prefill chunks (64 tokens each) are correct to under 1% and exactly one, tokens
  2240 to 2303, has a 38% error per token. Other layers have a different bad chunk;
* in the engine, the Q of the attention output is fused with the P.V matmul into one Myelin kernel
  (`__myl_FcCastMulCast`, tactic `sm80_xmma_gemm_f16f16_f16f16_f16_nn_..._by_fusion_tactic`, FP16 P and V in,
  FP8 out); without the quantizer the matmul is a plain GEMM and `o_proj` runs as an FP16 GEMM.

**Mechanism.** The fused kernel is launched on the current buffers but with the key-length dependent
parameters of an earlier call: the GEMM K extent, the row and batch strides of P and the batch stride of V.
Recomputing the window (2240, 2304) in PyTorch with the converter's GQA layout (P `[kv_heads, groups*T, S]`,
V `[kv_heads, S, head_dim]`) and all of these parameters taken from S=2240 matches the engine output to 0.3%;
the correct result is 38% away, and making only one of the parameters stale does not match either. The
quantize itself is correct. This is a TensorRT runtime bug (confirmed with 11.4.0.106 and 11.5.0.34), not a
problem of the ModelOpt Q/DQ or of the Torch-TensorRT converters.

**Conditions.** The engine uses `static_cache_v2` (dynamic key length) with a dynamic batch dimension (1 to 6);
a static batch did not fail. The plain TensorRT Python API reproduces it with the dumped engine, so the
Torch-TensorRT runtime is not involved. The result depends on the shape history of the execution context, not
on the data:

* of S=2240 and S=2304 (batch 1, 64 tokens), whichever runs first is correct and the other is wrong, and stays
  wrong when repeated; another shape in between (for example S=1000) avoids it. This looks like a shape-keyed
  cache of launch parameters in which the two shapes collide;
* a preceding call with zero activations and zero caches breaks the next call just as well; a preceding call
  with batch 2 or another number of tokens does not;
* `compute-sanitizer` (initcheck, memcheck) reports no errors and the output is still wrong under it;
  `max_aux_streams=0` changes nothing;
* running every 64-token window right after the window that ends at its start, 32 of 2959 windows are wrong
  (ends 2265 to 2272, 2289 to 2304 and 2801 to 2808); the engine without the quantizer has none.

The bug happens for any data, but it is only visible when the attention is peaked: with spread-out
attention the stale K and strides barely change the output. With random weights and activations and
q/k RMSNorm weights of 3 (the real ones reach 2.5 and 4.4) the same chunk is off by 110%; with 1 it is
off by only 7%, which is why random tensors seemed to pass at first.

**Reproduction.** `fp8_attn_out_fusion.py` in the `repro` folder: one attention block with the shapes of
layer 14 and random weights and activations, compiled with `static_cache_v2` and run as chunked prefill.
It fails on chunk 2240 to 2303; `--control` disables the `o_proj` input quantizer and passes. `--pack`
runs it with the real weights and activations of layer 14 instead (about 110 MB, written by the
`make-pack` subcommand), where the same chunk is off by 38%.

**Workaround.** `disable_attention_output_quantizers` leaves the attention output of the decoder unquantized
(the `o_proj` GEMMs of the decoder then run in FP16). `--fp8_attention_output_quant` brings the quantizer
back, for experiments. With the workaround the 18-layer engine matches the eager FP8 layers about as closely as
the eager FP8 model matches the FP16 one (mean 5.3% against 5.3%, 99th percentile 0.19 against 1.09), and:

| minADE (m), 3 seeds | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| FP16 eager | 2.003 | 1.662 | 1.916 |
| FP8 eager (ModelOpt) | 2.040 | 1.695 | 2.004 |
| TensorRT FP8, before | 2.634 | 2.135 | 2.675 |
| TensorRT FP8, after | 2.071 | 1.714 | 2.044 |

First-token logits match the FP8 reference at cosine 0.99998 with identical top-5 tokens.

## Where the FP8 error is, per component

Mixing FP16 and FP8 engines per component (3 seeds, mean minADE): vision FP8 +0.02 m, expert FP8 0.00 m,
decoder FP8 with the attention-output fusion problem +0.44 m (the errors of vision and decoder add up). After
the workaround the whole FP8 pipeline is within about 0.03 m of the eager FP8 model.

## Limits

One clip, three seeds, one GPU. FP8 only (no NVFP4). The attention output of the decoder is not quantized, so
those GEMMs are not FP8. Speed has not been measured against FP16.
