# OCRia on the Vulkan device: what was measured and what blocks it

Question: the detector and the four recognizers are depthwise-separable convolutional
networks, and the box has an AMD Radeon (RADV NAVI31). Can the page read run on it?

Short answer: **not yet, and the convolutions are not what blocks it.** Three things were
measured on that box (128 cores, `--threads 16`, `receipt.png`, F32 weights, cold process).

| run | wall clock |
| --- | ---: |
| `--compute cpu` | 0.55 s / 0.57 s |
| `--compute auto` (device where kernels exist) | 0.59 s / 0.73 s |
| `--compute vulkan` (per-convolution verification on) | 81 s |

## 1. The convolution kernel is correct

`ops::nn::tests::conv3d_matches_the_device_across_kernel_shapes` runs the device kernel
against the CPU one on ten shapes: 1x1 and 3x3 kernels, stride 1 and 2, a 5x5 kernel, 8 to
192 channels, spatial sizes from 1x64 to 32x32. Worst absolute difference **2.3e-5**
against a 1e-4-relative gate, i.e. the kernel agrees with the host arithmetic.

## 2. The device attention is not

Under `--compute auto` the page text comes out wrong:

```
cpu:   'CORNER STORE'  mean confidence 0.986
auto:  '144'           mean confidence 0.382
```

`RMI_GPU_ATTENTION_OFF=1 --compute auto` reproduces the CPU output **exactly** (same text,
mean confidence 0.986), so the divergence is in `try_attention`, not in the convolutions and
not in the projection path. `--compute vulkan` hides it because that policy also turns on
per-operator verification and retries, which is why the 81 s run is both slow and correct.

That is a pre-existing device-path bug, independent of this model: any graph whose attention
shapes hit the kernel would see it.

## 3. Even with that fixed, the transfers dominate

`Conv3dRuntime::forward` uploads the input and the weights and downloads the whole output on
every call. A page issues on the order of a hundred convolutions, so the round trips cost
more than the arithmetic they replace: that is the 0.55 s -> 0.59/0.73 s result above, on a
model whose host kernel is already fast (see `validation.md` on the 1x1 fast path).

The device path pays off for the VAE decoder because `AukVaeRuntime` keeps the activations in
device memory between operators and only moves weights and the final image. A convolutional
OCR graph would need the same treatment -- a resident arena for the detector and recognizer
feature maps, with the host only sequencing operators -- before the device can win.

## 4. Grouped (depthwise) convolutions on the device: fixed

`conv3d_tiled.comp` used to take `groups == 1` only, so the depthwise half of these graphs
stayed on the host. The generalisation is in and correct, and the first attempt failed for a
reason worth recording: the kernel stages one slice of activations in shared memory and all
64 lanes reduce against it, which is right only when every lane wants the same input
channels. A grouped convolution gives each lane its own group, so the staged slice belonged
to one lane's channels and every other lane summed the wrong inputs -- which looked like a
"random" corruption until a two-channel case showed channel 1 half right and channel 0
entirely wrong. Grouped convolutions now read their taps per lane instead (there is no
sharing to exploit when two lanes never read the same channel), and `groups == 1` keeps the
staged path untouched.

`ops::nn::tests::conv3d_matches_the_device_across_kernel_shapes` covers both: ten shapes,
`groups` 1, 2, 8, 32 and 128, 1x1/3x3/5x5 kernels, stride 1 and 2. Worst difference
**1.2e-7** for the grouped cases and 2.3e-5 overall against the 1e-4-relative gate.

With the depthwise half on the device, the page read is finally faster than the host:

| run (receipt.png, 16 threads, load average ~50 on a shared box) | median |
| --- | ---: |
| `--compute cpu` | 806 ms |
| `--compute auto` | **626 ms** |

and all three committed pages produce identical text under `auto` and `cpu`, two runs each.

## 4a. The device attention is not the problem

An earlier note here blamed `try_attention` because disabling it appeared to fix the page.
It was a false lead: `ops::nn::tests::attention_matches_the_device` runs the device kernel
against the host one on the shapes a page actually uses -- 8 queries over 1024 keys, 1 over
2048, a causal 64x64, and an 8-head/2-kv-head case -- and the worst difference is **5.2e-8**
against a 1e-4-relative limit. The corruption was the grouped-convolution staging bug above,
which is now fixed; the page read is exact under `auto` without touching attention.

## 4b. What still costs: one submit per convolution

`RMI_GPU_CPU_CHECK=1` counts **2072 device convolutions per page** -- the detector runs at
several scales and each recognizer pass is its own chain. Every one of them is a
`submit_and_wait`, so at 626 ms the page spends about 0.3 ms per convolution, which is
dispatch and fence latency, not arithmetic: with the weights now resident (the arena keeps
the weight and bias of the last call, keyed by address and length) the remaining traffic is
one input upload and one output download of a small feature map per call.

The next step is therefore not a bigger arena but fewer submissions: record a chain of
convolutions into one command buffer, ping-pong two device regions, and wait once. That
needs the elementwise ops between the convolutions (bias, ReLU) folded into the kernel, and
the host loops in `det.rs` / `rec.rs` to hand the runtime a chain instead of one call at a
time.

## What a real GPU path needs, in order

1. Fix or exclude the device attention for these shapes (point 2) -- without it the output is
   wrong under the policy that does not verify.
2. Keep the feature maps resident on the device across the graph (point 3) -- without it the
   device is slower than the host.
3. Only then the grouped convolution (point 4), which is a small shader change once 1 and 2
   are in place.
