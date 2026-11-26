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

## 4. Grouped (depthwise) convolutions on the device

`conv3d_tiled.comp` takes `groups == 1` only (`try_conv3d` refuses the rest), so the
depthwise half of these graphs stays on the host even under `--compute vulkan`. A grouped
generalisation was written and reverted: it is numerically wrong on the device for
`groups > 1` (correct for `groups == 1`, so the failure is in the group indexing), and with
the transfer cost above there was nothing to gain by fixing it before point 3 is addressed.

## What a real GPU path needs, in order

1. Fix or exclude the device attention for these shapes (point 2) -- without it the output is
   wrong under the policy that does not verify.
2. Keep the feature maps resident on the device across the graph (point 3) -- without it the
   device is slower than the host.
3. Only then the grouped convolution (point 4), which is a small shader change once 1 and 2
   are in place.
