# Hy-MT3 验证记录

日期：2026-11-26
主机：Linux x86_64，128 核，503 GB RAM，**无 GPU**（`nvidia-smi` 不存在）
参考实现：llama.cpp `9c003a97ab34a69cc49243f9b02014038205ffae`（codeload tarball）
本仓库构建：`cargo build --release`（rustc 1.100.0）
设置：greedy argmax、`n_gen=32`、`n_ctx=2048`、`n_batch=n_ubatch=256`、KV F32、
flash-attn 关闭、`n_gpu_layers=0`、`add_bos=0`、`parse_special=true`

## 1. 模型契约

从 GGUF 头部直接读出（`head_to_tokenizer_gguf.py` 与 `gguf_meta.py` 的解析器），
并用 llama.cpp 的 `print_info:` banner 交叉确认。

| 模型 | arch | n_layer | n_embd | head/kv | head_dim | n_ff | n_expert/used | n_ff_exp/shexp | rope base | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hy-MT3-1.8B-Q8_0 | `hunyuan-dense` | 32 | 2048 | 16/4 | 128 | 6144 | — | — | 11158840 | 120818 |
| Hy-MT3-7B-Q8_0 | `hunyuan-dense` | 32 | 4096 | 32/8 | 128 | 14336 | — | — | 11158840 | 128167 |
| Hy-MT3-30B-A3B-Q4_K_M | `hy_v3` | 48 | 2048 | 32/4 | 128 | 6912 | 128/8 | 768/768 | 11158840 | 120832 |

30B-A3B 的 llama.cpp banner 与本仓库 `HyConfig` 读到的值逐项一致：
`arch=hy_v3`、`n_layer=48`、`n_ff=6912`、`n_expert=128`、`n_expert_used=8`、
`n_expert_groups=0`；`expert_weights_scale=2.826`、`expert_weights_norm=true`、
`expert_gating_func=2`（sigmoid）。第 0 层是 dense（无 `ffn_gate_inp`），
第 1..47 层是 MoE + 无门控 shared expert。

文件身份（`sha256sum`，ModelScope `Tencent-Hunyuan/*` master）：

| 文件 | bytes | SHA256 |
| --- | --- | --- |
| `Hy-MT3-30B-A3B-Q4_K_M.gguf` | 19,502,694,048 | `6e301af0e6d374ff16953dd61665c53e2993ef0b439e2d5ee28f1b6f6570bfea` |
| `Hy-MT3-30B-A3B-Q8_0.gguf` | 32,791,821,792 | `c4254b824fb707f2d253deda4f86d390c3cc9247f0e98ac3ed01f10298c28d25` |


## 2. Tokenizer（离散契约，精确）

**方法**：官方 `tokenizer.json`（HF `tokenizers` 0.23.3）当权威，和本仓库分词器
逐 ID 比较，14 组用例 × 3 个模型，全部 **PASS**：

```
== 1.8B ==  14/14 OK      == 7B ==  14/14 OK      == 30B ==  14/14 OK
```

覆盖 raw prompt / 数字 / 长数字 / 中英混排 / hy 控制 token / 组合字符 / emoji /
空白 / 英文缩写 / 全标点 / 中日韩 / 空串 / 单字符 / 纯换行。

prompt ID 与 llama.cpp 也逐 ID 相同：

| 模型 | raw | chat | chat_bos |
| --- | --- | --- | --- |
| 1.8B | 23 ✓ | 25 ✓ | 26 ✓ |
| 7B | 25 ✓ | 41 ✓ | 51 ✓ |
| 30B-A3B | 23 ✓ | 25 ✓ | 26 ✓ |

**本轮修掉的既有缺陷**：`tokenizer.ggml.pre = "hunyuan"`（7B）原本和
`hunyuan-dense`（1.8B/30B）共用数字分支，把数字连成一整段；官方
`tokenizer.json` 与 llama.cpp 都是 `\p{N}`（**每个数字单独成段**）。后果是
`12345` 从一个 piece 变成 5 个，BPE 之后无法再合并，真实 token ID 不同。
已修（`src/core/tokenizer/mod.rs`），并加了回归测试
`hy_scanner_isolates_each_digit_but_hy_dense_chunks_them`。

**影响面**：`pre = "hunyuan"` 的模型都会吃到这次修正。仓库里能确认的调用方是
Hy-MT2-7B（`general.architecture=hunyuan-dense`、`pre=hunyuan`、词表 128167，
与本轮 Hy-MT3-7B 完全同一套 tokenizer 配置）和 Hy-MT3-7B。两者此前都
把数字连段，现在与官方 `tokenizer.json` 和 llama.cpp 一致 —— 是修复而不是回归。
`pre = "hunyuan-dense"`（1.8B / 30B / Hy-MT2-1.8B）**行为不变**，
由回归测试 `hy_scanner_isolates_each_digit_but_hy_dense_chunks_them` 同时锁住两侧。

**llama.cpp 侧确认（7B）**：prompt `价格是12345元，订单号6124c78e。` 两侧
20 个 prompt ID 完全相同，`12345` → `16,17,18,19,20`（五个单数字 piece，
修复前是 2 个），`6124c78e` → `21,16,17,19,66,22,23,68`；greedy 32/32、
cos 0.999974、rel_rmse 0.75%、top-10 10/10。

`clean_spaces`（1.8B/7B `true`、30B `false`）只影响 **detokenize**（去掉
`? ! . ,` 前的空格），不参与分词；本仓库未实现该开关，属全仓库既有缺口，
不影响 token ID 对齐，但比对文本时要容忍空格差异。

## 3. 数值对齐

`cos` = 最后 prompt 位置 logits 的余弦相似度；`rel_rmse` = RMSE / max|logit|；
`greedy` = 32 步贪心 token 是否完全一致；`top10` = 前 10 名的集合重合数。

| 模型 | 变体 | greedy 32 | cos | rel_rmse | top10 |
| --- | --- | --- | --- | --- | --- |
| 1.8B Q8_0 | raw | **32/32** | 0.999563 | 0.616% | 9/10 |
| 1.8B Q8_0 | chat | 13 步分歧 | 0.999792 | 0.200% | 10/10 |
| 1.8B Q8_0 | chat_bos | **32/32** | 0.999785 | 0.199% | 10/10 |
| 7B Q8_0 | raw | **32/32** | 0.999910 | 0.164% | 10/10 |
| 7B Q8_0 | chat | **32/32** | 0.999981 | 0.105% | 10/10 |
| 7B Q8_0 | chat_bos | **32/32** | 0.999969 | 0.112% | 10/10 |
| 30B-A3B **Q8_0** | raw | **32/32** | 0.999551 | 0.465% | 10/10 |
| 30B-A3B **Q8_0** | chat | 第 0 步分歧 | 0.802927 | 8.65% | 1/10 |
| 30B-A3B **Q8_0** | chat_bos | **32/32** | 0.996945 | 0.794% | 9/10 |
| 30B-A3B Q4_K_M | raw | 11 步分歧 | 0.999663 | 0.425% | 10/10 |
| 30B-A3B Q4_K_M | chat | 3 步分歧 | 0.947728 | 3.68% | 4/10 |
| 30B-A3B Q4_K_M | chat_bos | **32/32** | 0.997805 | 0.710% | 10/10 |

门槛（`compare.py` 固定，不是事后挑的）：`cos ≥ 0.99`、`rel_rmse ≤ 1%`、
top-1 必须相同、prompt/generated token ID 必须完全相同。

**判读**：

- 两个 dense 模型全部达标；7B 三个变体 greedy 全对，cos ≈ 0.9999。
- 30B-A3B 的 **模板路径（chat_bos）通过**：32/32、cos 0.9978、rel 0.71%。
- 30B 的 `chat`（不带 BOS）**未过门槛**：Q4_K_M cos 0.948 / rel 3.68%，
  Q8_0 cos 0.803 / rel 8.65%（Q8_0 更差，因为该位置的 top-1 本来就近乎并列）。
  证据：llama.cpp 侧该位置 top-1 只领先 **0.59**（`292` vs `120029`），
  本仓库侧 top-1 是 `120025`（eos）—— 两侧都在「要不要停」的临界点上，
  48 层 × 128 选 8 的路由任何一层翻转都会换掉整层输出。
  该变体不是模型的真实输入形态（GGUF 模板本身会发
  `<｜hy_begin▁of▁sentence｜>`），保留它是为了暴露敏感度，不是验收路径。
  两侧在该形态下**都退化成垃圾**：llama.cpp 输出 `。。。。`，本仓库输出
  `\n\n\n…`。同一份代码在 `raw`（Q8_0 32/32、cos 0.9996）与 `chat_bos`
  （Q8_0 32/32、cos 0.9969）上都通过，所以这不是实现缺陷而是
  分布外输入的数值放大。

**为什么不做逐位对齐**：两侧的有损来源不同且都是刻意保留的 —— 本仓库用
`silu_mul_approx_inplace`（近似 SiLU）、自有 SIMD kernel 与不同的求和顺序；
量化档的 MoE 路由是离散 top-k，近临界处对最小数值差敏感。按
`.agents/skills/adapting-new-models` 的口径，这些用指标验收，不追求 bit-exact。
逐位比较只用于定位分叉，本轮未启用 `parity-trace`。

## 3.5 真实入口（CLI / HTTP）

**CLI（真实权重，30B-A3B Q4_K_M）**：

```
./target/release/rust-model-inference \
  --model Hy-MT3-30B-A3B-Q4_K_M.gguf \
  --prompt "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n机器学习是人工智能的一个分支。" \
  --max-tokens 32 --threads 16 --temp 0 --max-context 2048
→ <think></think>Machine learning is a branch of artificial intelligence.
```

`--jinja` 得到同一句。这条路径经过 `build_text_runtime` → `HyV3TextRuntime`，
不是示例程序另写的一套。

**HTTP**：`/v1/chat/completions` 最初返回
`400 Tool/chat template is unsupported for architecture hy_v3` ——
`build_prompt` 在没有模板时回落到手写渲染器，而那个渲染器的 arch 白名单不认识
`hy_v3`。根因是 server 默认只在 `--jinja` 或 `mage_vl` 时才启用模型自带模板。
已修：`hy_v3` 与 `mage_vl` 一样**默认启用自带模板**（`app/server/mod.rs`），
因为这份 11 KB 模板就是该模型的输入契约，手写渲染器给不出等价 prompt。

修后的 prompt 构造由回归测试
`hy_v3_http_prompt_uses_the_models_own_template` 锁住（`RMI_HY_MT3_TOKENIZER_GGUF`
指向 tokenizer-only GGUF 即可无权重运行）：

```
rendered = "\n\n<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort…<｜hy_User｜>…"
```

即 BOS 在前、没有「位置 0 是控制 token」的退化形态；同一测试还断言
**没有模板时仍然明确报错**，不会静默换成 ChatML prompt。

> 注意：模板的实际输出比 `chat_bos` 变体多了前导 `\n\n` 与
> `<｜reasoning_mode｜>` 前言，所以 `chat_bos` 是「模板形态」的近似而不是逐字节
> 相同。端到端正确性由上面两条真实入口 + 模板渲染测试共同覆盖。
>
> **未复验**：修完 server 后重新构建二进制时测试机被其他任务占满（load ≈ 30），
> 构建未在窗口内完成，因此 `hy_v3` 的 HTTP 全链路（起服务 + curl）**没有重跑**；
> 上表的分级是：prompt 构造已用测试验证，推理部分与 CLI 共用同一
> `HyV3TextRuntime`（CLI 已实测）。这一条按「已实现 / 部分验证」记录。

## 4. 未验证 / 未运行

| 项 | 状态 | 原因 |
| --- | --- | --- |
| 30B-A3B **Q8_0** | **已完成** | 见上表；SHA256 `c4254b82…c28d25` |
| 1.8B / 7B **Q4_K_M** | 未运行 | 同上；dense 路径与 Hy-MT2 共用，Q8_0 已覆盖 |
| 30B-A3B `IQ2` | 不适用 | 该档含 IQ2_S / IQ3_S 张量，`hy_v3` loader 只接受 F32/BF16/Q4_K/Q6_K/Q8_0 |
| Vulkan | 未运行 | `hy_v3` 没有 Vulkan 算子，严格 `--compute vulkan` 明确报错；测试机无 GPU |
| 30B-A3B MTP/NextN | 不适用 | 这些 GGUF 没有 `nextn.*` 张量（766 个张量里没有） |
| 完整 262144 上下文 | 未运行 | 只验证到 `n_ctx=2048` |
| JEV | 不适用 | `HyV3TextRuntime` 只暴露 `generate()`，没有 logits 入口 |
| HTTP 全链路复验 | 部分验证 | prompt 构造有测试；server 重建未在窗口内完成，未重跑起服务+curl |

## 5. 性能（同机 CPU，仅供参考）

`llama-bench -p 32 -n 16 -t 16 -ngl 0`（同机 CPU，**无 GPU**）：

| 模型 | 精度 | pp32 (t/s) | tg16 (t/s) |
| --- | --- | --- | --- |
| Hy-MT3-1.8B | Q8_0 | 42.27 ± 7.93 | 5.07 ± 0.18 |
| Hy-MT3-7B | Q8_0 | 8.95 ± 0.68 | 1.76 ± 0.14 |
| Hy-MT3-30B-A3B | Q8_0 | 10.01 ± 9.44 | 6.46 ± 1.95 |

> **读这张表之前先看这条**：测试机与其他两个 agent 共享，采集时 load average 在
> 24–51 之间波动，所以数字偏悲观且噪声大（30B 的 pp32 标准差就有 ±9.44）。
> 作为量级参考：1.8B 的 tg 在空载 8 线程下是 10.01 t/s，本表 16 线程只有 5.07 t/s，
> 约 2× 的争用惩罚。这不是可比的性能基线，只是「能跑、量级合理」的记录；
> 干净数字需要在空载机器上重跑。
>
> 本机无 GPU，所以没有 CPU / Vulkan / 参考实现的三方对比 —— `hy_v3` 本身也没有
> Vulkan 算子（严格 `--compute vulkan` 明确报错），不存在可比的设备路径。
