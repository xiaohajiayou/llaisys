# Qwen2 模型需求设计

## 1. 功能需求

1. **权重加载**
   - 严格对齐当前头文件原型：后端通过 `llaisysQwen2ModelWeights(model)` 暴露权重槽位，而不是 name-based `load_weight(name, ptr)`。
   - Python 层负责：
     - 解析 safetensors。
     - 将 HF 权重名映射到 `LlaisysQwen2Weights` 的具体字段与层索引。
     - 使用 `tensorCreate + tensorLoad` 创建并填充 `llaisysTensor_t`，再写入对应槽位。
   - C++ 侧负责：
     - 在 create 阶段分配并初始化 `LlaisysQwen2Weights`（包括每层数组，长度为 `nlayer`）。
     - 在首次 infer 前进行 shape/dtype/device 校验并快速失败（`ASSERT/CHECK_*`）。
2. **模型推理（Infer）**
   - 通过 `llaisysQwen2ModelInfer(model, token_ids, ntoken)` 支持 batch=1 的 prompt + decode 两阶段推理。
   - `token_ids` 表示“本次新增”的 token 序列，infer 会更新 KV-Cache。
   - 返回值为“下一 token”的采样结果；阶段验收以 argmax 为准（需与 `test/test_infer.py --test` 一致）。
   - 必须遵循现有算子契约：
     - `ops::linear` 的权重方向为 `[out_dim, in_dim]`。
     - `ops::rope` 的 `pos_ids` 必须为长度等于 `ntoken` 的 int64 序列。
3. **KV-Cache**
   - KV-Cache 容量由 `meta.maxseq` 决定，并在 create 时初始化。
   - 必须支持 `kvlen > qlen` 的 causal attention（带缓存推理）。
   - 由于当前原型未暴露 `reset_cache`：
     - 阶段性约束为“一次 generate 使用一个模型实例”；
     - 后续如需复用实例，可再扩展 reset API，但不影响当前验收目标。
4. **接口（最小闭环）**
   - 通过 C API 公开以下最小集合：
     - `llaisysQwen2ModelCreate`
     - `llaisysQwen2ModelDestroy`
     - `llaisysQwen2ModelWeights`
     - `llaisysQwen2ModelInfer`
   - Python ctypes 需按头文件一一绑定结构体与函数，`python/llaisys/models/qwen2.py` 基于该闭环实现。
5. **运算支持**
   - 依赖现有 ops：embedding、rms_norm、rope、self_attention、linear、swiglu、argmax。
   - 适配 float32/float16/bfloat16（与 ops 能力保持一致）。

## 1.1 阶段性目标

- **阶段 1（当前作业）**：以 `Qwen2Model` 为中心直接管理权重槽位、KV-Cache、工作区，尽快跑通 `test/test_infer.py`。允许重复代码或专用结构，只要对外原型稳定。
- **阶段 2（后续优化）**：在功能确认后，再视需求抽象公共组件（模型上下文、KV-Cache 管理器、workspace allocator、reset API 等），不影响阶段 1 交付。

## 2. 非功能需求

- **性能**：满足作业测试即可；中间张量允许使用智能指针，后续优化可引入内存池。
- **可扩展性**：模块划分需能扩展至其它模型或 GPU 版本；接口已包含 `device/device_ids/ndevice` 与 `meta.maxseq`。
- **可维护性**：代码组织遵循现有目录结构（`include/llaisys/models`、`src/llaisys`、`python/llaisys`）；命名、断言风格与仓库一致。
- **可靠性**：
  - 关键步骤添加 `ASSERT/CHECK_*` 检查（shape、dtype、device、缓存长度等）。
  - 不允许 C++ 异常跨越 C API 边界传播；失败路径以“快速失败/终止”为主，重点依赖前置校验。

## 3. 约束与假设

1. **硬件**：当前实现仅需支持 CPU；后续可借助 Runtime API 扩展到 GPU，但接口形态不变。
2. **精度**：`meta.dtype` 为主 dtype；如权重为半精度，需保证 ops 路径可正确处理或在实现中集中转换。
3. **Batch**：作业阶段仅考虑单样本推理；输入 token 以 Python 提供的 `input_ids` 列表表示。
4. **Meta 不变量**（建议强校验）：
   - `hs == nh * dh`。
   - `nkvh <= nh`。
   - `maxseq > 0`，`voc > 0`。
5. **Tokenizer/Prompt**：LLAISYS Python 负责构建 prompt；C++ 模型仅接收 token IDs。
6. **错误处理**：当前函数签名无状态码返回；错误路径以断言失败/终止为主，Python 侧需尽量在调用前完成校验。

## 4. 验收标准

- 运行 `python test/test_infer.py --model <path> --test` 通过全部检查，输出与 PyTorch 一致。
- 权重加载阶段在首次 infer 前对 shape/dtype/device 进行强校验；不匹配时快速失败（允许为致命错误）。
- 以“一次 generate 一个模型实例”的使用方式，不产生内存泄漏，`valgrind`/`ASAN` 无异常。
- 文档（本需求文档、概要设计、接口设计）与实现保持同步，便于后续 Review。

## 5. 实现 Checklist（落地顺序建议）

**Step 1: Meta 与基础约束（fail fast）**
- 校验 `hs == nh * dh`。
- 校验 `nkvh <= nh`。
- 校验 `nlayer > 0`、`maxseq > 0`、`voc > 0`。
- 固定模型 `device` 与 `meta.dtype`，作为后续一致性检查基准。

**Step 2: 权重槽位分配（Create 阶段完成）**
- 在 `llaisysQwen2ModelCreate` 中分配并初始化 `LlaisysQwen2Weights`。
- 为所有每层数组字段分配长度 `nlayer` 的槽位。
- 记录 `meta/device` 到模型内部状态，供校验与推理使用。

**Step 3: 权重加载与映射（Python 侧主导）**
- 建立 HF 权重名到 `LlaisysQwen2Weights` 槽位的映射表（字段名 + 层索引）。
- 使用 `tensorCreate + tensorLoad` 创建并填充 `llaisysTensor_t`。
- 将 tensor 句柄写入 `weights` 对应槽位。
- 映射需覆盖完整清单（至少包含：embed、final norm、每层的 attn norm/qkv/o、mlp norm/gate/up/down）。

**Step 4: 权重一致性校验（首次 infer 前强校验）**
- 校验所有必需权重句柄非空。
- 校验 dtype 与 `meta.dtype` 兼容。
- 校验 device 与模型 device 一致。
- 校验 shape 与接口文档中的契约一致。
- shape 校验必须以 `ops::linear` 的真实方向（`weight=[out,in]`）为准。

**Step 5: KV-Cache + 推理流程（最小闭环跑通）**
- 按 `[maxseq, nkvh, dh]` 为每层预分配 K/V cache，并维护 `cur_len`。
- 实现 `infer(ntoken)` 的主路径：embedding → N 层 block → final norm → lm head → argmax。
- 在 infer 中正确更新 KV-Cache，并返回“下一 token”。
- 在每次 infer 开头构造 RoPE 位置：`pos_ids = arange(cur_len, cur_len + ntoken)`（int64，长度为 `ntoken`）。

## 6. 内存契约（验收口径）

- 接管规则：权重 tensor 句柄写入 `LlaisysQwen2Weights` 槽位后，由模型实例接管所有权。
- 释放规则：`llaisysQwen2ModelDestroy` 必须释放所有被接管的权重、KV-Cache 与 Workspace。
- 去重规则：允许 `in_embed == out_embed`；销毁时必须去重，确保同一句柄只释放一次。
- 覆盖规则：同一槽位被重复赋值时，必须先释放旧句柄再接管新句柄。
- 溢出规则：`infer` 开头必须检查 `cur_len + ntoken <= maxseq`；不满足时 fail-fast。
- 前端限流：Python `generate` 应基于剩余容量裁剪 `max_new_tokens`，避免触发后端 fail-fast。
- Workspace 策略：`ensure_workspace_(ntoken)` 在每次 infer 开头调用一次；策略为 grow-only + 复用，destroy 统一释放。
