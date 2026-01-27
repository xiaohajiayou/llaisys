# Qwen2 模型概要设计

## 1. 设计目标

- 复用 LLAISYS 现有张量与算子，完成 DeepSeek-R1-Distill-Qwen-1.5B 的纯 C++ 推理。
- 保持 Python/ctypes 层仅负责配置解析、权重映射与推理驱动，所有算子调度、KV-Cache 与内存生命周期在 C++ 层执行。
- 构建清晰的模块边界，后续可扩展多设备、优化器或其它模型。
- 严格对齐当前 C 头文件原型：通过 `llaisysQwen2ModelWeights` 暴露权重槽位，前端使用 tensor 句柄写入数据。

## 2. 总体架构

```
Python Qwen2 模型
    └── ctypes 封装 (python/llaisys/libllaisys/qwen2.py)
          └── C API (include/llaisys/models/qwen2.h)
                └── Qwen2Model C++ 实现 (src/llaisys/qwen2/)
                      ├── 权重槽位 (LlaisysQwen2Weights 句柄表)
                      ├── 层级执行图 (RMSNorm → SelfAttention → SwiGLU)
                      └── KV-Cache/工作区管理
```

- **Python 层**：解析 HuggingFace 配置与 safetensors，将 HF 权重名映射到 `LlaisysQwen2Weights` 各槽位；提供 `generate` 与采样逻辑。
- **ctypes 层**：声明 C API 与 tensor API，负责 Python buffer、`llaisysTensor_t` 与后端权重槽位互通。
- **C API**：屏蔽 C++ 具体实现，提供创建、销毁、获取权重槽位与推理函数（不提供 name-based loader）。
- **C++ 模型**：搭建完整推理管线，复用 ops 中的 embedding、rope、self_attention、linear、rms_norm、swiglu、argmax。

## 2.1 代码目录与注释规范

```
llaisys/
├─ include/
│  └─ llaisys/
│     └─ models/
│        └─ qwen2.h          # C API、配置结构、句柄定义
├─ src/
│  └─ llaisys/
│     └─ qwen2/
│        ├─ qwen2_api.cc     # C API 与 Qwen2Model 绑定
│        ├─ qwen2_model.hpp  # Qwen2Model / Weights / KvCache / Workspace 声明
│        ├─ qwen2_model.cpp  # 构造、infer、KV-Cache 逻辑
│        ├─ qwen2_block.cpp  # Transformer Block（Attention + MLP）实现
│        └─ qwen2_validate.cpp # 权重 shape/dtype/device 校验（建议）
├─ python/
│  └─ llaisys/
│     ├─ libllaisys/
│     │  └─ qwen2.py         # ctypes 声明与 Qwen2Handle
│     └─ models/
│        └─ qwen2.py         # Python 模型包装、HF→weights 映射、generate
└─ doc/
   └─ qwen2_*.md             # 设计/需求/接口文档
```

- **注释约定**：
  - `include/`：关键结构与 API 使用 `//` 说明参数含义和生命周期。
  - `src/llaisys/qwen2/*.cpp`：文件头注明模块职责，复杂函数前加简述；内部逻辑依赖 `ASSERT`/`CHECK_*`。
  - `python/llaisys/libllaisys/qwen2.py`：每个 ctypes 函数注明 `argtypes/restype` 对应的实际意义。
  - `python/llaisys/models/qwen2.py`：类 docstring 描述整体用途，私有方法附注 HF 权重映射与采样策略。
  - `doc/`：保持当前 Markdown 结构，记录设计演进。

以上目录中 `src/llaisys/qwen2/` 及对应 Python/文档文件为本次实现新增内容，提交时需一并创建并按注释约定维护。所有新增文件遵循仓库既有风格：`snake_case` 文件名、命名空间 `llaisys::models::qwen2`，注释使用英语或简明中文，避免冗长段落。

## 3. 关键模块

### 3.1 Qwen2Model 类

- 构造时接收 `LlaisysQwen2Meta + device/device_ids/ndevice`（与头文件一致）。
- 关键不变量（建议在构造期校验）：
  - `hs == nh * dh`。
  - `nkvh <= nh`。
  - `maxseq > 0`，`voc > 0`。
- 成员：
  - `LlaisysQwen2Meta meta_`：结构超参与数值超参。
  - `LlaisysQwen2Weights weights_`：对外暴露的权重槽位句柄表（create 后地址稳定）。
  - `std::vector<KvCache>`：K/V 缓冲区与当前长度。
  - `std::vector<TransformerBlock>`：封装单层计算流程（引用对应权重槽位与 `KvCache`）。
  - `Workspace`：预分配中间张量（q/k/v buffer、attn scores、mlp buffer 等）。
- 方法：
  - `weights()`：返回 `weights_` 指针，供前端填充权重句柄。
  - `infer(token_ids, ntoken)`：执行 embedding → N×Block → Final RMSNorm → Linear，更新 KV-Cache，并返回下一 token（阶段目标为 argmax）。
  - `validate_weights_or_die_()`（建议）：在首次 infer 前统一校验权重 shape/dtype/device。

### 3.2 Transformer Block

- **RMSNorm**：CPU 算子 `ops::rms_norm`，权重来自每层配置。
- **Self-Attention**：
  - 生成 Q/K/V，应用 `ops::rope`，拼接/读取 KV-Cache。
  - 使用 `ops::self_attention` 计算注意力输出，并执行残差。
- **SwiGLU MLP**：
  - 调用 `ops::linear` + `ops::swiglu` + `ops::linear` 组合。
  - 输出加残差。
- **线性层权重方向（写死约定）**：
  - 当前 `ops::linear` 约定为 `in=[M,K]`、`weight=[N,K]`、`out=[M,N]`，即 `weight=[out_dim, in_dim]`。
  - Q/K/V/O 与 MLP 的权重 shape 需按该方向校验与映射（与 PyTorch linear 一致，默认不转置）。
- **RoPE 位置口径（与缓存对齐）**：
  - 在 `infer(ntoken)` 中，对“新 token”构造 `pos_ids = arange(cur_len, cur_len + ntoken)`。
  - RoPE 仅作用于本次新增的 Q/K（缓存中历史部分视为已处理）。

### 3.3 KV-Cache

- 每层缓存结构：`tensor_t k_cache`, `tensor_t v_cache`, `size_t cur_len`。
- 预分配 shape `[max_seq_len, nkvhead, head_dim]`。
- Prompt：写入 `[cur_len : cur_len + token_len]`。
- Decode：切 `k_cache.slice(0, cur_len + token_len)` 与 `v_cache` 喂给 attention。
- 重置（当前主要通过重建模型实例或内部路径触发）时仅将 `cur_len = 0`。
- 在 `infer` 开头强制检查 `cur_len + ntoken <= maxseq`，不满足时 fail-fast。

### 3.4 内存管理契约（阶段 1 统一口径）

- 所有权：权重 tensor 句柄一旦写入 `weights_` 槽位，即由 `Qwen2Model` 接管。
- 释放责任：`llaisysQwen2ModelDestroy`（或 `Qwen2Model` 析构）统一释放权重、KV-Cache、Workspace。
- 去重释放：允许 `in_embed == out_embed`，销毁时基于指针去重，避免 double free。
- 覆盖赋值：同一槽位被重复赋值时，先释放旧句柄，再接管新句柄。
- Workspace：通过 `ensure_workspace_(ntoken)` 在每次 `infer` 开头进行容量检查与 grow-only 扩容，随后在本次推理中复用。

> **阶段性说明**：当前设计优先保证 Qwen2 推理尽快跑通。权重槽位、KV-Cache 与中间张量由 `Qwen2Model` 直接管理；HF 权重名到槽位的映射放在 Python 侧，避免在 C API 中引入 name-based loader。

## 4. 主要数据流

1. Python 解析配置并构造 `LlaisysQwen2Meta`。
2. 调用 `llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)`，后端分配权重槽位与 KV-Cache。
3. 调用 `llaisysQwen2ModelWeights(model)` 获取 `weights` 句柄表。
4. Python 遍历 safetensors：
   - 将 HF 权重名映射到 `weights` 的具体字段/层索引。
   - 使用 `tensorCreate + tensorLoad` 创建并填充 `llaisysTensor_t`。
   - 把 tensor 句柄写入 `weights` 对应槽位。
   - 对 RoPE 所需位置，在每次 infer 前构造 `pos_ids = arange(cur_len, cur_len + ntoken)`（int64，长度为 `ntoken`）。
5. `generate`：
   - For prompt: 一次性调用 `llaisysQwen2ModelInfer(model, token_ids, ntoken)`，更新 KV-Cache。
   - For decode: 循环 single-token infer + 采样（阶段目标为 argmax），直到达到 `max_new_tokens` 或命中 `end_token`。
   - Python 端负责维护输入 token 序列与停止条件。

## 5. 未来扩展

- **设备适配**：接口已包含 `device/device_ids/ndevice`，后续可实现 CUDA 版本 ops，只需在 `self_attention`/`linear` 等算子派发即可。
- **内存池**：初版使用智能指针管理 tensor，后续可扩展 `WorkspaceAllocator`，减少中间张量重复分配。
- **多批次/多请求**：默认 batch=1，可通过扩展 KV-Cache 结构与前端接口支持批量推理或多用户服务。

以上设计遵循现有仓库命名与代码组织，确保模块解耦、易于调试并满足作业 #3 的所有需求。
