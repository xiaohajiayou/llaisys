# Qwen2 模型概要设计

## 1. 设计目标

- 复用 LLAISYS 现有张量与算子，完成 DeepSeek-R1-Distill-Qwen-1.5B 的纯 C++ 推理。
- 保持 Python/ctypes 层仅负责权重加载与推理驱动，所有算子调度、KV-Cache 与内存生命周期在 C++ 层执行。
- 构建清晰的模块边界，后续可扩展多设备、优化器或其它模型。

## 2. 总体架构

```
Python Qwen2 模型
    └── ctypes 封装 (python/llaisys/libllaisys/qwen2.py)
          └── C API (include/llaisys/models/qwen2.h)
                └── Qwen2Model C++ 实现 (src/llaisys/qwen2/)
                      ├── 权重存储 (LayerWeights, Embedding, LMHead)
                      ├── 层级执行图 (RMSNorm → SelfAttention → SwiGLU)
                      └── KV-Cache/工作区管理
```

- **Python 层**：解析 HuggingFace 配置与 safetensors，将权重转交给后端；提供 `generate` 与 argmax 采样接口。
- **ctypes 层**：声明 C API，负责 Python buffer 与 C++ tensor 的互通。
- **C API**：屏蔽 C++ 具体实现，提供创建、销毁、权重加载、推理、KV-Cache 控制等函数。
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
│        ├─ qwen2_model.hpp  # Qwen2Model / LayerWeights / KvCache / Workspace 声明
│        ├─ qwen2_model.cpp  # 构造、forward、KV-Cache 逻辑
│        ├─ qwen2_block.cpp  # Transformer Block（Attention + MLP）实现
│        └─ qwen2_loader.cpp # safetensors 命名映射与写入
├─ python/
│  └─ llaisys/
│     ├─ libllaisys/
│     │  └─ qwen2.py         # ctypes 声明与 Qwen2Handle
│     └─ models/
│        └─ qwen2.py         # Python 模型包装、generate
└─ doc/
   └─ qwen2_*.md             # 设计/需求/接口文档
```

- **注释约定**：
  - `include/`：关键结构与 API 使用 `//` 说明参数含义和生命周期。
  - `src/llaisys/qwen2/*.cpp`：文件头注明模块职责，复杂函数前加简述；内部逻辑依赖 `ASSERT`/`CHECK_*`。
  - `python/llaisys/libllaisys/qwen2.py`：每个 ctypes 函数注明 `argtypes/restype` 对应的实际意义。
  - `python/llaisys/models/qwen2.py`：类 docstring 描述整体用途，私有方法附注权重命名、采样策略。
  - `doc/`：保持当前 Markdown 结构，记录设计演进。

以上目录中 `src/llaisys/qwen2/` 及对应 Python/文档文件为本次实现新增内容，提交时需一并创建并按注释约定维护。所有新增文件遵循仓库既有风格：`snake_case` 文件名、命名空间 `llaisys::models::qwen2`，注释使用英语或简明中文，避免冗长段落。

## 3. 关键模块

### 3.1 Qwen2Model 类

- 构造时接收 `Qwen2Config`（隐藏维度、层数、nhead、nkvhead、intermediate size、vocab、max seq len 等）。
- 成员：
  - `EmbeddingWeights`：token embedding，最终 linear/lm head（可共享 storage）。
  - `std::vector<LayerWeights>`：每层包含注意力、MLP、RMSNorm 权重。
  - `std::vector<KvCache>`：K/V 缓冲区与当前长度。
  - `std::vector<TransformerBlock>`：封装单层计算流程（引用对应 `LayerWeights` 与 `KvCache`）。
  - `Workspace`：预分配中间张量（q/k/v buffer、attn scores、mlp buffer 等）。
- 方法：
  - `load_weight(name, data_ptr)`：根据 safetensors 名称写入对应 tensor。
  - `forward(input_ids, logits_out)`：执行 embedding → N×Block → Final RMSNorm → Linear，更新 KV-Cache，并将 logits 拷出。
  - `reset_cache()` / `set_max_seq_len(max_len)`：管理缓存生命周期。

### 3.2 Transformer Block

- **RMSNorm**：CPU 算子 `ops::rms_norm`，权重来自每层配置。
- **Self-Attention**：
  - 生成 Q/K/V，应用 `ops::rope`，拼接/读取 KV-Cache。
  - 使用 `ops::self_attention` 计算注意力输出，并执行残差。
- **SwiGLU MLP**：
  - 调用 `ops::linear` + `ops::swiglu` + `ops::linear` 组合。
  - 输出加残差。

### 3.3 KV-Cache

- 每层缓存结构：`tensor_t k_cache`, `tensor_t v_cache`, `size_t cur_len`。
- 预分配 shape `[max_seq_len, nkvhead, head_dim]`。
- Prompt：写入 `[cur_len : cur_len + token_len]`。
- Decode：切 `k_cache.slice(0, cur_len + token_len)` 与 `v_cache` 喂给 attention。
- 重置时仅将 `cur_len = 0`。

> **阶段性说明**：当前设计优先保证 Qwen2 推理尽快跑通，权重/KV/中间张量由 `Qwen2Model` 直接管理。确认功能正确后，再评估是否抽象出共享的 Model Context、KV-Cache 管理器或 Workspace 池，以服务更多模型。

## 4. 主要数据流

1. Python 解析 safetensors → 调用 `qwen2LoadWeight(name, ptr, bytes)`。
2. 创建模型 → `set_max_seq_len` 初始化缓存。
3. `generate`：
   - For prompt: 一次性调用 `forward`，更新 KV-Cache。
   - For decode: 循环 single-token forward + argmax，直到达到 `max_new_tokens`。
   - Python 端负责维护输入 token 序列和 position ids。

## 5. 未来扩展

- **设备适配**：C API 已兼容 `RuntimeAPI`，后续可实现 CUDA 版本 ops，只需在 `self_attention`/`linear` 等算子派发即可。
- **内存池**：初版使用智能指针管理 tensor，后续可扩展 `WorkspaceAllocator`，减少中间张量重复分配。
- **多批次/多请求**：默认 batch=1，可通过扩展 KV-Cache 结构与前端接口支持批量推理或多用户服务。

以上设计遵循现有仓库命名与代码组织，确保模块解耦、易于调试并满足作业 #3 的所有需求。
