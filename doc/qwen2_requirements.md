# Qwen2 模型需求设计

## 1. 功能需求

1. **权重加载**
   - Python 层解析 safetensors 后，可将任意张量（以名称区分）传入 C++ 模型。
   - C++ 需校验名称、维度与 dtype，并将数据写入对应权重 tensor。
2. **模型前向**
   - 支持 batch=1 的 prompt + decode 两阶段推理。
   - 输出与 HuggingFace PyTorch 结果在 argmax 采样一致（`test/test_infer.py --test`）。
3. **KV-Cache**
   - 提供 `reset_cache`、`set_max_seq_len` 操作。
   - 允许 `kvlen > qlen` 的 causal attention（即带缓存推理）。
4. **接口**
   - 通过 C API 公开 `Create/Destroy/LoadWeight/Forward/ResetCache`。
   - Python ctypes 封装同等操作，`python/llaisys/models/qwen2.py` 可直接使用。
5. **运算支持**
   - 依赖现有 ops：embedding、rms_norm、rope、self_attention、linear、swiglu、argmax。
   - 适配 float32/float16/bfloat16（与 ops 能力保持一致）。

## 1.1 阶段性目标

- **阶段 1（当前作业）**：以 `Qwen2Model` 为中心直接管理权重、KV-Cache、工作区，尽快跑通 `test/test_infer.py`。允许重复代码或专用结构，只要接口稳定。
- **阶段 2（后续优化）**：在功能确认后，再视需求抽象公共组件（模型上下文、KV-Cache 管理器、workspace allocator 等），不影响阶段 1 交付。

## 2. 非功能需求

- **性能**：满足作业测试即可；中间张量允许使用智能指针，后续优化可引入内存池。
- **可扩展性**：模块划分需能扩展至其它模型或 GPU 版本；KV-Cache 结构需可调整 `max_seq_len`。
- **可维护性**：代码组织遵循现有目录结构（`include/llaisys/models`、`src/llaisys`、`python/llaisys`）；命名、断言风格与仓库一致。
- **可靠性**：关键步骤添加 `ASSERT/EXCEPTION` 检查（shape、dtype、缓存长度等）；在 forward 中对越界/未加载权重保持防御。

## 3. 约束与假设

1. **硬件**：当前实现仅需支持 CPU；后续可借助 Runtime API 扩展到 GPU。
2. **精度**：默认使用 float32 主路径；如权重为半精度，需在算子内部负责 cast。
3. **Batch**：作业阶段仅考虑单样本推理；输入 token 以 Python 提供的 `input_ids` 列表表示。
4. **Tokenizer/Prompt**：LLAISYS Python 负责构建 prompt；C++ 模型仅接收 token IDs 与可选 position IDs。
5. **错误处理**：C API 通过 `ASSERT` / `EXCEPTION_*` 抛出异常，Python 层调用需捕获并提示。

## 4. 验收标准

- 运行 `python test/test_infer.py --model <path> --test` 通过全部检查，输出与 PyTorch 一致。
- 权重加载阶段对未知名称或维度不匹配给出错误提示。
- 多次 `reset_cache` + `forward` 不产生内存泄漏，`valgrind`/`ASAN` 无异常。
- 文档（本需求文档、概要设计、接口设计）与实现保持同步，便于后续 Review。
