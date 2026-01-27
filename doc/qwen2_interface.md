# Qwen2 模型接口设计

## 1. C API

头文件：`include/llaisys/models/qwen2.h`

```c
typedef struct LlaisysQwenModel *llaisysQwenModel_t;

typedef struct {
    size_t hidden_size;
    size_t num_layers;
    size_t num_attention_heads;
    size_t num_kv_heads;
    size_t intermediate_size;
    size_t vocab_size;
    size_t max_position_embeddings;
    float  rms_norm_eps;
} Qwen2Config;

__export llaisysQwenModel_t qwen2Create(const Qwen2Config *cfg);
__export void qwen2Destroy(llaisysQwenModel_t model);

__export void qwen2LoadWeight(
    llaisysQwenModel_t model,
    const char *name,
    const void *data,
    size_t byte_size);

__export void qwen2Forward(
    llaisysQwenModel_t model,
    const int64_t *input_ids,
    size_t token_count,
    float *logits_out);

__export void qwen2ResetCache(llaisysQwenModel_t model);
__export void qwen2SetMaxSeqLen(llaisysQwenModel_t model, size_t max_len);
```

- `qwen2LoadWeight`：`name` 使用 safetensors 键名，内部建立映射。`data` 指向连续内存（row-major float/fp16/bf16）。
- `qwen2Forward`：输入 `token_count` 个新 token，`logits_out` 长度等于 `vocab_size`。
- KV-Cache 在 `model` 内部维护；`ResetCache` 清空长度，`SetMaxSeqLen` 重新分配缓存。
- **说明**：阶段 1 实现中，`Qwen2Model` 直接持有所有权重与缓存；接口保持稳定，后续若抽象公共上下文无需改动此层。

## 2. C++ 类接口

命名空间：`llaisys::models::qwen2`

```cpp
class Qwen2Model {
public:
    explicit Qwen2Model(const Qwen2Config &cfg);
    ~Qwen2Model();

    void load_weight(std::string_view name, const void *data, size_t bytes);
    void forward(const int64_t *input_ids, size_t len, float *logits_out);
    void reset_cache();
    void set_max_seq_len(size_t max_len);
private:
    void ensure_workspace(size_t len);
    void run_block(size_t layer_idx, tensor_t hidden, size_t new_tokens);

    // 权重与缓存结构 ...
};
```

- `ensure_workspace`：根据本次输入长度调整中间 buffer view。
- `run_block`：对单层执行 RMSNorm、SelfAttention、SwiGLU、残差。
- 权重与缓存以 `tensor_t` 表示，生命周期由 `Qwen2Model` 管理。

## 3. Python ctypes 封装

文件：`python/llaisys/libllaisys/qwen2.py`

```python
class Qwen2Config(Structure):
    _fields_ = [
        ("hidden_size", c_size_t),
        ("num_layers", c_size_t),
        ...
    ]

lib.qwen2Create.argtypes = [POINTER(Qwen2Config)]
lib.qwen2Create.restype = c_void_p
# Destroy/LoadWeight/Forward/ResetCache/SetMaxSeqLen 同理
```

提供包装类：

```python
class Qwen2Handle:
    def __init__(self, cfg: Qwen2Config): ...
    def load_weight(self, name: str, array: np.ndarray): ...
    def forward(self, input_ids: Sequence[int]) -> np.ndarray: ...
    def reset_cache(self): ...
```

## 4. Python 模型 API

文件：`python/llaisys/models/qwen2.py`

```python
class Qwen2:
    def __init__(self, model_path: Union[str, Path], device=DeviceType.CPU):
        cfg = parse_config(model_path)
        self._handle = Qwen2Handle(cfg)
        self._load_safetensors(model_path)

    def generate(self, inputs, max_new_tokens, top_k=1, ...):
        self._handle.reset_cache()
        # prompt forward
        # iterative decode + argmax (可调用 C API 或在 Python 取 logits)
        return outputs
```

`_load_safetensors`：遍历 `.safetensors` 文件，通过 `_handle.load_weight(name, tensor)` 传入。

## 5. 权重命名映射

在 C++ 侧维护 `std::unordered_map<std::string, WeightHandle>`，将 HF 键名映射到具体 tensor，例如：

| safetensors 名称 | C++ 目标 |
|------------------|-----------|
| `model.embed_tokens.weight` | `EmbeddingWeights::token_embedding` |
| `model.layers.0.attention.wq.weight` | `LayerWeights[0].w_q` |
| ... | ... |

未匹配的名称抛出 `EXCEPTION_INVALID_ARGUMENT("Unknown weight")`。

## 6. 数据类型

- 内部权重保持原精度（f32/f16/bf16），与现有 ops 转换逻辑兼容。
- `qwen2Forward` 输出 logits 使用 float32 buffer，Python argmax 后可转换为 int64 token id。

## 7. 错误处理

- C++ 内部使用 `ASSERT`/`EXCEPTION_*`。ctypes 层通过 `check_call`（可选）捕获异常，抛 Python `RuntimeError`。
- KV-Cache 越界、权重未加载、输入长度超过 `max_seq_len` 时立即报错。
