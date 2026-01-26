#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3-D.");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3-D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3-D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3-D.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttention: batch size mismatch.");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "SelfAttention: seqlen mismatch.");
    ASSERT(attn_val->shape()[2] == q->shape()[2], "SelfAttention: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == k->shape()[2], "SelfAttention: head dim mismatch.");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "SelfAttention: head dim mismatch.");
    // ASSERT(q->dtype() == LLAISYS_DTYPE_F32, "SelfAttention: q must be float32.");
    // ASSERT(k->dtype() == LLAISYS_DTYPE_F32, "SelfAttention: k must be float32.");
    // ASSERT(v->dtype() == LLAISYS_DTYPE_F32, "SelfAttention: v must be float32.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val, q, k, v, scale);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
