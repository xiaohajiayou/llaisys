#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

template<typename T>
void self_attention_kernel(
    T* out,
    const T* q,
    const T* k,
    const T* v,
    size_t qlen,
    size_t kvlen,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale)
{
    size_t group_size = nhead / nkvhead;

    std::vector<float> scores(kvlen);

    for (size_t t = 0; t < qlen; ++t) {
        for (size_t h = 0; h < nhead; ++h) {

            size_t kv_h = h / group_size;

            const T* q_ptr = q + (t * nhead + h) * d;
            // const T* k_base = k + kv_h * d;
            // const T* v_base = v + kv_h * dv;
            T* out_ptr = out + (t * nhead + h) * dv;

            // ---- 1. 计算 QK^T ----
            for (size_t tk = 0; tk < kvlen; ++tk) {
                if (tk > t) {         // causal mask
                    scores[tk] = -1e30f;
                    continue;
                }

                const T* k_ptr = k + (tk * nkvhead + kv_h) * d;

                float dot = 0.f;
                for (size_t i = 0; i < d; ++i) {
                    dot += llaisys::utils::cast<float>(q_ptr[i]) * llaisys::utils::cast<float>(k_ptr[i]);
                }
                scores[tk] = dot * scale;
            }

            // ---- 2. softmax ----
            float maxv = -1e30f;
            for (size_t i = 0; i < kvlen; ++i)
                maxv = std::max(maxv, scores[i]);

            float sum = 0.f;
            for (size_t i = 0; i < kvlen; ++i) {
                scores[i] = std::exp(scores[i] - maxv);
                sum += scores[i];
            }

            for (size_t i = 0; i < kvlen; ++i)
                scores[i] /= sum;

            // ---- 3. 加权 V ----
            for (size_t i = 0; i < dv; ++i) {
                float acc = 0.f;
                for (size_t tk = 0; tk < kvlen; ++tk) {
                    const T* v_ptr = v + (tk * nkvhead + kv_h) * dv;
                    acc += scores[tk] * llaisys::utils::cast<float>(v_ptr[i]);
                }
                out_ptr[i] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}
    

} // namespace llaisys::ops::cpu


namespace llaisys::ops::cpu {
void self_attention(tensor_t attn_val,
        tensor_t q,
        tensor_t k,
        tensor_t v,
        float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(attn_val->isContiguous() && q->isContiguous() &&
    k->isContiguous() && v->isContiguous(),
    "self_attention: tensors must be contiguous");

    size_t seqlen   = q->shape()[0];
    size_t nhead    = q->shape()[1];
    size_t d        = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t dv        = v->shape()[2];

    ASSERT(attn_val->shape()[0] == seqlen, "attn_val seqlen mismatch");
    ASSERT(attn_val->shape()[1] == nhead,  "attn_val head mismatch");
    ASSERT(attn_val->shape()[2] == dv,     "attn_val dv mismatch");

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_kernel(
            reinterpret_cast<float*>(attn_val->data()),
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k->data()),
            reinterpret_cast<const float*>(v->data()),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_kernel(
            reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t*>(q->data()),
            reinterpret_cast<const llaisys::bf16_t*>(k->data()),
            reinterpret_cast<const llaisys::bf16_t*>(v->data()),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_kernel(
            reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t*>(q->data()),
            reinterpret_cast<const llaisys::fp16_t*>(k->data()),
            reinterpret_cast<const llaisys::fp16_t*>(v->data()),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops::cpu