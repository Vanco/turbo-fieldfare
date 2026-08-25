#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qwen.metal — Qwen 3.5 forward-pass kernels the Gemma stack cannot express:
//
//   qwen_attn_qk_epilogue     Per-head Q/K RMSNorm + partial-segment NeoX RoPE.
//                             Q carries [head | gate] halves; only the head half
//                             is normalized/roped and compacted. V passes through.
//   qwen_sigmoid_gate_mul     attention_out *= sigmoid(gate half) per q head.
//   qwen_router_topk_softmax_all  Softmax over ALL experts, then top-k select;
//                             routing weights keep their global probabilities.
//   qwen_gdn_forward          Fused gated-delta-net linear attention step/chunk:
//                             causal conv -> silu -> q/k RMS scaling ->
//                             delta-rule state update -> gated output norm.
//   qwen_residual_add_batch   hidden += projection (token-major batches).
//   qwen_ffn_combine_batch    hidden += routed + shared * sigmoid(scalar gate).
//
// All kernels are runtime-parameterized (no function constants): the Qwen path
// is not part of the tuned Gemma decode pipeline and PSO specialization is not
// worth the cache complexity here.
// ============================================================================

constant constexpr uint kQwenEpilogueThreads = 256;
constant constexpr uint kQwenMaxHeadDim = 512;
constant constexpr uint kQwenRouterThreads = 256;
constant constexpr uint kQwenMaxExperts = 256;
constant constexpr uint kQwenGdnThreads = 128;
constant constexpr uint kQwenGdnLDim = 128;

static inline float qwen_softplus(float x) {
    // log(1 + exp(x)) with the linear branch for large x.
    return x > 20.0f ? x : log(1.0f + exp(x));
}

static inline void qwen_rope_segment_pair(thread float& x0,
                                          thread float& x1,
                                          uint pair_index,
                                          uint rotated_pairs,
                                          float position,
                                          float theta_base) {
    // Segment RoPE: pair i couples dims (i, i + rotated_pairs) inside the
    // first 2*rotated_pairs dims of the head; frequency divisor is the
    // rotation segment length, not the full head dimension.
    const float exponent = -float(2u * pair_index) / float(2u * rotated_pairs);
    const float angle = position * pow(theta_base, exponent);
    const float c = cos(angle);
    const float s = sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// Two-stage block sum (mirrors rmsnorm.metal's reduction order).
static inline float qwen_block_sum(threadgroup float* partial,
                                   float value,
                                   uint lid,
                                   uint simd_lane_id,
                                   uint simd_group_id,
                                   uint simdgroups) {
    value = simd_sum(value);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// ============================================================================
// qwen_attn_qk_epilogue — one threadgroup owns one logical head for one token.
//
// Grid: ((numQHeads + numKVHeads), tokenCount). Q heads read their [head|gate]
// halves from qRaw, apply weighted RMSNorm to the head half, RoPE the first
// 2*rotatedPairs dims, and write the compact result to qOut. Gate halves are
// left untouched in qRaw for qwen_sigmoid_gate_mul. K heads normalize + rope
// into kOut (decode: the KV slot; prefill: a staged buffer). V is untouched —
// Qwen 3.5 has no v-norm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kQwenEpilogueThreads)]]
void qwen_attn_qk_epilogue(
    device const half*   q_raw          [[buffer(0)]],
    device const half*   k_raw          [[buffer(1)]],
    device const bfloat* q_weight       [[buffer(2)]],
    device const bfloat* k_weight       [[buffer(3)]],
    device       half*   q_out          [[buffer(4)]],
    device       half*   k_out          [[buffer(5)]],
    constant     uint&   head_dim       [[buffer(6)]],
    constant     uint&   num_q_heads    [[buffer(7)]],
    constant     uint&   num_kv_heads   [[buffer(8)]],
    constant     uint&   rotated_pairs  [[buffer(9)]],
    constant     float&  theta_base     [[buffer(10)]],
    constant     uint&   position_base  [[buffer(11)]],
    constant     float&  rms_eps        [[buffer(12)]],
    constant     uint&   q_in_stride    [[buffer(13)]],
    constant     uint&   k_in_stride    [[buffer(14)]],
    constant     uint&   q_out_stride   [[buffer(15)]],
    constant     uint&   k_out_stride   [[buffer(16)]],
    constant     uint&   k_out_offset   [[buffer(17)]],
    uint   gid            [[threadgroup_position_in_grid]],
    uint   lid            [[thread_position_in_threadgroup]],
    uint   lsize          [[threads_per_threadgroup]],
    uint   simd_lane_id   [[thread_index_in_simdgroup]],
    uint   simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint   simdgroups     [[simdgroups_per_threadgroup]])
{
    threadgroup half  staged[kQwenMaxHeadDim];
    threadgroup float partial[kQwenEpilogueThreads / 32];

    const uint total_heads = num_q_heads + num_kv_heads;
    const uint gx = gid % total_heads;
    const uint token = gid / total_heads;
    const bool is_q = gx < num_q_heads;
    const bool is_k = !is_q && gx < total_heads;
    if (!is_q && !is_k) { return; }

    const uint HD = head_dim;
    const uint local_head = is_q ? gx : gx - num_q_heads;
    device const half* src = is_q
        ? (q_raw + token * q_in_stride + local_head * 2u * HD)
        : (k_raw + token * k_in_stride + local_head * HD);

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]);
        acc = fma(xv, xv, acc);
    }
    const float ssq = qwen_block_sum(partial, acc, lid, simd_lane_id,
                                     simd_group_id, simdgroups);
    const float inv = rsqrt(ssq / float(HD) + rms_eps);

    device const bfloat* w = is_q ? q_weight : k_weight;
    for (uint i = lid; i < HD; i += lsize) {
        staged[i] = half(float(src[i]) * inv * float(w[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rp = rotated_pairs;
    for (uint pair = lid; pair < rp; pair += lsize) {
        float x0 = float(staged[pair]);
        float x1 = float(staged[pair + rp]);
        qwen_rope_segment_pair(x0, x1, pair, rp,
                               float(position_base + token), theta_base);
        staged[pair] = half(x0);
        staged[pair + rp] = half(x1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* dst = is_q
        ? (q_out + token * q_out_stride + local_head * HD)
        : (k_out + k_out_offset / sizeof(half)
           + token * k_out_stride + local_head * HD);
    for (uint i = lid; i < HD; i += lsize) {
        dst[i] = staged[i];
    }
}

// ============================================================================
// qwen_sigmoid_gate_mul — elementwise attention-output gating.
//
// attn[idx] *= sigmoid(q_raw gate half). One dimension maps to
// (token, head, dim); the gate lives at q_raw[token*qInStride +
// head*2*headDim + headDim + dim].
// ============================================================================

[[kernel]]
void qwen_sigmoid_gate_mul(
    device       half* attn           [[buffer(0)]],
    device const half* q_raw          [[buffer(1)]],
    constant     uint& head_dim       [[buffer(2)]],
    constant     uint& num_q_heads    [[buffer(3)]],
    constant     uint& token_count    [[buffer(4)]],
    constant     uint& q_in_stride    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const uint per_token = num_q_heads * head_dim;
    const uint total = token_count * per_token;
    if (tid >= total) { return; }
    const uint token = tid / per_token;
    const uint rem = tid % per_token;
    const uint head = rem / head_dim;
    const uint dim = rem % head_dim;
    const float gate = float(q_raw[token * q_in_stride
                                   + head * 2u * head_dim
                                   + head_dim + dim]);
    attn[tid] = half(float(attn[tid]) * (1.0f / (1.0f + exp(-gate))));
}

// ============================================================================
// qwen_router_topk_softmax_all — Qwen routing semantics.
//
// One threadgroup per token. The gemma router selects top-k FIRST and
// softmaxes the winners; Qwen softmaxes over ALL experts and then takes the
// top-k probabilities without renormalization. Logits are scaled by
// D^-0.5 here because the plain int4 GEMV produces unscaled projections.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kQwenRouterThreads)]]
void qwen_router_topk_softmax_all(
    device const half* logits        [[buffer(0)]],
    device       uint* out_indices   [[buffer(1)]],
    device       half* out_weights   [[buffer(2)]],
    constant     uint& num_experts   [[buffer(3)]],
    constant     uint& top_k         [[buffer(4)]],
    constant     float& input_scale  [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint simdgroups     [[simdgroups_per_threadgroup]])
{
    threadgroup float probs[kQwenMaxExperts];
    threadgroup float partial_val[kQwenRouterThreads / 32];
    threadgroup uint  partial_idx[kQwenRouterThreads / 32];

    device const half* row = logits + token * num_experts;

    float m = -INFINITY;
    for (uint i = lid; i < num_experts; i += lsize) {
        float x = float(row[i]) * input_scale;
        probs[i] = x;
        m = max(m, x);
    }
    m = qwen_block_sum(partial_val, m, lid, simd_lane_id, simd_group_id, simdgroups);

    float denom = 0.0f;
    for (uint i = lid; i < num_experts; i += lsize) {
        float p = exp(probs[i] - m);
        probs[i] = p;
        denom += p;
    }
    denom = qwen_block_sum(partial_val, denom, lid, simd_lane_id, simd_group_id, simdgroups);

    for (uint i = lid; i < num_experts; i += lsize) {
        probs[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < top_k; ++k) {
        float best_val = -1.0f;
        uint best_idx = 0;
        for (uint i = lid; i < num_experts; i += lsize) {
            if (probs[i] > best_val) {
                best_val = probs[i];
                best_idx = i;
            }
        }
        // Paired arg-max reduction over (value, index).
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float ov = simd_shuffle_xor(best_val, offset);
            uint oi = simd_shuffle_xor(best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (simd_lane_id == 0) {
            partial_val[simd_group_id] = best_val;
            partial_idx[simd_group_id] = best_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float wval = partial_val[0];
            uint widx = partial_idx[0];
            for (uint s = 1; s < simdgroups; ++s) {
                if (partial_val[s] > wval
                    || (partial_val[s] == wval && partial_idx[s] < widx)) {
                    wval = partial_val[s];
                    widx = partial_idx[s];
                }
            }
            out_indices[token * top_k + k] = widx;
            out_weights[token * top_k + k] = half(wval);
            probs[widx] = -2.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// qwen_gdn_forward — fused gated-delta-net layer over T sequential tokens.
//
// One threadgroup per value head (32 groups, 128 threads). Each token runs:
//   1. depthwise causal conv (kernel 4) over this head's q/k/v channels,
//      then silu — the conv history rolls through threadgroup memory during
//      the chunk and syncs with the persistent device state at the edges;
//   2. beta = sigmoid(b_h), gamma = exp(-exp(A_log_h)*softplus(a_h + dt_h));
//   3. q/k RMS normalization with the 128^-1 / 128^-0.5 scale factors;
//   4. delta rule: S = gamma*S + outer(k, beta*(v - S^T*k)), o = S^T*q;
//   5. out = silu(z) * rmsnorm(o) per value head.
// Column-owner layout: thread m owns state column m, so all three state
// passes stream consecutive addresses across the threadgroup.
// Even-headed groups persist the shared q/k conv windows; every group
// persists its own v window.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kQwenGdnThreads)]]
void qwen_gdn_forward(
    device const half*   mixed          [[buffer(0)]],   // [T, 8192]
    device const half*   z              [[buffer(1)]],   // [T, 4096]
    device const half*   a_vec          [[buffer(2)]],   // [T, 32]
    device const half*   b_vec          [[buffer(3)]],   // [T, 32]
    device const bfloat* conv_weight    [[buffer(4)]],   // [8192, 4]
    device       half*   conv_state     [[buffer(5)]],   // [3, 8192]
    device       float*  state          [[buffer(6)]],   // [heads, 128, 128]
    device const float*  a_log          [[buffer(7)]],   // [32]
    device const bfloat* dt_bias        [[buffer(8)]],   // [32]
    device const bfloat* norm_weight    [[buffer(9)]],   // [128]
    device       half*   out            [[buffer(10)]],  // [T, 4096]
    constant     uint&   token_count    [[buffer(11)]],
    constant     uint&   value_heads    [[buffer(12)]],
    constant     uint&   key_heads      [[buffer(13)]],
    constant     uint&   conv_dim       [[buffer(14)]],
    constant     uint&   kernel_dim     [[buffer(15)]],
    constant     float&  rms_eps        [[buffer(16)]],
    constant     float&  input_scale    [[buffer(17)]],  // linDim^-0.5
    uint  head           [[threadgroup_position_in_grid]],
    uint  lid            [[thread_position_in_threadgroup]],
    uint  lsize          [[threads_per_threadgroup]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]])
{
    const uint L = kQwenGdnLDim;
    const uint group = head / 2u;                    // key-head slice
    const bool even_head = (head % 2u) == 0u;
    const uint q_chan = group * L;                   // shared by the head pair
    const uint k_chan = key_heads * L + group * L;
    const uint v_chan = 2u * key_heads * L + head * L;

    // Rolling conv windows for this group's three channel segments.
    threadgroup float hist[3][3u * kQwenGdnLDim];
    threadgroup float qv[kQwenGdnLDim];
    threadgroup float kv[kQwenGdnLDim];
    threadgroup float vv[kQwenGdnLDim];
    threadgroup float rbuf[kQwenGdnLDim];
    threadgroup float ubuf[kQwenGdnLDim];
    threadgroup float obuf[kQwenGdnLDim];
    threadgroup float partial[kQwenGdnThreads / 32];

    if (lid < L) {
        hist[0][lid] = float(conv_state[0 * conv_dim + q_chan + lid]);
        hist[0][L + lid] = float(conv_state[0 * conv_dim + k_chan + lid]);
        hist[0][2u * L + lid] = float(conv_state[0 * conv_dim + v_chan + lid]);
        hist[1][lid] = float(conv_state[1 * conv_dim + q_chan + lid]);
        hist[1][L + lid] = float(conv_state[1 * conv_dim + k_chan + lid]);
        hist[1][2u * L + lid] = float(conv_state[1 * conv_dim + v_chan + lid]);
        hist[2][lid] = float(conv_state[2 * conv_dim + q_chan + lid]);
        hist[2][L + lid] = float(conv_state[2 * conv_dim + k_chan + lid]);
        hist[2][2u * L + lid] = float(conv_state[2 * conv_dim + v_chan + lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* S = state + head * L * L;

    for (uint t = 0; t < token_count; ++t) {
        device const half* mixed_t = mixed + t * conv_dim;
        // Per-token scalar gates.
        const float gamma_decay =
            exp(-exp(a_log[head])
                * qwen_softplus(float(a_vec[t * value_heads + head])
                                + float(dt_bias[head])));
        const float beta_scale =
            1.0f / (1.0f + exp(-float(b_vec[t * value_heads + head])));

        // Conv + silu for the three segments. Threads cover L channels each;
        // lsize == L in production, the loop keeps other shapes correct.
        for (uint c = lid; c < L; c += lsize) {
            float seg[3];
            const uint chans[3] = { q_chan, k_chan, v_chan };
            for (uint seg_i = 0; seg_i < 3; ++seg_i) {
                const uint chan = chans[seg_i] + c;
                float acc = float(conv_weight[chan * kernel_dim + 0]) * hist[0][seg_i * L + c]
                          + float(conv_weight[chan * kernel_dim + 1]) * hist[1][seg_i * L + c]
                          + float(conv_weight[chan * kernel_dim + 2]) * hist[2][seg_i * L + c]
                          + float(conv_weight[chan * kernel_dim + 3]) * float(mixed_t[chan]);
                const float y = acc / (1.0f + exp(-acc));  // silu
                if (seg_i == 0) { qv[c] = y; }
                else if (seg_i == 1) { kv[c] = y; }
                else { vv[c] = y; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Roll the raw-token window forward (raw inputs, pre-conv).
        if (lid < L) {
            hist[0][lid] = hist[1][lid];
            hist[0][L + lid] = hist[1][L + lid];
            hist[0][2u * L + lid] = hist[1][2u * L + lid];
            hist[1][lid] = hist[2][lid];
            hist[1][L + lid] = hist[2][L + lid];
            hist[1][2u * L + lid] = hist[2][2u * L + lid];
            hist[2][lid] = float(mixed_t[q_chan + lid]);
            hist[2][L + lid] = float(mixed_t[k_chan + lid]);
            hist[2][2u * L + lid] = float(mixed_t[v_chan + lid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // q/k RMS normalization with asymmetric scale factors. Production
        // launches exactly L threads, so every lane participates uniformly.
        {
            float acc = 0.0f;
            for (uint c = lid; c < L; c += lsize) { acc = fma(qv[c], qv[c], acc); }
            const float ssq = qwen_block_sum(partial, acc, lid, simd_lane_id,
                                             simd_group_id, simdgroups);
            const float inv = rsqrt(ssq + rms_eps);  // L2 norm (Qwen3.5)
            for (uint c = lid; c < L; c += lsize) {
                qv[c] = qv[c] * inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            float acc = 0.0f;
            for (uint c = lid; c < L; c += lsize) { acc = fma(kv[c], kv[c], acc); }
            const float ssq = qwen_block_sum(partial, acc, lid, simd_lane_id,
                                             simd_group_id, simdgroups);
            const float inv = rsqrt(ssq + rms_eps);  // L2 norm (Qwen3.5)
            for (uint c = lid; c < L; c += lsize) {
                kv[c] = kv[c] * inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Retrieval: r[m] = gamma * sum_j S[j][m] * k[j].
        {
            float acc = 0.0f;
            for (uint j = lid; j < L; j += lsize) {
                acc = fma(S[j * L + lid], kv[j], acc);
            }
            rbuf[lid] = gamma_decay * acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write slot: u[m] = beta * (v[m] - r[m]).
        if (lid < L) {
            ubuf[lid] = beta_scale * (vv[lid] - rbuf[lid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update + output in one streaming pass per column owner:
        //   S'[j][m] = gamma*S[j][m] + k[j]*u[m]
        //   o[m]     = sum_j S'[j][m] * q[j]
        {
            float acc = 0.0f;
            const float u_m = ubuf[lid];
            for (uint j = 0; j < L; ++j) {
                float s = fma(gamma_decay, S[j * L + lid], kv[j] * u_m);
                S[j * L + lid] = s;
                acc = fma(s, qv[j], acc);
            }
            obuf[lid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Gated output norm: out = silu(z) * rmsnorm(o) * weight.
        {
            float acc = 0.0f;
            for (uint c = lid; c < L; c += lsize) { acc = fma(obuf[c], obuf[c], acc); }
            const float ssq = qwen_block_sum(partial, acc, lid, simd_lane_id,
                                             simd_group_id, simdgroups);
            const float inv = rsqrt(ssq / float(L) + rms_eps);  // g_norm (RMSNorm)
            const float zg = float(z[t * (value_heads * L) + head * L + lid]);
            const float silu_z = zg / (1.0f + exp(-zg));
            out[t * (value_heads * L) + head * L + lid] =
                half(silu_z * obuf[lid] * input_scale * inv * float(norm_weight[lid]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist the conv windows. Even heads own the shared q/k segments so the
    // paired threadgroups never race on the same channels.
    if (lid < L) {
        if (even_head) {
            conv_state[0 * conv_dim + q_chan + lid] = half(hist[0][lid]);
            conv_state[1 * conv_dim + q_chan + lid] = half(hist[1][lid]);
            conv_state[2 * conv_dim + q_chan + lid] = half(hist[2][lid]);
            conv_state[0 * conv_dim + k_chan + lid] = half(hist[0][L + lid]);
            conv_state[1 * conv_dim + k_chan + lid] = half(hist[1][L + lid]);
            conv_state[2 * conv_dim + k_chan + lid] = half(hist[2][L + lid]);
        }
        conv_state[0 * conv_dim + v_chan + lid] = half(hist[0][2u * L + lid]);
        conv_state[1 * conv_dim + v_chan + lid] = half(hist[1][2u * L + lid]);
        conv_state[2 * conv_dim + v_chan + lid] = half(hist[2][2u * L + lid]);
    }
}

// ============================================================================
// qwen_residual_add_batch — hidden[t][:] += delta[t][:].
// ============================================================================

[[kernel]]
void qwen_residual_add_batch(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) { return; }
    hidden[tid] = half(float(hidden[tid]) + float(delta[tid]));
}

// ============================================================================
// qwen_ffn_combine_batch — hidden[t][:] += routed[t][:] + shared[t][:] *
// sigmoid(gate[t]). The shared-expert scalar gate comes from a [1, D]
// int4 GEMV written per token.
// ============================================================================

[[kernel]]
void qwen_ffn_combine_batch(
    device half*       hidden [[buffer(0)]],
    device const half* routed [[buffer(1)]],
    device const half* shared [[buffer(2)]],
    device const half* gate   [[buffer(3)]],
    constant uint&     d      [[buffer(4)]],
    constant uint&     count  [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) { return; }
    const float g = float(gate[tid % d]);
    const float s = 1.0f / (1.0f + exp(-g));
    hidden[tid] = half(float(hidden[tid])
                       + float(routed[tid]) + float(shared[tid]) * s);
}
