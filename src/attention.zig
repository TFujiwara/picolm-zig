const std = @import("std");
const types = @import("types.zig");
const simd = @import("simd.zig");

pub const FlashAttention = struct {
    pub fn forward(
        q: []const f32,
        k_cache: []const types.F16,
        v_cache: []const types.F16,
        out: []f32,
        pos: usize,
        head_dim: usize,
    ) void {
        const seq_len = pos + 1;
        const block_size = 64;

        var m: f32 = -std.math.inf(f32);
        var l: f32 = 0;
        var acc = [_]f32{0} ** 2048;

        var j: usize = 0;
        while (j < seq_len) : (j += block_size) {
            const end = @min(j + block_size, seq_len);

            var s_block: [64]f32 = undefined;
            var t = j;
            while (t < end) : (t += 1) {
                s_block[t - j] = computeScore(q, k_cache, t, head_dim);
            }

            var m_new = m;
            t = j;
            while (t < end) : (t += 1) {
                if (s_block[t - j] > m_new) m_new = s_block[t - j];
            }

            const exp_scale = @exp(m - m_new);
            l *= exp_scale;

            var d: usize = 0;
            while (d < head_dim) : (d += 1) acc[d] *= exp_scale;

            t = j;
            while (t < end) : (t += 1) {
                const p = @exp(s_block[t - j] - m_new);
                l += p;

                const v_offset = t * head_dim;
                d = 0;
                while (d < head_dim) : (d += 1) {
                    acc[d] += p * types.fp16ToFp32(v_cache[v_offset + d]);
                }
            }

            m = m_new;
        }

        const inv_l = 1.0 / l;
        var d: usize = 0;
        while (d < head_dim) : (d += 1) out[d] = acc[d] * inv_l;
    }

    fn computeScore(q: []const f32, k_cache: []const types.F16, t: usize, head_dim: usize) f32 {
        const k_offset = t * head_dim;
        var sum_vec = @as(simd.Vec8f, @splat(0));

        const vec_dim = head_dim - (head_dim % 8);
        var d: usize = 0;
        while (d < vec_dim) : (d += 8) {
            const qv: simd.Vec8f = q[d..][0..8].*;
            var kv: [8]f32 = undefined;
            for (0..8) |k| kv[k] = types.fp16ToFp32(k_cache[k_offset + d + k]);
            sum_vec += qv * @as(simd.Vec8f, kv);
        }

        var sum = @reduce(.Add, sum_vec);
        while (d < head_dim) : (d += 1) sum += q[d] * types.fp16ToFp32(k_cache[k_offset + d]);

        return sum / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    }
};
