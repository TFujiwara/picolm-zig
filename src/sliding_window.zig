const std = @import("std");
const types = @import("types.zig");

pub const SlidingWindowAttention = struct {
    window_size: usize,

    pub fn forward(
        self: SlidingWindowAttention,
        q: []const f32,
        k_cache: []const types.F16,
        v_cache: []const types.F16,
        out: []f32,
        pos: usize,
        head_dim: usize,
    ) void {
        const start_pos = if (pos > self.window_size) pos - self.window_size else 0;

        var m: f32 = -std.math.inf(f32);
        var l: f32 = 0;
        var acc = [_]f32{0} ** 2048;

        var j = start_pos;
        while (j <= pos) : (j += 1) {
            const score = computeScore(q, k_cache, j, head_dim);

            if (score > m) {
                const scale = @exp(m - score);
                l *= scale;
                for (0..head_dim) |d| acc[d] *= scale;
                m = score;
            }

            const p = @exp(score - m);
            l += p;

            const v_offset = j * head_dim;
            for (0..head_dim) |d| {
                acc[d] += p * types.fp16ToFp32(v_cache[v_offset + d]);
            }
        }

        const inv_l = 1.0 / l;
        for (0..head_dim) |d| out[d] = acc[d] * inv_l;
    }

    fn computeScore(q: []const f32, k_cache: []const types.F16, pos: usize, head_dim: usize) f32 {
        const k_offset = pos * head_dim;
        var sum: f32 = 0;
        for (0..head_dim) |i| {
            sum += q[i] * types.fp16ToFp32(k_cache[k_offset + i]);
        }
        return sum / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    }
};
