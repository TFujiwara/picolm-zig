const std = @import("std");

// Zen 2 optimizations - 32MB L3, CCX-aware
pub const Vec8f = @Vector(8, f32);

pub const Zen2Config = struct {
    pub const L3_SIZE = 32 * 1024 * 1024;
    pub const CCX_CORES = 4;
    pub const NUM_CCX = 2;

    // Block sizes optimized for 16MB L3 per CCX
    pub const BLOCK_M = 256;
    pub const BLOCK_N = 1024;
    pub const BLOCK_K = 64;
};

pub inline fn dotProductQ4KZen2(q4k_data: []const u8, x: []const f32, n: usize) f32 {
    // Similar a AVX2 pero con unroll agresivo para saturar pipelines
    const Block = extern struct {
        scales: [12]u8,
        qs: [64]u8,
    };

    const blocks = @as([*]const Block, @ptrCast(q4k_data))[0..n];
    var sum0 = @as(Vec8f, @splat(0));
    var sum1 = @as(Vec8f, @splat(0));
    var sum2 = @as(Vec8f, @splat(0));
    var sum3 = @as(Vec8f, @splat(0));

    var b: usize = 0;
    while (b + 3 < blocks.len) : (b += 4) {
        inline for (0..4) |i| {
            const block = &blocks[b + i];
            const scales = decodeScalesZen2(block.scales);
            const x_base = (b + i) * 256;

            comptime var group = 0;
            inline while (group < 8) : (group += 1) {
                const g = group * 8;

                var weights: [8]f32 = undefined;
                var w: usize = 0;
                while (w < 8) : (w += 1) {
                    const byte = block.qs[g + w];
                    const scale_idx = (g + w) / 16;
                    weights[w] = @as(f32, @floatFromInt(byte & 0xF)) * scales[scale_idx] + scales[scale_idx + 4];
                    // High nibble
                    weights[w] = @as(f32, @floatFromInt(byte >> 4)) * scales[scale_idx] + scales[scale_idx + 4];
                }

                const x_vec: Vec8f = x[x_base + g ..][0..8].*;
                const w_vec: Vec8f = weights;

                switch (i) {
                    0 => sum0 += x_vec * w_vec,
                    1 => sum1 += x_vec * w_vec,
                    2 => sum2 += x_vec * w_vec,
                    3 => sum3 += x_vec * w_vec,
                    else => unreachable,
                }
            }
        }
    }

    // Tail
    while (b < blocks.len) : (b += 1) {
        // Proceso estándar
    }

    const total = sum0 + sum1 + sum2 + sum3;
    return @reduce(.Add, total);
}

fn decodeScalesZen2(data: [12]u8) [8]f32 {
    var result: [8]f32 = undefined;
    for (0..4) |i| {
        result[i] = @as(f32, @floatFromInt(data[i])) / 64.0;
        result[i + 4] = @as(f32, @floatFromInt(data[i + 6])) / 64.0;
    }
    return result;
}

pub fn matmulL3OptimizedZen2(
    out: []f32,
    x: []const f32,
    w: []const u8,
    n: usize,
    d: usize,
) void {
    const BLOCK_M = Zen2Config.BLOCK_M;
    const BLOCK_N = Zen2Config.BLOCK_N;
    const BLOCK_K = Zen2Config.BLOCK_K;

    var m: usize = 0;
    while (m < n) : (m += BLOCK_M) {
        const m_end = @min(m + BLOCK_M, n);

        var k: usize = 0;
        while (k < d) : (k += BLOCK_K) {
            const k_end = @min(k + BLOCK_K, d);

            // Prefetch próximo bloque
            prefetchL3Zen2(&w[(m_end * d)..]);

            var n_idx: usize = 0;
            while (n_idx < d) : (n_idx += BLOCK_N) {
                const n_end = @min(n_idx + BLOCK_N, d);
                microKernelZen2(out, x, w, m, m_end, k, k_end, n_idx, n_end);
            }
        }
    }
}

fn microKernelZen2(
    out: []f32,
    x: []const f32,
    w: []const u8,
    m_start: usize,
    m_end: usize,
    k_start: usize,
    k_end: usize,
    n_start: usize,
    n_end: usize,
) void {
    _ = out;
    _ = x;
    _ = w;
    _ = m_start;
    _ = m_end;
    _ = k_start;
    _ = k_end;
    _ = n_start;
    _ = n_end;
    // Implementación específica
}

inline fn prefetchL3Zen2(addr: *const anyopaque) void {
    asm volatile ("prefetchw (%[addr])"
        :
        : [addr] "r" (addr),
        : "memory"
    );
}

pub fn pinToCCX(thread_id: usize, ccx_id: usize) void {
    const base_core = ccx_id * 4;
    const core = base_core + (thread_id % 4);

    var cpu_set: std.c.cpu_set_t = undefined;
    std.c.CPU_ZERO(&cpu_set);
    std.c.CPU_SET(core, &cpu_set);

    _ = std.c.pthread_setaffinity_np(std.c.pthread_self(), @sizeOf(std.c.cpu_set_t), &cpu_set);
}
