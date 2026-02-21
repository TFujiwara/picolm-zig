const std = @import("std");

// Bay Trail (Silvermont) - Solo SSE4.1/4.2, sin AVX
pub const Vec4f = @Vector(4, f32);

pub inline fn dotProductQ4KSSE4(q4k_data: []const u8, x: []const f32, n: usize) f32 {
    const Block = extern struct {
        scales: [12]u8,
        qs: [64]u8,
    };

    const blocks = @as([*]const Block, @ptrCast(q4k_data))[0..n];
    var sum0 = @as(Vec4f, @splat(0));
    var sum1 = @as(Vec4f, @splat(0));
    var sum2 = @as(Vec4f, @splat(0));
    var sum3 = @as(Vec4f, @splat(0));

    for (blocks, 0..) |*block, b_idx| {
        const scales = decodeScalesSSE4(block.scales);
        const x_base = b_idx * 256;

        // Procesar 256 elementos en 64 grupos de 4
        var i: usize = 0;
        while (i < 64) : (i += 4) {
            inline for (0..4) |u| {
                const idx = i + u;
                const byte = block.qs[idx];
                const scale_idx = idx / 16;
                const scale = @as(Vec4f, @splat(scales[scale_idx]));
                const min = @as(Vec4f, @splat(scales[scale_idx + 4]));

                var weights: [4]f32 = undefined;
                weights[0] = @as(f32, @floatFromInt(byte & 0xF));
                weights[1] = @as(f32, @floatFromInt(byte >> 4));
                weights[2] = 0;
                weights[3] = 0; // Padding

                const w_vec = @as(Vec4f, weights) * scale + min;
                const x_vec: Vec4f = x[x_base + idx * 2 ..][0..4].*;

                switch (u) {
                    0 => sum0 += x_vec * w_vec,
                    1 => sum1 += x_vec * w_vec,
                    2 => sum2 += x_vec * w_vec,
                    3 => sum3 += x_vec * w_vec,
                    else => unreachable,
                }
            }
        }
    }

    return @reduce(.Add, sum0) + @reduce(.Add, sum1) + @reduce(.Add, sum2) + @reduce(.Add, sum3);
}

pub inline fn rmsNormSSE4(o: []f32, x: []const f32, weight: []const f32) void {
    const n = x.len;
    const vec_n = n - (n % 4);

    var sum_vec = @as(Vec4f, @splat(0));
    var i: usize = 0;

    while (i < vec_n) : (i += 4) {
        const xv: Vec4f = x[i..][0..4].*;
        sum_vec += xv * xv;
    }

    var ss = @reduce(.Add, sum_vec);
    while (i < n) : (i += 1) ss += x[i] * x[i];

    ss /= @as(f32, @floatFromInt(n));
    ss += 1e-5;
    const norm = 1.0 / std.math.sqrt(ss);
    const norm_vec = @as(Vec4f, @splat(norm));

    i = 0;
    while (i < vec_n) : (i += 4) {
        const xv: Vec4f = x[i..][0..4].*;
        const wv: Vec4f = weight[i..][0..4].*;
        o[i..][0..4].* = xv * norm_vec * wv;
    }
    while (i < n) : (i += 1) o[i] = x[i] * norm * weight[i];
}

fn decodeScalesSSE4(data: [12]u8) [8]f32 {
    var result: [8]f32 = undefined;
    for (0..4) |i| {
        result[i] = @as(f32, @floatFromInt(data[i])) / 64.0;
        result[i + 4] = @as(f32, @floatFromInt(data[i + 6])) / 64.0;
    }
    return result;
}

pub inline fn prefetchBayTrail(addr: *const anyopaque) void {
    asm volatile ("prefetcht0 (%[addr])"
        :
        : [addr] "r" (addr),
        : "memory"
    );
}
