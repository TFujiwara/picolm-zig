const std = @import("std");

pub const Vec8f = @Vector(8, f32);
pub const Vec16f = @Vector(16, f32);

// AVX2 Q4_K
pub inline fn dotProductQ4KAVX2(q4k_data: []const u8, x: []const f32, n: usize) f32 {
    const Block = extern struct {
        scales: [12]u8,
        qs: [64]u8,
    };

    const blocks = @as([*]const Block, @ptrCast(q4k_data))[0..n];
    var sum0 = @as(Vec8f, @splat(0));
    var sum1 = @as(Vec8f, @splat(0));

    for (blocks, 0..) |*block, b_idx| {
        const scales = decodeScalesAVX2(block.scales);
        const x_base = b_idx * 256;

        var i: usize = 0;
        while (i < 64) : (i += 8) {
            var weights0: [8]f32 = undefined;
            var weights1: [8]f32 = undefined;

            for (0..8) |j| {
                const byte = block.qs[i + j];
                const scale_idx = (i + j) / 16;
                weights0[j] = @as(f32, @floatFromInt(byte & 0xF)) * scales[scale_idx] + scales[scale_idx + 4];
                weights1[j] = @as(f32, @floatFromInt(byte >> 4)) * scales[scale_idx] + scales[scale_idx + 4];
            }

            const x0: Vec8f = x[x_base + i * 2 ..][0..8].*;
            const x1: Vec8f = x[x_base + i * 2 + 8 ..][0..8].*;

            sum0 += x0 * weights0;
            sum1 += x1 * weights1;
        }
    }

    return @reduce(.Add, sum0) + @reduce(.Add, sum1);
}

fn decodeScalesAVX2(data: [12]u8) [8]f32 {
    var result: [8]f32 = undefined;
    for (0..4) |i| {
        result[i] = @as(f32, @floatFromInt(data[i])) / 64.0;
        result[i + 4] = @as(f32, @floatFromInt(data[i + 6])) / 64.0;
    }
    return result;
}

// AVX-512 Q4_K
pub inline fn dotProductQ4KAVX512(q4k_data: []const u8, x: []const f32, n: usize) f32 {
    const Block = extern struct {
        scales: [12]u8,
        qs: [64]u8,
    };

    const blocks = @as([*]const Block, @ptrCast(q4k_data))[0..n];
    var sum0 = @as(Vec16f, @splat(0));
    var sum1 = @as(Vec16f, @splat(0));
    var sum2 = @as(Vec16f, @splat(0));
    var sum3 = @as(Vec16f, @splat(0));

    for (blocks, 0..) |*block, b_idx| {
        const scales = decodeScalesAVX512(block.scales);
        const x_base = b_idx * 256;

        // Procesar 256 elementos en 16 grupos de 16
        comptime var group: usize = 0;
        inline while (group < 4) : (group += 1) {
            const g = group * 16;

            var weights: [16]f32 = undefined;
            var w: usize = 0;
            while (w < 16) : (w += 1) {
                const byte_idx = g / 2 + w / 2;
                const byte = block.qs[byte_idx];
                const nibble = if (w % 2 == 0) byte & 0xF else byte >> 4;
                const scale_idx = (g + w) / 32;
                weights[w] = @as(f32, @floatFromInt(nibble)) * scales[scale_idx] + scales[scale_idx + 4];
            }

            const x_vec: Vec16f = x[x_base + g ..][0..16].*;
            const w_vec: Vec16f = weights;

            switch (group) {
                0 => sum0 += x_vec * w_vec,
                1 => sum1 += x_vec * w_vec,
                2 => sum2 += x_vec * w_vec,
                3 => sum3 += x_vec * w_vec,
                else => unreachable,
            }
        }
    }

    const total = sum0 + sum1 + sum2 + sum3;
    return @reduce(.Add, total);
}

fn decodeScalesAVX512(data: [12]u8) [8]f32 {
    return decodeScalesAVX2(data); // Misma decodificación
}
