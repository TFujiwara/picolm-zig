const std = @import("std");
const types = @import("types.zig");

// Q2_K Block
pub const BlockQ2K = extern struct {
    scales: [16]u8,
    qs: [64]u8,
};

// Q4_K Block
pub const BlockQ4K = extern struct {
    scales: [12]u8,
    qs: [64]u8,
};

// Q8_0 Block
pub const BlockQ8_0 = extern struct {
    delta: types.F16,
    qs: [32]i8,
};

// Q8_K Block
pub const BlockQ8K = extern struct {
    d: f32,
    qs: [256]i8,
    bsums: [16]i16,
};

pub const Quantizer = union(enum) {
    q2_k: void,
    q3_k: void,
    q4_k: void,
    q5_k: void,
    q6_k: void,
    q8_0: void,
    q8_k: void,
    f16: void,
    f32: void,

    pub fn dotProduct(self: Quantizer, weights: []const u8, x: []const f32, n: usize) f32 {
        return switch (self) {
            .q2_k => dotProductQ2K(weights, x, n),
            .q4_k => dotProductQ4K(weights, x, n),
            .q8_0 => dotProductQ8_0(weights, x, n),
            .q8_k => dotProductQ8K(weights, x, n),
            .f16 => dotProductF16(weights, x, n),
            .f32 => dotProductF32(weights, x, n),
            else => dotProductQ4K(weights, x, n), // Fallback
        };
    }

    pub fn rowSize(self: Quantizer, dim: usize) usize {
        return switch (self) {
            .q2_k => (dim / 256) * 96,
            .q4_k => (dim / 256) * 144,
            .q8_0 => (dim / 32) * 34,
            .q8_k => (dim / 256) * 292,
            .f16 => dim * 2,
            .f32 => dim * 4,
            else => dim * 4,
        };
    }
};

// Q2_K scalar implementation
pub fn dotProductQ2K(q2k_data: []const u8, x: []const f32, n: usize) f32 {
    const num_blocks = n / 256;
    const blocks = @as([*]const BlockQ2K, @ptrCast(q2k_data))[0..num_blocks];
    var total: f32 = 0;

    for (blocks, 0..) |*block, b_idx| {
        const scales = decodeQ2KScales(block.scales);
        const x_offset = b_idx * 256;
        var sum: f32 = 0;

        var i: usize = 0;
        while (i < 64) : (i += 1) {
            const byte = block.qs[i];
            const w0 = @as(f32, @floatFromInt(byte & 0x3));
            const w1 = @as(f32, @floatFromInt((byte >> 2) & 0x3));
            const w2 = @as(f32, @floatFromInt((byte >> 4) & 0x3));
            const w3 = @as(f32, @floatFromInt((byte >> 6) & 0x3));

            const scale_idx = i / 8;
            const scale = scales[scale_idx];

            sum += (w0 * scale) * x[x_offset + i * 4 + 0];
            sum += (w1 * scale) * x[x_offset + i * 4 + 1];
            sum += (w2 * scale) * x[x_offset + i * 4 + 2];
            sum += (w3 * scale) * x[x_offset + i * 4 + 3];
        }
        total += sum;
    }
    return total;
}

// Q4_K scalar implementation
pub fn dotProductQ4K(q4k_data: []const u8, x: []const f32, n: usize) f32 {
    const num_blocks = n / 256;
    const blocks = @as([*]const BlockQ4K, @ptrCast(q4k_data))[0..num_blocks];
    var total: f32 = 0;

    for (blocks, 0..) |*block, b_idx| {
        var scales: [4]f32 = undefined;
        var mins: [4]f32 = undefined;
        for (0..4) |i| {
            scales[i] = @as(f32, @floatFromInt(block.scales[i])) / 64.0;
            mins[i] = @as(f32, @floatFromInt(block.scales[i + 6])) / 64.0;
        }

        const x_offset = b_idx * 256;
        var sum: f32 = 0;

        var i: usize = 0;
        while (i < 64) : (i += 1) {
            const byte = block.qs[i];
            const low = @as(f32, @floatFromInt(byte & 0xF));
            const high = @as(f32, @floatFromInt(byte >> 4));

            const scale_idx = i / 16;
            const deq_low = low * scales[scale_idx] + mins[scale_idx];
            const deq_high = high * scales[scale_idx] + mins[scale_idx];

            sum += deq_low * x[x_offset + i * 2];
            sum += deq_high * x[x_offset + i * 2 + 1];
        }
        total += sum;
    }
    return total;
}

// Q8_0 scalar implementation
pub fn dotProductQ8_0(q8_0_data: []const u8, x: []const f32, n: usize) f32 {
    const num_blocks = n / 32;
    const blocks = @as([*]const BlockQ8_0, @ptrCast(q8_0_data))[0..num_blocks];
    var total: f32 = 0;

    for (blocks, 0..) |*block, b_idx| {
        const delta = types.fp16ToFp32(block.delta);
        const x_offset = b_idx * 32;
        var sum: f32 = 0;

        for (0..32) |i| {
            sum += (@as(f32, @floatFromInt(block.qs[i])) * delta) * x[x_offset + i];
        }
        total += sum;
    }
    return total;
}

// Q8_K scalar implementation
pub fn dotProductQ8K(q8k_data: []const u8, x: []const f32, n: usize) f32 {
    const num_blocks = n / 256;
    const blocks = @as([*]const BlockQ8K, @ptrCast(q8k_data))[0..num_blocks];
    var total: f32 = 0;

    for (blocks, 0..) |*block, b_idx| {
        const delta = block.d;
        const x_offset = b_idx * 256;
        var sum: f32 = 0;

        for (0..256) |i| {
            sum += (@as(f32, @floatFromInt(block.qs[i])) * delta) * x[x_offset + i];
        }
        total += sum;
    }
    return total;
}

fn decodeQ2KScales(data: [16]u8) [8]f32 {
    var result: [8]f32 = undefined;
    for (0..8) |i| {
        result[i] = @as(f32, @floatFromInt(data[i] & 0x3F)) / 32.0;
    }
    return result;
}

fn dotProductF16(data: []const u8, x: []const f32, n: usize) f32 {
    const f16_data = @as([*]const types.F16, @ptrCast(@alignCast(data)))[0..n];
    var sum: f32 = 0;
    for (f16_data, 0..) |w, i| {
        sum += types.fp16ToFp32(w) * x[i];
    }
    return sum;
}

fn dotProductF32(data: []const u8, x: []const f32, n: usize) f32 {
    const f32_data = @as([*]const f32, @ptrCast(@alignCast(data)))[0..n];
    var sum: f32 = 0;
    for (f32_data, 0..) |w, i| {
        sum += w * x[i];
    }
    return sum;
}
