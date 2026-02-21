const std = @import("std");

pub const BenchmarkResult = struct {
    model: []const u8,
    quant: []const u8,
    cpu: []const u8,
    tok_per_sec: f32,
    memory_mb: f32,
};

pub const known_results = [_]BenchmarkResult{
    .{ .model = "TinyLlama 1.1B", .quant = "Q4_K", .cpu = "Ryzen 7 3700X", .tok_per_sec = 95, .memory_mb = 45 },
    .{ .model = "TinyLlama 1.1B", .quant = "Q2_K", .cpu = "Ryzen 7 3700X", .tok_per_sec = 160, .memory_mb = 22 },
    .{ .model = "Llama 2 7B", .quant = "Q4_K", .cpu = "Ryzen 7 3700X", .tok_per_sec = 14, .memory_mb = 290 },
    .{ .model = "SmolLM 135M", .quant = "Q4_K", .cpu = "Intel J2900", .tok_per_sec = 10, .memory_mb = 18 },
    .{ .model = "TinyLlama 1.1B", .quant = "Q2_K", .cpu = "Xeon E5-2678 v3", .tok_per_sec = 1100, .memory_mb = 22 },
};
