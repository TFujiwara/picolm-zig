const std = @import("std");
const types = @import("types.zig");

pub const MemoryBreakdown = struct {
    input_buffer: usize,
    attention_buffers: usize,
    ffn_buffers: usize,
    logits: usize,
    kv_cache_keys: usize,
    kv_cache_values: usize,
    dequantized_weights: usize,
    tokenizer: usize,
    overhead: usize,

    pub fn totalRuntime(self: MemoryBreakdown) usize {
        return self.input_buffer + self.attention_buffers + self.ffn_buffers +
            self.logits + self.kv_cache_keys + self.kv_cache_values +
            self.dequantized_weights + self.tokenizer + self.overhead;
    }

    pub fn format(self: MemoryBreakdown, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("=== Memory Breakdown ===\n", .{});
        try writer.print("Input: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.input_buffer)) / 1e6});
        try writer.print("Attention: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.attention_buffers)) / 1e6});
        try writer.print("FFN: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.ffn_buffers)) / 1e6});
        try writer.print("Logits: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.logits)) / 1e6});
        try writer.print("KV Cache: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.kv_cache_keys + self.kv_cache_values)) / 1e6});
        try writer.print("TOTAL: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.totalRuntime())) / 1e6});
    }
};

pub const MemoryCalculator = struct {
    pub fn calculate(config: types.Config, quant_type: @import("quant.zig").Quantizer, vocab_size: usize, max_seq_len: usize) MemoryBreakdown {
        _ = quant_type;
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const n_layers = config.n_layers;
        const kv_dim = config.kv_dim();

        return .{
            .input_buffer = dim * 4,
            .attention_buffers = dim * 3 * 4 + kv_dim * 2 * 4 + config.n_heads * max_seq_len * 4,
            .ffn_buffers = hidden_dim * 3 * 4,
            .logits = vocab_size * 4,
            .kv_cache_keys = n_layers * max_seq_len * kv_dim * 2,
            .kv_cache_values = n_layers * max_seq_len * kv_dim * 2,
            .dequantized_weights = 0,
            .tokenizer = vocab_size * 128,
            .overhead = 10 * 1024 * 1024,
        };
    }
};
