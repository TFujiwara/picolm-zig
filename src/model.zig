const std = @import("std");
const types = @import("types.zig");
const gguf = @import("gguf.zig");

pub const Transformer = struct {
    allocator: std.mem.Allocator,
    config: types.Config,

    weights: struct {
        token_embedding: []const u8,
        rms_att: []const f32,
        rms_ffn: []const f32,
        rms_final: []const f32,
        wq: []const []const u8,
        wk: []const []const u8,
        wv: []const []const u8,
        wo: []const []const u8,
        w1: []const []const u8,
        w2: []const []const u8,
        w3: []const []const u8,
        wcls: []const u8,
        quant: @import("quant.zig").Quantizer,
    },

    state: struct {
        x: []f32,
        xb: []f32,
        xb2: []f32,
        hb: []f32,
        hb2: []f32,
        q: []f32,
        k: []f32,
        v: []f32,
        logits: []f32,
        key_cache: []types.F16,
        value_cache: []types.F16,
        rope_cos: []f32,
        rope_sin: []f32,
    },

    pub fn init(allocator: std.mem.Allocator, gguf_model: *gguf.GGUFModel, quant_type: @import("quant.zig").Quantizer) !Transformer {
        const config = types.Config{
            .dim = @as(usize, @intCast(gguf_model.embedding_length orelse 2048)),
            .hidden_dim = @as(usize, @intCast(gguf_model.feed_forward_length orelse 5504)),
            .n_layers = @as(usize, @intCast(gguf_model.block_count orelse 22)),
            .n_heads = @as(usize, @intCast(gguf_model.attention_head_count orelse 32)),
            .n_kv_heads = @as(usize, @intCast(gguf_model.attention_head_count_kv orelse 4)),
            .vocab_size = @as(usize, @intCast(gguf_model.vocab_size orelse 32000)),
            .seq_len = @as(usize, @intCast(gguf_model.context_length orelse 2048)),
        };

        // Allocate state
        const dim = config.dim;
        const kv_dim = config.kv_dim();
        const hidden_dim = config.hidden_dim;

        const state = .{
            .x = try allocator.alloc(f32, dim),
            .xb = try allocator.alloc(f32, dim),
            .xb2 = try allocator.alloc(f32, dim),
            .hb = try allocator.alloc(f32, hidden_dim),
            .hb2 = try allocator.alloc(f32, hidden_dim),
            .q = try allocator.alloc(f32, dim),
            .k = try allocator.alloc(f32, kv_dim),
            .v = try allocator.alloc(f32, kv_dim),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .key_cache = try allocator.alloc(types.F16, config.n_layers * config.seq_len * kv_dim),
            .value_cache = try allocator.alloc(types.F16, config.n_layers * config.seq_len * kv_dim),
            .rope_cos = try allocator.alloc(f32, config.seq_len * (config.head_dim() / 2)),
            .rope_sin = try allocator.alloc(f32, config.seq_len * (config.head_dim() / 2)),
        };

        @memset(state.key_cache, 0);
        @memset(state.value_cache, 0);

        // Precompute RoPE
        Transformer.precomputeRoPE(state.rope_cos, state.rope_sin, config);

        // Load weights...
        _ = quant_type;

        return .{
            .allocator = allocator,
            .config = config,
            .weights = undefined, // Load from GGUF
            .state = state,
        };
    }

    pub fn deinit(self: *Transformer) void {
        // Free all allocations
        _ = self;
    }

    fn precomputeRoPE(cos: []f32, sin: []f32, config: types.Config) void {
        const head_dim = config.head_dim();
        const half_dim = head_dim / 2;

        var pos: usize = 0;
        while (pos < config.seq_len) : (pos += 1) {
            var i: usize = 0;
            while (i < half_dim) : (i += 1) {
                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                const val = @as(f32, @floatFromInt(pos)) * freq;
                cos[pos * half_dim + i] = @cos(val);
                sin[pos * half_dim + i] = @sin(val);
            }
        }
    }

    pub fn forward(self: *Transformer, token: types.Token, pos: usize) []f32 {
        // Embedding
        self.embed(token);

        // Layers...
        _ = pos;

        return self.state.logits;
    }

    fn embed(self: *Transformer, token: types.Token) void {
        const row_size = self.config.dim;
        const offset = @as(usize, token) * row_size;

        const emb = @as([*]const types.F16, @ptrCast(@alignCast(self.weights.token_embedding.ptr)))[offset..][0..row_size];

        for (0..row_size) |i| {
            self.state.x[i] = types.fp16ToFp32(emb[i]);
        }
    }
};
