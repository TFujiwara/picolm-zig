const std = @import("std");

pub const Architecture = enum {
    llama,
    mistral,
    phi,
    gemma,
    qwen,
    smollm,
    unknown,

    pub fn fromGGUF(gguf_model: anytype) Architecture {
        const arch_str = gguf_model.getString("general.architecture") orelse return .unknown;

        if (std.mem.indexOf(u8, arch_str, "llama") != null) return .llama;
        if (std.mem.indexOf(u8, arch_str, "mistral") != null) return .mistral;
        if (std.mem.indexOf(u8, arch_str, "mixtral") != null) return .mistral;
        if (std.mem.indexOf(u8, arch_str, "phi") != null) return .phi;
        if (std.mem.indexOf(u8, arch_str, "gemma") != null) return .gemma;
        if (std.mem.indexOf(u8, arch_str, "qwen") != null) return .qwen;

        return .unknown;
    }
};

pub const ArchConfig = struct {
    vocab_size: usize,
    context_length: usize,
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    use_sliding_window: bool = false,
    sliding_window: usize = 0,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,
    use_swiglu: bool = true,
};
