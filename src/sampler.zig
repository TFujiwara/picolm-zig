const std = @import("std");
const types = @import("types.zig");

pub const Sampler = struct {
    vocab_size: usize,
    temperature: f32,
    topp: f32,
    rng_state: u64,

    pub fn init(vocab_size: usize, temperature: f32, topp: f32, seed: u64) Sampler {
        return .{
            .vocab_size = vocab_size,
            .temperature = temperature,
            .topp = topp,
            .rng_state = seed,
        };
    }

    pub fn sample(self: *Sampler, logits: []f32) types.Token {
        // Apply temperature
        if (self.temperature != 1.0) {
            for (logits) |*l| l.* /= self.temperature;
        }

        // Softmax
        var max_logit = logits[0];
        for (logits[1..]) |l| {
            if (l > max_logit) max_logit = l;
        }

        var sum: f32 = 0;
        for (logits) |*l| {
            l.* = std.math.exp(l.* - max_logit);
            sum += l.*;
        }

        for (logits) |*l| l.* /= sum;

        // Top-p sampling
        if (self.topp < 1.0) {
            return self.sampleTopp(logits);
        }
        return self.sampleMultinomial(logits);
    }

    fn sampleTopp(self: *Sampler, probs: []f32) types.Token {
        // Sort and truncate
        _ = self;
        _ = probs;
        return 0;
    }

    fn sampleMultinomial(self: *Sampler, probs: []f32) types.Token {
        const r = self.randomFloat();
        var cumsum: f32 = 0;
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (r < cumsum) return @intCast(i);
        }
        return @intCast(probs.len - 1);
    }

    fn randomFloat(self: *Sampler) f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        return @as(f32, @floatFromInt(self.rng_state & 0xFFFFFFFF)) / 4294967296.0;
    }
};
