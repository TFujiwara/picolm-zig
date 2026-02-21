const std = @import("std");
const types = @import("types.zig");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: [][]const u8,
    scores: []f32,
    max_token_length: usize,

    pub fn init(allocator: std.mem.Allocator) Tokenizer {
        return .{
            .allocator = allocator,
            .vocab = &.{},
            .scores = &.{},
            .max_token_length = 0,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab) |token| self.allocator.free(token);
        self.allocator.free(self.vocab);
        self.allocator.free(self.scores);
    }

    pub fn encode(self: *Tokenizer, text: []const u8) ![]types.Token {
        var tokens = std.ArrayList(types.Token).init(self.allocator);
        defer tokens.deinit();

        // BPE encoding
        var i: usize = 0;
        while (i < text.len) {
            // Find longest matching token
            const best_len: usize = 0;
            const best_id: types.Token = 0;

            var len: usize = 1;
            while (len <= self.max_token_length and i + len <= text.len) : (len += 1) {
                // Binary search in vocab
            }

            if (best_len == 0) {
                try tokens.append(@intCast(256 + @as(u16, text[i])));
                i += 1;
            } else {
                try tokens.append(best_id);
                i += best_len;
            }
        }

        return tokens.toOwnedSlice();
    }
};
