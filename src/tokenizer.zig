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

        // Simple character-based tokenization for now
        // In a full implementation, this would perform proper BPE
        var i: usize = 0;
        while (i < text.len) {
            // For now, we'll just convert characters to tokens directly
            // This is a simplified approach - a real implementation would use BPE
            const c = text[i];
            
            // Look for the character in the vocabulary
            var found = false;
            var j: usize = 0;
            while (j < self.vocab.len) : (j += 1) {
                if (self.vocab[j].len == 1 and self.vocab[j][0] == c) {
                    try tokens.append(@intCast(j));
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // If character not found in vocab, use byte value as fallback
                // This matches the original fallback behavior
                try tokens.append(@intCast(256 + @as(u16, c)));
            }
            
            i += 1;
        }

        return tokens.toOwnedSlice();
    }
    
    pub fn decode(self: *Tokenizer, token: types.Token) ![]const u8 {
        if (token < @as(types.Token, @intCast(self.vocab.len))) {
            return self.allocator.dupe(u8, self.vocab[token]);
        } else if (token >= 256 and token < 512) {
            // Handle fallback bytes
            const byte_val = @as(u8, @truncate(token - 256));
            var byte_str: [1]u8 = [_]u8{byte_val};
            return self.allocator.dupe(u8, &byte_str);
        } else {
            // Return empty string for unknown tokens
            return self.allocator.dupe(u8, "");
        }
    }
};
