const std = @import("std");

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,

    pub fn head_dim(self: Config) usize {
        return self.dim / self.n_heads;
    }

    pub fn kv_dim(self: Config) usize {
        return (self.dim * self.n_kv_heads) / self.n_heads;
    }
};

pub const Token = u32;
pub const Score = f32;

pub const QuantType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
};

pub const F16 = u16;

pub fn fp32ToFp16(x: f32) F16 {
    const bits = @as(u32, @bitCast(x));
    const sign = (bits >> 31) & 0x1;
    const exp = @as(i32, @intCast((bits >> 23) & 0xFF)) - 127 + 15;
    var mant = bits & 0x7FFFFF;

    if (exp >= 31) {
        return @as(u16, @intCast(sign << 15)) | 0x7C00;
    } else if (exp <= 0) {
        if (exp < -10) return @as(u16, @intCast(sign << 15));
        mant = (mant | 0x800000) >> @as(u5, @intCast(1 - exp));
        return @as(u16, @intCast(sign << 15)) | @as(u16, @intCast(mant >> 13));
    }

    return @as(u16, @intCast(sign << 15)) |
        @as(u16, @intCast(@as(u32, @intCast(exp)) << 10)) |
        @as(u16, @intCast(mant >> 13));
}

pub fn fp16ToFp32(h: F16) f32 {
    const sign = (h >> 15) & 0x1;
    const exp = (h >> 10) & 0x1F;
    const mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return @as(f32, @bitCast(@as(u32, sign) << 31));
        const val = @as(f32, @floatFromInt(mant)) / 1024.0 * 0.00006103515625;
        return if (sign == 1) -val else val;
    } else if (exp == 31) {
        return @as(f32, @bitCast((@as(u32, sign) << 31) | 0x7F800000 | (@as(u32, mant) << 13)));
    }

    const e = @as(i32, @intCast(exp)) - 15 + 127;
    return @as(f32, @bitCast((@as(u32, sign) << 31) |
        (@as(u32, @intCast(e)) << 23) |
        (@as(u32, mant) << 13)));
}
