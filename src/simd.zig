const std = @import("std");

pub const Vec4f = @Vector(4, f32);
pub const Vec8f = @Vector(8, f32);

pub inline fn rmsNormVec8(o: []f32, x: []const f32, weight: []const f32) void {
    const n = x.len;
    const vec_n = n - (n % 8);

    var sum_vec = @as(Vec8f, @splat(0));
    var i: usize = 0;
    while (i < vec_n) : (i += 8) {
        const xv: Vec8f = x[i..][0..8].*;
        sum_vec += xv * xv;
    }

    var ss = @reduce(.Add, sum_vec);
    while (i < n) : (i += 1) ss += x[i] * x[i];

    ss /= @as(f32, @floatFromInt(n));
    ss += 1e-5;
    const norm = 1.0 / std.math.sqrt(ss);
    const norm_vec = @as(Vec8f, @splat(norm));

    i = 0;
    while (i < vec_n) : (i += 8) {
        const xv: Vec8f = x[i..][0..8].*;
        const wv: Vec8f = weight[i..][0..8].*;
        o[i..][0..8].* = xv * norm_vec * wv;
    }
    while (i < n) : (i += 1) o[i] = x[i] * norm * weight[i];
}

pub inline fn softmaxOnline(x: []f32) void {
    const n = x.len;
    var max_val = x[0];
    for (x[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0;
    for (x[0..n]) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    const inv_sum = 1.0 / sum;
    const inv_vec = @as(Vec8f, @splat(inv_sum));

    const vec_n = n - (n % 8);
    var i: usize = 0;
    while (i < vec_n) : (i += 8) {
        const xv: Vec8f = x[i..][0..8].*;
        x[i..][0..8].* = xv * inv_vec;
    }
    while (i < n) : (i += 1) x[i] *= inv_sum;
}

pub inline fn siluVec8(x: []f32) void {
    const n = x.len;
    const vec_n = n - (n % 8);
    var i: usize = 0;

    while (i < vec_n) : (i += 8) {
        const xv: Vec8f = x[i..][0..8].*;
        const neg_x = -xv;
        const exp_neg_x = @exp(neg_x);
        const sigmoid = @as(Vec8f, @splat(1.0)) / (@as(Vec8f, @splat(1.0)) + exp_neg_x);
        x[i..][0..8].* = xv * sigmoid;
    }
    while (i < n) : (i += 1) x[i] = x[i] / (1.0 + @exp(-x[i]));
}

pub inline fn vecAdd(out: []f32, a: []const f32, b: []const f32) void {
    const n = out.len;
    const vec_n = n - (n % 8);
    var i: usize = 0;
    while (i < vec_n) : (i += 8) {
        const av: Vec8f = a[i..][0..8].*;
        const bv: Vec8f = b[i..][0..8].*;
        out[i..][0..8].* = av + bv;
    }
    while (i < n) : (i += 1) out[i] = a[i] + b[i];
}

pub inline fn vecMul(out: []f32, a: []const f32, b: []const f32) void {
    const n = out.len;
    const vec_n = n - (n % 8);
    var i: usize = 0;
    while (i < vec_n) : (i += 8) {
        const av: Vec8f = a[i..][0..8].*;
        const bv: Vec8f = b[i..][0..8].*;
        out[i..][0..8].* = av * bv;
    }
    while (i < n) : (i += 1) out[i] = a[i] * b[i];
}
