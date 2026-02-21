const std = @import("std");

pub const MoERouter = struct {
    num_experts: usize,

    pub fn route(self: MoERouter, router_logits: []const f32) struct {
        experts: [2]usize,
        weights: [2]f32,
    } {
        var top1_idx: usize = 0;
        var top1_val: f32 = -std.math.inf(f32);
        var top2_idx: usize = 1;
        var top2_val: f32 = -std.math.inf(f32);

        for (router_logits, 0..) |logit, i| {
            if (logit > top1_val) {
                top2_val = top1_val;
                top2_idx = top1_idx;
                top1_val = logit;
                top1_idx = i;
            } else if (logit > top2_val) {
                top2_val = logit;
                top2_idx = i;
            }
        }

        const exp1 = @exp(top1_val);
        const exp2 = @exp(top2_val);
        const sum = exp1 + exp2;

        _ = self;

        return .{
            .experts = .{ top1_idx, top2_idx },
            .weights = .{ exp1 / sum, exp2 / sum },
        };
    }
};
