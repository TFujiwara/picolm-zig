const std = @import("std");
const gpu = @import("gpu_backend.zig");

pub const HybridCompute = struct {
    cpu_dispatcher: @import("architecture_dispatcher.zig").ArchDispatcher,
    gpu: gpu.GPUBackend,

    pub fn init(allocator: std.mem.Allocator) !HybridCompute {
        const gpu_backend = try gpu.GPUBackend.init(allocator, .auto);
        const cpu = @import("architecture_dispatcher.zig").ArchDispatcher.init();

        return .{
            .cpu_dispatcher = cpu,
            .gpu = gpu_backend,
        };
    }

    pub fn deinit(self: *HybridCompute) void {
        self.gpu.deinit();
    }

    pub fn scheduleMatmul(self: *HybridCompute, out: []f32, x: []const f32, w: []const u8, n: usize, d: usize) void {
        if (n < 512 or self.gpu == .none) {
            // CPU
            _ = out;
            _ = x;
            _ = w;
        } else if (self.gpu.shouldOffload(n, d)) {
            // GPU
        } else {
            // CPU
        }
    }
};
