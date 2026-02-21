const std = @import("std");
const cpu_detect = @import("cpu_detect.zig");

pub const ArchDispatcher = struct {
    cpu: cpu_detect.CPUFeatures,

    pub fn init() ArchDispatcher {
        return .{ .cpu = cpu_detect.CPUFeatures.detect() };
    }

    pub fn getDotProductQ4K(self: ArchDispatcher) *const fn ([]const u8, []const f32, usize) f32 {
        if (self.cpu.has_avx512f) return @import("simd_x86.zig").dotProductQ4KAVX512;
        if (self.cpu.has_avx2) return @import("simd_x86.zig").dotProductQ4KAVX2;
        if (self.cpu.has_sse4_1) return @import("simd_bay_trail.zig").dotProductQ4KSSE4;
        return @import("quant.zig").dotProductQ4K;
    }

    pub fn getRMSNorm(self: ArchDispatcher) *const fn ([]f32, []const f32, []const f32) void {
        if (self.cpu.has_avx2) return @import("simd_x86.zig").rmsNormAVX2;
        if (self.cpu.has_sse4_1) return @import("simd_bay_trail.zig").rmsNormSSE4;
        return @import("simd.zig").rmsNormVec8;
    }

    pub fn getOptimalThreads(self: ArchDispatcher) usize {
        return if (self.cpu.is_xeon)
            self.cpu.getOptimalThreadCountXeon()
        else
            self.cpu.logical_cores;
    }
};
