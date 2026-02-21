const std = @import("std");

pub const GPUBackend = union(enum) {
    cuda: CUDABackend,
    vulkan: VulkanBackend,
    opencl: OpenCLBackend,
    none: void,

    pub fn init(allocator: std.mem.Allocator, preference: GPUPreference) !GPUBackend {
        switch (preference) {
            .cuda => if (CUDABackend.init(allocator)) |cuda| return .{ .cuda = cuda } else |_| {},
            .vulkan => if (VulkanBackend.init(allocator)) |vk| return .{ .vulkan = vk } else |_| {},
            .opencl => if (OpenCLBackend.init(allocator)) |cl| return .{ .opencl = cl } else |_| {},
            .auto => {
                if (CUDABackend.init(allocator)) |cuda| return .{ .cuda = cuda } else |_| {}
                if (VulkanBackend.init(allocator)) |vk| return .{ .vulkan = vk } else |_| {}
                if (OpenCLBackend.init(allocator)) |cl| return .{ .opencl = cl } else |_| {}
            },
        }
        return .{ .none = {} };
    }

    pub fn deinit(self: *GPUBackend) void {
        switch (self.*) {
            .cuda => |*cuda| cuda.deinit(),
            .vulkan => |*vk| vk.deinit(),
            .opencl => |*cl| cl.deinit(),
            .none => {},
        }
    }

    pub fn shouldOffload(self: GPUBackend, rows: usize, cols: usize) bool {
        const info = self.getInfo();
        const matrix_size = rows * cols * 4;
        return matrix_size > 10 * 1024 * 1024 and matrix_size < info.vram_bytes / 4;
    }

    pub fn getInfo(self: GPUBackend) GPUInfo {
        return switch (self) {
            .cuda => |cuda| cuda.getInfo(),
            .vulkan => |vk| vk.getInfo(),
            .opencl => |cl| cl.getInfo(),
            .none => .{ .name = "CPU Only", .vram_bytes = 0, .compute_units = 0, .max_work_group_size = 0, .supports_fp16 = false, .supports_int8 = false },
        };
    }

    pub const GPUInfo = struct {
        name: []const u8,
        vram_bytes: u64,
        compute_units: u32,
        max_work_group_size: usize,
        supports_fp16: bool,
        supports_int8: bool,
    };
};

pub const GPUPreference = enum { auto, cuda, vulkan, opencl };

pub const CUDABackend = struct {
    pub fn init(_: std.mem.Allocator) !CUDABackend {
        return error.NotImplemented;
    }
    pub fn deinit(_: *CUDABackend) void {}
    pub fn getInfo(_: CUDABackend) GPUBackend.GPUInfo {
        return .{ .name = "CUDA", .vram_bytes = 0, .compute_units = 0, .max_work_group_size = 1024, .supports_fp16 = true, .supports_int8 = true };
    }
};

pub const VulkanBackend = struct {
    pub fn init(_: std.mem.Allocator) !VulkanBackend {
        return error.NotImplemented;
    }
    pub fn deinit(_: *VulkanBackend) void {}
    pub fn getInfo(_: VulkanBackend) GPUBackend.GPUInfo {
        return .{ .name = "Vulkan", .vram_bytes = 0, .compute_units = 0, .max_work_group_size = 256, .supports_fp16 = true, .supports_int8 = true };
    }
};

pub const OpenCLBackend = struct {
    pub fn init(_: std.mem.Allocator) !OpenCLBackend {
        return error.NotImplemented;
    }
    pub fn deinit(_: *OpenCLBackend) void {}
    pub fn getInfo(_: OpenCLBackend) GPUBackend.GPUInfo {
        return .{ .name = "OpenCL", .vram_bytes = 0, .compute_units = 0, .max_work_group_size = 256, .supports_fp16 = false, .supports_int8 = false };
    }
};
