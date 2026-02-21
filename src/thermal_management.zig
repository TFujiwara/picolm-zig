const std = @import("std");

pub const ThermalManager = struct {
    cpu_temp: Sensor,
    vrm_temp: ?Sensor,
    target_cpu_temp: u32,
    target_vrm_temp: u32,
    throttle_active: bool,
    msr_fd: ?std.fs.File,

    pub const Sensor = struct {
        path: []const u8,
        current: u32,
        max: u32,
        critical: u32,

        pub fn read(self: *Sensor) !void {
            const file = try std.fs.cwd().openFile(self.path, .{});
            defer file.close();
            var buf: [32]u8 = undefined;
            const n = try file.read(&buf);
            const temp = try std.fmt.parseInt(u32, std.mem.trim(u8, buf[0..n], " \n"), 10);
            self.current = temp / 1000;
            if (self.current > self.max) self.max = self.current;
        }
    };

    pub fn init(motherboard: @import("motherboard_chinese.zig").ChineseMotherboard) !ThermalManager {
        const settings = motherboard.getRecommendedSettings();
        return .{
            .cpu_temp = .{ .path = "/sys/class/thermal/thermal_zone0/temp", .current = 0, .max = 0, .critical = 100 },
            .vrm_temp = null,
            .target_cpu_temp = 85,
            .target_vrm_temp = settings.target_vrm_temp,
            .throttle_active = false,
            .msr_fd = std.fs.cwd().openFile("/dev/cpu/0/msr", .{ .mode = .read_write }) catch null,
        };
    }

    pub fn deinit(self: *ThermalManager) void {
        if (self.msr_fd) |*fd| fd.close();
    }

    pub fn update(self: *ThermalManager) !ThermalAction {
        try self.cpu_temp.read();

        if (self.cpu_temp.current > self.cpu_temp.critical - 5) {
            self.throttle_active = true;
            return .{ .throttle_level = 3, .reason = .critical_temp };
        }
        if (self.cpu_temp.current > self.target_cpu_temp + 10) {
            self.throttle_active = true;
            return .{ .throttle_level = 2, .reason = .high_temp };
        }
        if (self.cpu_temp.current > self.target_cpu_temp) {
            self.throttle_active = true;
            return .{ .throttle_level = 1, .reason = .warm };
        }

        if (self.throttle_active and self.cpu_temp.current < self.target_cpu_temp - 10) {
            self.throttle_active = false;
            return .{ .throttle_level = 0, .reason = .recovered };
        }

        return .{ .throttle_level = if (self.throttle_active) 1 else 0, .reason = .maintain };
    }

    pub const ThermalAction = struct {
        throttle_level: u32,
        reason: enum { critical_temp, high_temp, warm, recovered, maintain },
    };
};
