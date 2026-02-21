const std = @import("std");

pub const NUMAManager = struct {
    numa_nodes: u32,

    pub fn init(_: std.mem.Allocator) !NUMAManager {
        return .{ .numa_nodes = detectNUMANodes() };
    }

    fn detectNUMANodes() u32 {
        var count: u32 = 1;
        var dir = std.fs.cwd().openDir("/sys/devices/system/node/", .{ .iterate = true }) catch return 1;
        defer dir.close();
        var iter = dir.iterate();
        while (iter.next() catch return 1) |entry| {
            if (std.mem.startsWith(u8, entry.name, "node")) count += 1;
        }
        return @max(1, count - 1);
    }

    pub fn setInterleavePolicy() void {
        if (@import("builtin").os.tag != .linux) return;
        const SYS_set_mempolicy = 238;
        var mask: [16]c_ulong = undefined;
        @memset(&mask, 0xFF);
        _ = std.os.linux.syscall3(@enumFromInt(SYS_set_mempolicy), 3, @intFromPtr(&mask), 1024);
    }
};
