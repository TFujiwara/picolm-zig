const std = @import("std");

pub const Zen2Scheduler = struct {
    ccx_threads: [2][4]?std.Thread,

    pub fn init() Zen2Scheduler {
        return .{ .ccx_threads = .{ .{null} ** 4, .{null} ** 4 } };
    }

    pub fn scheduleByCCX(
        self: *Zen2Scheduler,
        comptime func: anytype,
        args: anytype,
        num_tasks: usize,
    ) !void {
        const Wrapper = struct {
            fn worker(
                scheduler: *Zen2Scheduler,
                ccx_id: usize,
                core_start: usize,
                core_end: usize,
                runtime_args: @TypeOf(args),
                task_start: usize,
                task_end: usize,
            ) void {
                _ = scheduler;
                _ = ccx_id;
                _ = core_end;
                pinToCore(core_start);
                var i = task_start;
                while (i < task_end) : (i += 1) {
                    func(runtime_args, i);
                }
            }
        };

        const tasks_per_ccx = (num_tasks + 1) / 2;

        const ccx0_handle = try std.Thread.spawn(.{}, Wrapper.worker, .{
            self, @as(usize, 0), @as(usize, 0), @as(usize, 4), args, @as(usize, 0), tasks_per_ccx,
        });

        const ccx1_handle = try std.Thread.spawn(.{}, Wrapper.worker, .{
            self, @as(usize, 1), @as(usize, 4), @as(usize, 8), args, tasks_per_ccx, num_tasks,
        });

        ccx0_handle.join();
        ccx1_handle.join();
    }
};

fn pinToCore(core: usize) void {
    _ = core;
    // Implementación con pthread_setaffinity_np
}
