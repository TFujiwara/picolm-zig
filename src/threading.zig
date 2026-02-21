const std = @import("std");

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    queue: WorkQueue,
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    shutdown: std.atomic.Value(bool),

    const Task = struct {
        func: *const fn (*anyopaque) void,
        ctx: *anyopaque,
    };

    const WorkQueue = struct {
        tasks: []Task,
        head: std.atomic.Value(usize),
        tail: std.atomic.Value(usize),
        capacity: usize,
    };

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !ThreadPool {
        var pool = ThreadPool{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, num_threads),
            .queue = .{
                .tasks = try allocator.alloc(Task, 1024),
                .head = std.atomic.Value(usize).init(0),
                .tail = std.atomic.Value(usize).init(0),
                .capacity = 1024,
            },
            .mutex = .{},
            .cond = .{},
            .shutdown = std.atomic.Value(bool).init(false),
        };

        for (0..num_threads) |i| {
            pool.threads[i] = try std.Thread.spawn(.{}, worker, .{ &pool, i });
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        self.shutdown.store(true, .release);
        self.cond.broadcast();
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        self.allocator.free(self.queue.tasks);
    }

    pub fn submit(self: *ThreadPool, comptime func: anytype, ctx: anytype) !void {
        const Ctx = @TypeOf(ctx);
        const Wrapper = struct {
            fn call(ptr: *anyopaque) void {
                const c = @as(*Ctx, @ptrCast(@alignCast(ptr)));
                func(c.*);
            }
        };

        const tail = self.queue.tail.load(.acquire);
        const next_tail = (tail + 1) % self.queue.capacity;
        if (next_tail == self.queue.head.load(.acquire)) return error.QueueFull;

        self.queue.tasks[tail] = .{ .func = Wrapper.call, .ctx = @ptrCast(ctx) };
        self.queue.tail.store(next_tail, .release);
        self.cond.signal();
    }

    fn worker(pool: *ThreadPool, id: usize) void {
        _ = id;
        while (!pool.shutdown.load(.acquire)) {
            pool.mutex.lock();
            while (pool.queue.head.load(.acquire) == pool.queue.tail.load(.acquire) and !pool.shutdown.load(.acquire)) {
                pool.cond.wait(&pool.mutex);
            }
            if (pool.shutdown.load(.acquire)) {
                pool.mutex.unlock();
                return;
            }

            const head = pool.queue.head.load(.acquire);
            const task = pool.queue.tasks[head];
            pool.queue.head.store((head + 1) % pool.queue.capacity, .release);
            pool.mutex.unlock();

            task.func(task.ctx);
        }
    }
};

pub const MatmulContext = struct {
    w: []const u8,
    x: []const f32,
    out: []f32,
    start_row: usize,
    end_row: usize,
    dim: usize,
    quant_type: @import("quant.zig").Quantizer,

    pub fn compute(self: *MatmulContext) void {
        var i = self.start_row;
        while (i < self.end_row) : (i += 1) {
            const row_size = self.quant_type.rowSize(self.dim);
            const row_offset = i * row_size;
            self.out[i] = self.quant_type.dotProduct(self.w[row_offset..], self.x, self.dim);
        }
    }
};
