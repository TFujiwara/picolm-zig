const std = @import("std");
const gguf = @import("gguf.zig");
const memory = @import("memory.zig");
const cpu_detect = @import("cpu_detect.zig");
const motherboard = @import("motherboard_chinese.zig");
const thermal = @import("thermal_management.zig");
const numa = @import("numa.zig");

const usage =
    \\PicoLM-Zig Ultra v0.2.0
    \\Usage: picolm <model.gguf> [options]
    \\
    \\Options:
    \\  -p <prompt>      Input prompt
    \\  -n <int>         Max tokens (default: 256)
    \\  -t <float>       Temperature (default: 0.8)
    \\  -k <float>       Top-p (default: 0.9)
    \\  -s <int>         Seed (default: 42)
    \\  -j <int>         Threads (default: auto)
    \\  -c <int>         Context length
    \\  --q2k, --q4k, --q80, --q8k  Quantization selection
    \\  --info           Show model info
    \\  --memory         Show memory breakdown
    \\  --benchmark      Run speed benchmark
    \\  --ccx-optimize   Zen 2/3 CCX pinning
    \\  --numa-interleave Dual-socket NUMA
    \\  --thermal-limit  Enable thermal throttling
    \\  --gpu auto/cuda/vulkan/none
    \\  -h, --help       Show help
    \\
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(1);
    }

    // Parse args
    var model_path: ?[]const u8 = null;
    var show_info = false;
    var show_memory = false;
    var benchmark = false;
    var selected_quant: @import("quant.zig").Quantizer = .q4_k;
    var num_threads: ?usize = null;
    var context_override: ?usize = null;
    var ccx_optimize = false;
    var numa_interleave = false;
    var thermal_limit = false;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            std.debug.print("{s}", .{usage});
            return;
        } else if (std.mem.eql(u8, arg, "--info")) {
            show_info = true;
        } else if (std.mem.eql(u8, arg, "--memory")) {
            show_memory = true;
        } else if (std.mem.eql(u8, arg, "--benchmark")) {
            benchmark = true;
        } else if (std.mem.eql(u8, arg, "--q2k")) {
            selected_quant = .q2_k;
        } else if (std.mem.eql(u8, arg, "--q4k")) {
            selected_quant = .q4_k;
        } else if (std.mem.eql(u8, arg, "--q80")) {
            selected_quant = .q8_0;
        } else if (std.mem.eql(u8, arg, "--q8k")) {
            selected_quant = .q8_k;
        } else if (std.mem.eql(u8, arg, "--ccx-optimize")) {
            ccx_optimize = true;
        } else if (std.mem.eql(u8, arg, "--numa-interleave")) {
            numa_interleave = true;
        } else if (std.mem.eql(u8, arg, "--thermal-limit")) {
            thermal_limit = true;
        } else if (std.mem.eql(u8, arg, "-j")) {
            i += 1;
            num_threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-c")) {
            i += 1;
            context_override = try std.fmt.parseInt(usize, args[i], 10);
        } else if (arg[0] != '-' and model_path == null) {
            model_path = arg;
        }
    }

    const mpath = model_path orelse {
        std.debug.print("Error: No model specified\n", .{});
        std.process.exit(1);
    };

    // Detect CPU
    const cpu = cpu_detect.CPUFeatures.detect();
    var stderr_buf: [4096]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buf).interface;
    try cpu.print(&stderr);

    // Detect motherboard if Xeon
    var mb: ?motherboard.ChineseMotherboard = null;
    if (cpu.is_xeon) {
        mb = motherboard.ChineseMotherboard.detect();
        try mb.?.printWarnings(&stderr);
    }

    // Setup NUMA if dual socket
    var numa_manager: ?numa.NUMAManager = null;
    if (numa_interleave and cpu.numa_nodes > 1) {
        numa_manager = try numa.NUMAManager.init(allocator);
        numa.NUMAManager.setInterleavePolicy();
    }

    // Setup thermal management
    var thermal_mgr: ?thermal.ThermalManager = null;
    if (thermal_limit) {
        if (mb) |m| {
            thermal_mgr = try thermal.ThermalManager.init(m);
        }
    }
    defer if (thermal_mgr) |*tm| tm.deinit();

    // Load model
    std.debug.print("Loading: {s}\n", .{mpath});

    var gguf_model = gguf.GGUFModel.init(allocator);
    defer gguf_model.deinit();

    try gguf_model.load(mpath);

    if (show_info) {
        try gguf_model.printInfo(&stderr);
        return;
    }

    // Calculate memory
    const config = types.Config{
        .dim = gguf_model.embedding_length orelse 2048,
        .hidden_dim = gguf_model.feed_forward_length orelse 5504,
        .n_layers = gguf_model.block_count orelse 22,
        .n_heads = gguf_model.attention_head_count orelse 32,
        .n_kv_heads = gguf_model.attention_head_count_kv orelse 4,
        .vocab_size = gguf_model.vocab_size orelse 32000,
        .seq_len = context_override orelse (gguf_model.context_length orelse 2048),
    };

    const quant_type = selected_quant;
    const mem_breakdown = memory.MemoryCalculator.calculate(config, quant_type, config.vocab_size, config.seq_len);

    if (show_memory) {
        try mem_breakdown.format("", .{}, &stderr);
        try stderr.writeAll("\n");
        return;
    }

    try mem_breakdown.format("", .{}, &stderr);
    try stderr.writeAll("\n");

    // Determine thread count
    const threads = num_threads orelse if (cpu.is_xeon)
        cpu.getOptimalThreadCountXeon()
    else if (ccx_optimize)
        8 // Zen 2 CCX
    else
        cpu.logical_cores;

    std.debug.print("Threads: {d}\n", .{threads});

    // Run
    if (benchmark) {
        // Benchmark code
    }

    // Main loop with thermal monitoring
    if (thermal_mgr) |*tm| {
        const action = try tm.update();
        if (action.throttle_level > 0) {
            std.log.warn("Thermal throttling: level {d}", .{action.throttle_level});
        }
    }
}

const types = @import("types.zig");
