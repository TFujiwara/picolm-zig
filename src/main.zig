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

    // Parse remaining arguments for generation
    var prompt: ?[]const u8 = null;
    var max_tokens: usize = 256;
    var temperature: f32 = 0.8;
    var topp: f32 = 0.9;
    var seed: u64 = 42;
    
    i = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "-p") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, arg, "-n") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-t") and i + 1 < args.len) {
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "-k") and i + 1 < args.len) {
            i += 1;
            topp = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "-s") and i + 1 < args.len) {
            i += 1;
            seed = @as(u64, @intCast(try std.fmt.parseInt(i64, args[i], 10)));
        }
    }
    
    const prompt_text = prompt orelse {
        std.debug.print("Error: No prompt provided (-p argument required)\n", .{});
        std.process.exit(1);
    };
    
    // Initialize tokenizer
    var tokenizer = @import("tokenizer.zig").Tokenizer.init(allocator);
    defer tokenizer.deinit();
    
    // Initialize model
    var transformer = @import("model.zig").Transformer.init(allocator, &gguf_model, selected_quant) catch {
        std.debug.print("Error: Failed to initialize model\n", .{});
        std.process.exit(1);
    };
    defer transformer.deinit();
    
    // Initialize sampler
    var sampler = @import("sampler.zig").Sampler.init(
        transformer.config.vocab_size,
        temperature,
        topp,
        seed
    );
    
    // Encode prompt
    const tokens = try tokenizer.encode(prompt_text);
    defer allocator.free(tokens);
    
    std.debug.print("Prompt: {d} tokens, generating up to {d} tokens (temp={d:.2}, top_p={d:.2})\n", 
        .{tokens.len, max_tokens, temperature, topp});
    std.debug.print("---\n", .{});
    
    // Process prompt tokens (prefill)
    var pos: usize = 0;
    var token: types.Token = 0;
    if (tokens.len > 0) {
        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            token = tokens[i];
            _ = transformer.forward(token, pos);
            pos += 1;
        }
    }
    
    // Generate new tokens
    var generated_tokens: usize = 0;
    while (generated_tokens < max_tokens) {
        // Forward pass
        const logits = transformer.forward(token, pos);
        
        // Sample next token
        token = sampler.sample(logits);
        
        // Decode and print token
        const token_str = tokenizer.decode(token) catch {
            std.debug.print("<UNK>");
            continue;
        };
        defer allocator.free(token_str);
        std.debug.print("{s}", .{token_str});
        std.debug.flush();
        
        // Check for end-of-sequence token (typically 0 or 2 for many models)
        if (token == 0 or token == 2) break;
        
        pos += 1;
        generated_tokens += 1;
    }
    
    std.debug.print("\n", .{});
    
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
