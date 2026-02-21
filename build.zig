const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "picolm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.linkLibC();

    // Optimizaciones agresivas
    exe.root_module.strip = optimize == .ReleaseFast;
    exe.want_lto = optimize == .ReleaseFast;
    exe.root_module.addCMacro("USE_SIMD", "1");

    // Features por arquitectura
    const cpu_features = target.result.cpu.features;

    if (target.result.cpu.arch.isX86()) {
        if (std.Target.x86.featureSetHas(cpu_features, .avx512f)) {
            exe.root_module.addCMacro("USE_AVX512", "1");
        } else if (std.Target.x86.featureSetHas(cpu_features, .avx2)) {
            exe.root_module.addCMacro("USE_AVX2", "1");
        }

        if (std.Target.x86.featureSetHas(cpu_features, .fma)) {
            exe.root_module.addCMacro("USE_FMA", "1");
        }
    } else if (target.result.cpu.arch.isAARCH64()) {
        if (std.Target.aarch64.featureSetHas(cpu_features, .neon)) {
            exe.root_module.addCMacro("USE_NEON", "1");
        }
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run PicoLM");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
