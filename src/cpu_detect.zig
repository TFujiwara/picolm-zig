const std = @import("std");

pub const XeonVariant = enum {
    unknown,
    e5_v3_2620,
    e5_v3_2630,
    e5_v3_2650,
    e5_v3_2670,
    e5_v3_2680,
    e5_v3_2690,
    e5_v3_2697,
    e5_v3_2699,
    e5_v4_2620,
    e5_v4_2630,
    e5_v4_2650,
    e5_v4_2670,
    e5_v4_2680,
    e5_v4_2690,
    e5_v4_2697,
    e5_v4_2699,
    bronze,
    silver,
    gold,
    platinum,
};

pub const CPUFeatures = struct {
    has_sse2: bool = false,
    has_sse4_1: bool = false,
    has_sse4_2: bool = false,
    has_avx: bool = false,
    has_avx2: bool = false,
    has_fma: bool = false,
    has_avx512f: bool = false,
    has_avx512bw: bool = false,
    has_avx512vnni: bool = false,
    has_bmi1: bool = false,
    has_bmi2: bool = false,
    has_popcnt: bool = false,

    is_intel: bool = false,
    is_amd: bool = false,
    is_xeon: bool = false,
    xeon_variant: XeonVariant = .unknown,

    physical_cores: usize = 0,
    logical_cores: usize = 0,
    socket_count: u32 = 1,
    numa_nodes: u32 = 1,

    l1d_cache: usize = 0,
    l2_cache: usize = 0,
    l3_cache: usize = 0,

    avx2_turbo: bool = true,

    pub fn detect() CPUFeatures {
        var features: CPUFeatures = .{};

        var eax: u32 = 0;
        var ebx: u32 = 0;
        var ecx: u32 = 0;
        var edx: u32 = 0;

        // Vendor
        eax = 0;
        asm volatile ("cpuid"
            : [eax] "={eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "={ecx}" (ecx),
              [edx] "={edx}" (edx),
            : [in_eax] "{eax}" (eax),
            : .{ .memory = true }
        );

        var vendor: [12]u8 = undefined;
        @memcpy(vendor[0..4], std.mem.asBytes(&ebx));
        @memcpy(vendor[4..8], std.mem.asBytes(&edx));
        @memcpy(vendor[8..12], std.mem.asBytes(&ecx));

        if (std.mem.eql(u8, &vendor, "GenuineIntel")) features.is_intel = true;
        if (std.mem.eql(u8, &vendor, "AuthenticAMD")) features.is_amd = true;

        // Features
        eax = 1;
        asm volatile ("cpuid"
            : [eax] "={eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "={ecx}" (ecx),
              [edx] "={edx}" (edx),
            : [in_eax] "{eax}" (eax),
            : .{ .memory = true }
        );

        features.has_sse2 = (edx & (1 << 26)) != 0;
        features.has_sse4_1 = (ecx & (1 << 19)) != 0;
        features.has_sse4_2 = (ecx & (1 << 20)) != 0;
        features.has_avx = (ecx & (1 << 28)) != 0;

        // Extended features
        eax = 7;
        ecx = 0;
        asm volatile ("cpuid"
            : [eax] "={eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "={ecx}" (ecx),
              [edx] "={edx}" (edx),
            : [in_eax] "{eax}" (eax),
              [in_ecx] "{ecx}" (ecx),
            : .{ .memory = true }
        );

        features.has_avx2 = (ebx & (1 << 5)) != 0;
        features.has_bmi1 = (ebx & (1 << 3)) != 0;
        features.has_bmi2 = (ebx & (1 << 8)) != 0;
        features.has_avx512f = (ebx & (1 << 16)) != 0;
        features.has_avx512bw = (ebx & (1 << 30)) != 0;
        features.has_avx512vnni = (ecx & (1 << 11)) != 0;

        features.logical_cores = std.Thread.getCpuCount() catch 1;

        if (features.is_intel) detectXeon(&features);

        return features;
    }

    fn detectXeon(features: *CPUFeatures) void {
        // Detectar Xeon por brand string
        // Simplificado
        features.is_xeon = true;
    }

    pub fn getOptimalSimdWidth(self: CPUFeatures) usize {
        if (self.has_avx512f) return 16;
        if (self.has_avx2) return 8;
        if (self.has_sse4_1) return 4;
        return 1;
    }

    pub fn getOptimalThreadCountXeon(self: CPUFeatures) usize {
        if (!self.is_xeon) return self.logical_cores;
        const physical = self.logical_cores / 2;
        if (self.socket_count > 1) return physical - self.socket_count;
        return physical;
    }

    pub fn print(self: CPUFeatures, writer: anytype) !void {
        try writer.print("=== CPU Features ===\n", .{});
        try writer.print("Vendor: {s}\n", .{if (self.is_intel) "Intel" else if (self.is_amd) "AMD" else "Unknown"});
        try writer.print("Cores: {d} logical\n", .{self.logical_cores});
        try writer.print("SSE4.1: {}, AVX2: {}, AVX-512: {}\n", .{ self.has_sse4_1, self.has_avx2, self.has_avx512f });
    }
};
