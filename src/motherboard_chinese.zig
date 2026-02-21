const std = @import("std");

pub const ChineseMotherboard = struct {
    pub const Model = enum {
        machinist_x79,
        jingsha_x79,
        huananzhi_x79,
        kllisre_x79,
        x79z_v602,
        machinist_x99,
        jingsha_x99,
        huananzhi_x99,
        x99z_v102,
        x99_v202,
        x79_dual,
        x99_dual,
        unknown,
    };

    pub const VRMConfig = struct {
        phases: u32,
        doublers: bool,
        mosfet_type: enum { low_rds, standard, cheap, unknown },
        heatsink_quality: enum { excellent, good, adequate, poor },
        max_tdp_per_socket: u32,
    };

    pub const PowerConfig = struct {
        eps_connectors: u32,
        atx_24pin: bool,
        pcie_6pin_aux: bool,
        vrm_fan_header: bool,
        cpu_fan_headers: u32,
    };

    pub const ThermalCharacteristics = struct {
        vrm_sensor: bool,
        vrm_throttle_temp: u32,
        cpu_offset: i32,
        ambient_sensor: bool,
    };

    model: Model,
    vrm: VRMConfig,
    power: PowerConfig,
    thermal: ThermalCharacteristics,

    pub fn detect() ChineseMotherboard {
        // Detección por DMI/ACPI
        return .{
            .model = .unknown,
            .vrm = .{ .phases = 4, .doublers = true, .mosfet_type = .unknown, .heatsink_quality = .adequate, .max_tdp_per_socket = 120 },
            .power = .{ .eps_connectors = 1, .atx_24pin = true, .pcie_6pin_aux = false, .vrm_fan_header = false, .cpu_fan_headers = 1 },
            .thermal = .{ .vrm_sensor = false, .vrm_throttle_temp = 105, .cpu_offset = 0, .ambient_sensor = false },
        };
    }

    pub fn getRecommendedSettings(self: ChineseMotherboard) RecommendedSettings {
        return .{
            .max_turbo_bins = if (self.vrm.mosfet_type == .cheap) 0 else 2,
            .avx2_offset = if (self.vrm.heatsink_quality == .poor) 4 else 2,
            .power_limit_watts = @min(self.vrm.max_tdp_per_socket, 120),
            .disable_ht_if_hot = self.vrm.heatsink_quality == .poor,
            .target_vrm_temp = @min(self.thermal.vrm_throttle_temp - 15, 95),
        };
    }

    pub const RecommendedSettings = struct {
        max_turbo_bins: i32,
        avx2_offset: i32,
        power_limit_watts: u32,
        disable_ht_if_hot: bool,
        target_vrm_temp: u32,
    };

    pub fn printWarnings(self: ChineseMotherboard, writer: anytype) !void {
        if (self.vrm.mosfet_type == .cheap) {
            try writer.print("WARNING: Cheap MOSFETs! Limit power to {d}W\n", .{self.vrm.max_tdp_per_socket});
        }
        if (self.vrm.heatsink_quality == .poor) {
            try writer.print("WARNING: Poor VRM heatsink! Add fan immediately!\n", .{});
        }
    }
};
