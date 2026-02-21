pub const Zen2BiosSettings = struct {
    pub const Status = enum { enabled, disabled, auto };
    pub const IdleControl = enum { typical_current, low_current };
    pub const MemProfile = enum { docp_xmp, auto, manual };
    pub const CmdRate = enum { @"1t", @"2t", auto };

    pub const performance = .{
        .precision_boost_overdrive = Status.enabled,
        .cppc = Status.enabled,
        .cppc_preferred_cores = Status.enabled,
        .global_c_state_control = Status.disabled,
        .power_supply_idle_control = IdleControl.typical_current,
        .oc_mode = Status.auto,
        .memory_profile = MemProfile.docp_xmp,
        .fclk_frequency = @as(u32, 1600),
        .cmd_rate = CmdRate.@"1t",
        .max_cpu_boost_clock_override = @as(u32, 200),
        .platform_thermal_limit = @as(u32, 95),
    };
};
