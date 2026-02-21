const std = @import("std");

pub fn prefetchRead(addr: *const anyopaque) void {
    asm volatile ("prefetcht0 (%[addr])"
        :
        : [addr] "r" (addr),
        : "memory"
    );
}

pub fn prefetchWrite(addr: *anyopaque) void {
    asm volatile ("prefetchw (%[addr])"
        :
        : [addr] "r" (addr),
        : "memory"
    );
}

pub const CacheOptimizer = struct {
    l1_size: usize,
    l2_size: usize,
    l3_size: usize,
    cache_line: usize,

    pub fn detect() CacheOptimizer {
        // Valores típicos x86-64
        return .{
            .l1_size = 32 * 1024,
            .l2_size = 512 * 1024,
            .l3_size = 32 * 1024 * 1024,
            .cache_line = 64,
        };
    }

    pub fn getBlockSize(self: CacheOptimizer, element_size: usize) usize {
        // Usar 1/4 del L1 para evitar eviction
        return (self.l1_size / 4) / element_size;
    }
};
