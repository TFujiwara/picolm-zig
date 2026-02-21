const std = @import("std");
const QuantType = @import("types.zig").QuantType;

pub const GGUF_MAGIC = 0x46554747;
pub const GGUF_DEFAULT_ALIGNMENT = 32;

pub const MetadataValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,

    pub fn size(self: MetadataValueType) u32 {
        return switch (self) {
            .u8, .i8, .bool => 1,
            .u16, .i16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
            .string, .array => 0,
        };
    }
};

pub const MetadataValue = union(MetadataValueType) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    array: ArrayValue,
    u64: u64,
    i64: i64,
    f64: f64,

    pub fn format(self: MetadataValue, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        switch (self) {
            .u8 => |v| try writer.print("{d}", .{v}),
            .i8 => |v| try writer.print("{d}", .{v}),
            .u16 => |v| try writer.print("{d}", .{v}),
            .i16 => |v| try writer.print("{d}", .{v}),
            .u32 => |v| try writer.print("{d}", .{v}),
            .i32 => |v| try writer.print("{d}", .{v}),
            .f32 => |v| try writer.print("{d:.6}", .{v}),
            .bool => |v| try writer.print("{}", .{v}),
            .string => |v| try writer.print("\"{s}\"", .{v}),
            .array => |v| try writer.print("[array {d} items]", .{v.len}),
            .u64 => |v| try writer.print("{d}", .{v}),
            .i64 => |v| try writer.print("{d}", .{v}),
            .f64 => |v| try writer.print("{d:.6}", .{v}),
        }
    }
};

pub const ArrayValue = struct {
    type_id: MetadataValueType,
    data: []const u8,
    len: u64,

    pub fn asSlice(self: ArrayValue, comptime T: type) []const T {
        return @as([*]const T, @ptrCast(@alignCast(self.data.ptr)))[0..@as(usize, @intCast(self.len))];
    }

    pub fn getString(self: ArrayValue, index: usize) ?[]const u8 {
        if (self.type_id != .string) return null;
        var offset: usize = 0;
        var i: usize = 0;
        while (i < index) : (i += 1) {
            if (offset + 8 > self.data.len) return null;
            const str_len = std.mem.readInt(u64, self.data[offset..][0..8], .little);
            offset += 8 + @as(usize, @intCast(str_len));
        }
        if (offset + 8 > self.data.len) return null;
        const str_len = std.mem.readInt(u64, self.data[offset..][0..8], .little);
        return self.data[offset + 8 .. offset + 8 + @as(usize, @intCast(str_len))];
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dimensions: []const u64,
    type_id: QuantType,
    offset: u64,
    size: u64,

    pub fn numElements(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dimensions) |d| n *= d;
        return n;
    }

    pub fn format(self: TensorInfo, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{s} [", .{self.name});
        for (self.dimensions, 0..) |dim, i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{d}", .{dim});
        }
        try writer.print("] {s} offset={d} size={d}MB", .{
            @tagName(self.type_id),
            self.offset,
            self.size / (1024 * 1024),
        });
    }
};

pub const GGUFModel = struct {
    allocator: std.mem.Allocator,
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    metadata: std.StringHashMap(MetadataValue),
    tensors: std.StringHashMap(TensorInfo),
    data_offset: u64,
    alignment: u32,
    mapped_file: ?std.fs.File,
    mapped_mem: ?[]align(std.heap.page_size_min) u8,
    file_size: u64,

    // Cached metadata
    arch: ?[]const u8 = null,
    vocab_size: ?u32 = null,
    context_length: ?u32 = null,
    embedding_length: ?u32 = null,
    block_count: ?u32 = null,
    feed_forward_length: ?u32 = null,
    attention_head_count: ?u32 = null,
    attention_head_count_kv: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator) GGUFModel {
        return .{
            .allocator = allocator,
            .magic = 0,
            .version = 0,
            .tensor_count = 0,
            .metadata_count = 0,
            .metadata = std.StringHashMap(MetadataValue).init(allocator),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .data_offset = 0,
            .alignment = GGUF_DEFAULT_ALIGNMENT,
            .mapped_file = null,
            .mapped_mem = null,
            .file_size = 0,
        };
    }

    pub fn deinit(self: *GGUFModel) void {
        if (self.mapped_mem) |mem| {
            if (@import("builtin").os.tag == .windows) {
                self.allocator.free(mem);
            } else {
                std.posix.munmap(mem);
            }
        }
        if (self.mapped_file) |file| file.close();

        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            switch (entry.value_ptr.*) {
                .string => |s| self.allocator.free(s),
                .array => |a| self.allocator.free(a.data),
                else => {},
            }
        }
        self.metadata.deinit();

        var tensor_iter = self.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*.name);
            self.allocator.free(entry.value_ptr.*.dimensions);
        }
        self.tensors.deinit();
    }

    pub fn load(self: *GGUFModel, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        self.mapped_file = file;

        const stat = try file.stat();
        self.file_size = stat.size;

        const ptr = if (@import("builtin").os.tag == .windows) blk: {
            const m = try self.allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(std.heap.page_size_min), self.file_size);
            var reader_buf: [4096]u8 = undefined;
            var r = file.reader(&reader_buf);
            try r.interface.readSliceAll(m);
            break :blk m;
        } else try std.posix.mmap(null, self.file_size, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, file.handle, 0);
        self.mapped_mem = ptr;

        var stream = std.io.fixedBufferStream(ptr);
        const reader = stream.reader();

        self.magic = try reader.readInt(u32, .little);
        if (self.magic != GGUF_MAGIC) return error.InvalidMagic;

        self.version = try reader.readInt(u32, .little);
        if (self.version < 2 or self.version > 3) return error.UnsupportedVersion;

        self.tensor_count = try reader.readInt(u64, .little);
        self.metadata_count = try reader.readInt(u64, .little);

        try self.parseMetadata(reader);
        try self.parseTensors(reader);

        self.data_offset = self.alignOffset(stream.pos);
        try self.cacheCommonMetadata();

        if (comptime @hasDecl(std.posix, "madvise") and std.posix.MADV != void) {
            _ = std.posix.madvise(ptr.ptr, self.file_size, std.posix.MADV.SEQUENTIAL) catch {};
        }
    }

    fn parseMetadata(self: *GGUFModel, reader: anytype) !void {
        var i: u64 = 0;
        while (i < self.metadata_count) : (i += 1) {
            const key_len = try reader.readInt(u64, .little);
            const key = try self.allocator.alloc(u8, @as(usize, @intCast(key_len)));
            _ = try reader.readAll(key);

            const value_type = @as(MetadataValueType, @enumFromInt(try reader.readInt(u32, .little)));
            const value = try self.readValue(reader, value_type);

            try self.metadata.put(key, value);
        }
    }

    fn readValue(self: *GGUFModel, reader: anytype, t: MetadataValueType) !MetadataValue {
        return switch (t) {
            .u8 => .{ .u8 = try reader.readInt(u8, .little) },
            .i8 => .{ .i8 = try reader.readInt(i8, .little) },
            .u16 => .{ .u16 = try reader.readInt(u16, .little) },
            .i16 => .{ .i16 = try reader.readInt(i16, .little) },
            .u32 => .{ .u32 = try reader.readInt(u32, .little) },
            .i32 => .{ .i32 = try reader.readInt(i32, .little) },
            .f32 => .{ .f32 = @bitCast(try reader.readInt(u32, .little)) },
            .bool => .{ .bool = (try reader.readInt(u8, .little)) != 0 },
            .string => blk: {
                const len = try reader.readInt(u64, .little);
                const str = try self.allocator.alloc(u8, @as(usize, @intCast(len)));
                _ = try reader.readAll(str);
                break :blk .{ .string = str };
            },
            .array => try self.readArray(reader),
            .u64 => .{ .u64 = try reader.readInt(u64, .little) },
            .i64 => .{ .i64 = try reader.readInt(i64, .little) },
            .f64 => .{ .f64 = @bitCast(try reader.readInt(u64, .little)) },
        };
    }

    fn readArray(self: *GGUFModel, reader: anytype) !MetadataValue {
        const elem_type = @as(MetadataValueType, @enumFromInt(try reader.readInt(u32, .little)));
        const len = try reader.readInt(u64, .little);

        if (elem_type == .string) {
            var total_size: usize = 0;
            const start_pos = try reader.context.getPos();
            var i: u64 = 0;
            while (i < len) : (i += 1) {
                const str_len = try reader.readInt(u64, .little);
                try reader.skipBytes(@as(usize, @intCast(str_len)), .{});
                total_size += 8 + @as(usize, @intCast(str_len));
            }
            try reader.context.seekTo(start_pos);
            const data = try self.allocator.alloc(u8, total_size);
            _ = try reader.readAll(data);
            return .{ .array = .{ .type_id = elem_type, .data = data, .len = len } };
        } else {
            const elem_size = elem_type.size();
            const data = try self.allocator.alloc(u8, @as(usize, @intCast(len * elem_size)));
            _ = try reader.readAll(data);
            return .{ .array = .{ .type_id = elem_type, .data = data, .len = len } };
        }
    }

    fn parseTensors(self: *GGUFModel, reader: anytype) !void {
        var i: u64 = 0;
        while (i < self.tensor_count) : (i += 1) {
            const name_len = try reader.readInt(u64, .little);
            const name = try self.allocator.alloc(u8, @as(usize, @intCast(name_len)));
            _ = try reader.readAll(name);

            const n_dims = try reader.readInt(u32, .little);
            const dims = try self.allocator.alloc(u64, @as(usize, n_dims));
            for (0..@as(usize, n_dims)) |d| dims[d] = try reader.readInt(u64, .little);

            const type_id = @as(QuantType, @enumFromInt(try reader.readInt(u32, .little)));
            const offset = try reader.readInt(u64, .little);

            const info = TensorInfo{
                .name = name,
                .dimensions = dims,
                .type_id = type_id,
                .offset = offset,
                .size = self.calculateTensorSize(dims, type_id),
            };

            const key = try self.allocator.dupe(u8, name);
            try self.tensors.put(key, info);
        }
    }

    fn calculateTensorSize(_: GGUFModel, dims: []const u64, quant_type: QuantType) u64 {
        var num_elements: u64 = 1;
        for (dims) |d| num_elements *= d;

        return switch (quant_type) {
            .f32 => num_elements * 4,
            .f16 => num_elements * 2,
            .q4_0 => (num_elements / 32) * 18,
            .q4_1 => (num_elements / 32) * 20,
            .q8_0 => (num_elements / 32) * 34,
            .q2_k => (num_elements / 256) * 96,
            .q3_k => (num_elements / 256) * 110,
            .q4_k => (num_elements / 256) * 144,
            .q5_k => (num_elements / 256) * 176,
            .q6_k => (num_elements / 256) * 210,
            .q8_k => (num_elements / 256) * 292,
            else => num_elements * 4,
        };
    }

    fn alignOffset(self: GGUFModel, offset: u64) u64 {
        const mask = @as(u64, self.alignment) - 1;
        return (offset + mask) & ~mask;
    }

    fn cacheCommonMetadata(self: *GGUFModel) !void {
        self.arch = self.getString("general.architecture");
        self.vocab_size = if (self.getU32("llama.vocab_size")) |v| v else self.getU32("general.vocab_size");
        self.context_length = if (self.getU32("llama.context_length")) |v| v else self.getU32("general.context_length");
        self.embedding_length = self.getU32("llama.embedding_length");
        self.block_count = self.getU32("llama.block_count");
        self.feed_forward_length = self.getU32("llama.feed_forward_length");
        self.attention_head_count = self.getU32("llama.attention.head_count");
        self.attention_head_count_kv = self.getU32("llama.attention.head_count_kv");
    }

    pub fn get(self: GGUFModel, key: []const u8) ?MetadataValue {
        return self.metadata.get(key);
    }

    pub fn getString(self: GGUFModel, key: []const u8) ?[]const u8 {
        const val = self.metadata.get(key) orelse return null;
        return if (val == .string) val.string else null;
    }

    pub fn getU32(self: GGUFModel, key: []const u8) ?u32 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .u32 => |v| v,
            .i32 => |v| @intCast(v),
            else => null,
        };
    }

    pub fn getF32(self: GGUFModel, key: []const u8) ?f32 {
        const val = self.metadata.get(key) orelse return null;
        return if (val == .f32) val.f32 else null;
    }

    pub fn getTensor(self: GGUFModel, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn getTensorData(self: GGUFModel, info: TensorInfo) []const u8 {
        const offset = self.data_offset + info.offset;
        return self.mapped_mem.?[offset .. offset + info.size];
    }

    pub fn getTensorDataByName(self: GGUFModel, name: []const u8) ?[]const u8 {
        const info = self.tensors.get(name) orelse return null;
        return self.getTensorData(info);
    }

    pub fn printInfo(self: GGUFModel, writer: anytype) !void {
        try writer.print("=== GGUF Model Info ===\n", .{});
        try writer.print("Version: {d}\n", .{self.version});
        try writer.print("Tensors: {d}\n", .{self.tensor_count});
        try writer.print("Metadata: {d}\n", .{self.metadata_count});
        try writer.print("File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(self.file_size)) / (1024 * 1024)});

        if (self.arch) |a| try writer.print("Architecture: {s}\n", .{a});
        if (self.vocab_size) |v| try writer.print("Vocab: {d}\n", .{v});
        if (self.context_length) |v| try writer.print("Context: {d}\n", .{v});
        if (self.embedding_length) |v| try writer.print("Dim: {d}\n", .{v});
        if (self.block_count) |v| try writer.print("Layers: {d}\n", .{v});
    }
};
