// test_penelope.zig — integration tests for the Zig version of penelope
// Compiles penelope.zig, runs the binary, and checks output.

const std = @import("std");
const testing = std.testing;
const Child = std.process.Child;

const penelope_src = "../penelope.zig";
const penelope_bin = "/tmp/penelope_zig_test_bin";
const save_path = "/tmp/penelope_zig_test_model.bin";

fn runPenelope(allocator: std.mem.Allocator, args: []const []const u8) !Child.RunResult {
    var argv: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.append(allocator, penelope_bin);
    for (args) |a| try argv.append(allocator, a);
    return Child.run(.{
        .allocator = allocator,
        .argv = argv.items,
        .max_output_bytes = 256 * 1024,
    });
}

fn contains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

test "zig compiles" {
    const result = try Child.run(.{
        .allocator = testing.allocator,
        .argv = &.{ "zig", "build-exe", penelope_src, "-femit-bin=" ++ penelope_bin },
        .max_output_bytes = 64 * 1024,
    });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);
    try testing.expect(result.term == .Exited and result.term.Exited == 0);
}

test "header output" {
    const result = try runPenelope(testing.allocator, &.{"darkness"});
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);
    const out = if (result.stdout.len > 0) result.stdout else result.stderr;
    try testing.expect(contains(out, "1984 words"));
    try testing.expect(contains(out, "Dario Equation"));
    try testing.expect(contains(out, "Arianna Method"));
}

test "param count" {
    const result = try runPenelope(testing.allocator, &.{"darkness"});
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);
    const out = if (result.stdout.len > 0) result.stdout else result.stderr;
    try testing.expect(contains(out, "13152768"));
}

test "generates words" {
    const result = try runPenelope(testing.allocator, &.{ "darkness", "eats" });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);
    const out = if (result.stdout.len > 0) result.stdout else result.stderr;
    // Count lines matching word output (indented words, possibly starred)
    var count: usize = 0;
    var iter = std.mem.splitScalar(u8, out, '\n');
    while (iter.next()) |line| {
        const trimmed = std.mem.trimLeft(u8, line, " ");
        if (trimmed.len == 0) continue;
        // Word lines start with spaces or "  *"
        if (line.len > 2 and (line[0] == ' ' or line[0] == '\t')) {
            if (trimmed[0] == '*' or (trimmed[0] >= 'a' and trimmed[0] <= 'z')) {
                count += 1;
            }
        }
    }
    try testing.expect(count >= 12);
}

test "prophecy target" {
    const result = try runPenelope(testing.allocator, &.{ "darkness", "eats" });
    defer testing.allocator.free(result.stdout);
    defer testing.allocator.free(result.stderr);
    const out = if (result.stdout.len > 0) result.stdout else result.stderr;
    try testing.expect(contains(out, "destined:"));
}

test "save file size" {
    // Pipe empty stdin to trigger interactive mode exit, with --save
    var child = Child.init(&.{ penelope_bin, "--save", save_path }, testing.allocator);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    try child.spawn();
    // Close stdin immediately so the program exits
    child.stdin.?.close();
    child.stdin = null;

    var stdout: std.ArrayListUnmanaged(u8) = .empty;
    defer stdout.deinit(testing.allocator);
    var stderr: std.ArrayListUnmanaged(u8) = .empty;
    defer stderr.deinit(testing.allocator);
    try child.collectOutput(testing.allocator, &stdout, &stderr, 256 * 1024);
    _ = try child.wait();

    defer std.fs.deleteFileAbsolute(save_path) catch {};

    const file = try std.fs.openFileAbsolute(save_path, .{});
    defer file.close();
    const stat = try file.stat();
    const expected: u64 = 16 + 13152768 * 4; // 52,611,088 bytes
    try testing.expectEqual(expected, stat.size);
}
