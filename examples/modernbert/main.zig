const clap = @import("clap");
const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log.scoped(.modernbert);
const Tensor = zml.Tensor;
const modernbert = @import("modernbert.zig");
const stdx = @import("stdx");

// set this to false to disable the verbose logging
// const show_mlir = true;
pub const std_options = .{
    .log_level = .info,
};

fn softmax(logits: []f32, allocator: std.mem.Allocator) ![]f32 {
    var result = try allocator.alloc(f32, logits.len);
    errdefer allocator.free(result);

    // Find max for numerical stability
    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |logit| {
        max_logit = @max(max_logit, logit);
    }

    // Compute exp(logits - max_logit) and sum
    var exp_sum: f32 = 0.0;
    for (logits, 0..) |logit, i| {
        const exp_diff = @exp(logit - max_logit);
        result[i] = exp_diff;
        exp_sum += result[i];
    }

    if (exp_sum == 0) return error.DivisionByZero;

    // Normalize
    for (result) |*logit| {
        logit.* = logit.* / exp_sum;
    }

    return result;
}

// fill-mask pipeline
// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py
pub fn unmask(
    mod: zml.ModuleExe(modernbert.ModernBertForMaskedLM.forward),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    seq_len: i64,
    text: []const u8,
) !void {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    // tokenize input text
    const tokens: []const u32 = try tokenize(allocator, tokenizer, text);
    defer allocator.free(tokens);
    log.info("\tInput tokens: {d}", .{tokens});

    // find "[MASK]" positions
    var mask_positions = std.ArrayList(usize).init(allocator);
    defer mask_positions.deinit();

    for (tokens, 0..) |token, i| {
        if (token == 50284) {
            try mask_positions.append(i);
        }
    }

    if (mask_positions.items.len == 0) {
        log.err("Missing mask in input text : {s}", .{text});
        return error.InvalidInput;
    }

    // prepare input tensors
    var input_ids = try allocator.alloc(i64, @intCast(seq_len));
    var attention_mask = try allocator.alloc(i64, @intCast(seq_len));
    defer allocator.free(input_ids);
    defer allocator.free(attention_mask);

    // init all values to 0
    @memset(input_ids, 0);
    @memset(attention_mask, 0);

    // fill with tokens and set mask to 1
    for (tokens, 0..) |token, i| {
        input_ids[i] = @intCast(token);
        attention_mask[i] = 1;
    }

    // create input tensors
    const input_shape = zml.Shape.init(.{ .b = 1, .s = seq_len }, .i64);
    const input_ids_tensor = try zml.Buffer.fromSlice(mod.platform(), input_shape.dims(), input_ids);
    defer input_ids_tensor.deinit();
    const attention_mask_tensor = try zml.Buffer.fromSlice(mod.platform(), input_shape.dims(), attention_mask);
    defer attention_mask_tensor.deinit();

    // forward
    const outputs: zml.Buffer = mod.call(.{ input_ids_tensor, attention_mask_tensor });
    defer outputs.deinit();

    // logits
    var outputs_buffer = try outputs.toHostAlloc(allocator);
    defer outputs_buffer.deinit(allocator);

    const vocab_size = 50368; // tokenizer.tokens.len;
    const mask_pos = mask_positions.items[0];
    const base_offset = mask_pos * vocab_size;

    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    // copy is useless
    const raw_logits = @as([*]const f32, @ptrCast(@alignCast(outputs_buffer.data.ptr)));
    @memcpy(logits, raw_logits[base_offset..][0..vocab_size]);

    var indices = try allocator.alloc(usize, vocab_size);
    defer allocator.free(indices);
    for (0..vocab_size) |i| {
        indices[i] = i;
    }

    // sort desc
    const Context = struct {
        logits: []const f32,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.logits[a] > ctx.logits[b];
        }
    };
    std.mem.sort(usize, indices, Context{ .logits = logits }, Context.lessThan);

    // convert to probabilities
    const probabilities = try softmax(logits, allocator);
    defer allocator.free(probabilities);

    log.info("✅\tTop 5 predictions:", .{});
    for (indices[0..5]) |token_id| {
        const token_text = try tokenizer_decoder.next(@intCast(token_id));

        if (token_text) |text_slice| {
            log.info("\t\tscore: {d:.6}, word: '{s}', token: {}", .{
                probabilities[token_id],
                text_slice,
                token_id,
            });
        }
    }
}

const params = clap.parseParamsComptime(
    \\--help                                print this help
    \\--text                    <STRING>    the prompt
    \\--model                   <PATH>      model path
    \\--tokenizer               <PATH>      tokenizer path
    \\--seq-len                 <UINT>      sequence length
    \\--num-attention-heads     <UINT>      number of attention heads
    \\--create-options          <STRING>    platform creation options JSON, defaults to {}
    \\--sharding                <BOOL>      default: true: sharding on or off
);

pub fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

fn printUsageAndExit(stderr: anytype) noreturn {
    stderr.print("usage: ", .{}) catch {};
    clap.usage(stderr, clap.Help, &params) catch {};
    stderr.print("\n", .{}) catch {};
    std.process.exit(0);
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;
    const stderr = std.io.getStdErr().writer();

    // parse args
    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(usize, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };

    var diag: clap.Diagnostic = .{};
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.report(stderr, err);
        try printUsageAndExit(stderr);
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.help(stderr, clap.Help, &params, .{});
        return;
    }

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/modernbert/cache");

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/modernbert",
        .sharding_enabled = res.args.sharding orelse true,
    };

    const create_opts_json = res.args.@"create-options" orelse "{}";
    const create_opts = try std.json.parseFromSliceLeaky(zml.Platform.CreateOptions, allocator, create_opts_json, .{});
    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    const model_file = res.args.model.?;
    var ts = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer ts.deinit();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    // const num_attention_heads = if (res.args.@"num-attention-heads ") |num_heads| {
    //     num_heads;
    // } else {
    //     log.err("Missing required input --num-attention-heads", .{});
    //     return;
    // };

    const modernbert_options = modernbert.ModernBertOptions{
        .num_attention_heads = 12, // TODO: from res.args
        .tie_word_embeddings = true, // TODO: from res.args
    };
    var modern_bert_for_masked_lm = try zml.aio.populateModel(modernbert.ModernBertForMaskedLM, model_arena, ts);

    modern_bert_for_masked_lm.init(modernbert_options);

    log.info("✅\tParsed ModernBERT config: {}", .{modernbert_options});

    var tokenizer = blk: {
        if (res.args.tokenizer) |tok| {
            log.info("Loading tokenizer from {s}", .{tok});
            var timer = try stdx.time.Timer.start();
            defer log.info("Loaded tokenizer from {s} [{}]", .{ tok, timer.read() });

            break :blk try zml.tokenizer.Tokenizer.fromFile(model_arena, tok);
        } else {
            log.err("Missing --tokenizer", .{});
            return;
        }
    };
    errdefer tokenizer.deinit();

    // Prepare shapes for compilation
    const seq_len = @as(i64, @intCast(res.args.@"seq-len" orelse 64));
    const input_shape = zml.Shape.init(.{ .b = 1, .s = seq_len }, .i64);
    const attention_mask_shape = input_shape;

    // Compile the model
    log.info("\tCompiling model...", .{});
    var start = try std.time.Timer.start();
    var fut_mod = try asynk.asyncc(zml.compile, .{
        allocator,
        modernbert.ModernBertForMaskedLM.forward,
        .{modernbert_options},
        .{ input_shape, attention_mask_shape },
        ts,
        platform,
    });

    // Load weights
    log.info("\tLoading ModernBERT weights from {s}...", .{model_file});
    var bert_weights = try zml.aio.loadBuffers(modernbert.ModernBertForMaskedLM, .{modernbert_options}, ts, model_arena, platform);
    defer zml.aio.unloadBuffers(&bert_weights);
    log.info("✅\tLoaded weights in {d}ms", .{start.read() / std.time.ns_per_ms});

    var bert_module = (try fut_mod.awaitt()).prepare(bert_weights);
    defer bert_module.deinit();
    log.info("✅\tCompiled model in {d}ms", .{start.read() / std.time.ns_per_ms});

    const text = res.args.text orelse "Paris is the [MASK] of France.";
    log.info("\tInput text: {s}", .{text});

    try unmask(bert_module, tokenizer, allocator, seq_len, text);
}

pub fn tokenize(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]const u32 {
    var tokens = std.ArrayList(u32).init(allocator);
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const bos = tokenizer.tokenToId("[CLS]") orelse return error.NoSuchToken;
    const eos = tokenizer.tokenToId("[SEP]") orelse return error.NoSuchToken;

    try tokens.append(bos);
    try tokens.appendSlice(try encoder.encode(prompt));
    try tokens.append(eos);

    return tokens.toOwnedSlice();
}
