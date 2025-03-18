const std = @import("std");
const zap = @import("zap");
const clap = @import("clap");
const asynk = @import("async");
const zml = @import("zml");
const modernbert = @import("modernbert.zig");
const stdx = @import("stdx");

const log = std.log.scoped(.server);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .modernbert, .level = .info },
    },
    .logFn = asynk.logFn(std.log.defaultLog),
};

// ---- CLI Parsing Configuration ----

/// Command-line argument parameters definition
const cli_params = clap.parseParamsComptime(
    \\--help                                print this help
    \\--model                   <PATH>      model path
    \\--tokenizer               <PATH>      tokenizer path
    \\--seq-len                 <UINT>      sequence length
    \\--num-attention-heads     <UINT>      number of attention heads
    \\--tie-word-embeddings     <BOOL>      default: false: tied weights
    \\--port                    <UINT>      port to listen on (default: 3000)
    \\--create-options          <STRING>    platform creation options JSON, defaults to {}
    \\--sharding                <BOOL>      default: true: sharding on or off
);

/// CLI parameter parsers
const cli_parsers = .{
    .BOOL = parseBool,
    .UINT = clap.parsers.int(usize, 0),
    .STRING = clap.parsers.string,
    .PATH = clap.parsers.string,
};

/// Parse boolean values from command line arguments
fn parseBool(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

// ---- Data Structures ----

/// Request for masked language model prediction
const UnmaskRequest = struct {
    text: []const u8,
};

/// Result for a single token prediction
const TokenPrediction = struct {
    token: []const u8,
    score: f32,
};

/// Response for masked language model prediction
const UnmaskResponse = struct {
    text: []const u8,
    predictions: []TokenPrediction,
    processing_time_ms: u64,
};

/// Error response structure
const ErrorResponse = struct {
    error_message: []const u8,
    code: u32,
};

// ---- HTTP Route Handlers ----

/// Handler for the unmask API route
const UnmaskRouteHandler = struct {
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    bert_module: zml.ModuleExe(modernbert.ModernBertForMaskedLM.forward),
    seq_len: i64,
    route: zap.Endpoint = undefined,
    mutex: std.Thread.Mutex = .{},

    /// Initialize the unmask route handler
    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer: zml.tokenizer.Tokenizer,
        bert_module: zml.ModuleExe(modernbert.ModernBertForMaskedLM.forward),
        seq_len: i64,
    ) !*UnmaskRouteHandler {
        const handler = try allocator.create(UnmaskRouteHandler);
        handler.* = .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .bert_module = bert_module,
            .seq_len = seq_len,
            .mutex = .{},
        };

        // Initialize the endpoint with our handler methods
        handler.route = zap.Endpoint.init(.{
            .path = "/api/unmask",
            .post = handlePostRequest,
            .options = handleOptionsRequest,
            .get = handleMethodNotAllowed,
            .put = handleMethodNotAllowed,
            .delete = handleMethodNotAllowed,
            .patch = handleMethodNotAllowed,
        });

        return handler;
    }

    /// Free resources when done
    pub fn deinit(self: *UnmaskRouteHandler) void {
        self.allocator.destroy(self);
    }

    /// Get the route endpoint
    pub fn getRoute(self: *UnmaskRouteHandler) *zap.Endpoint {
        return &self.route;
    }

    /// Handle OPTIONS HTTP method
    fn handleOptionsRequest(route: *zap.Endpoint, request: zap.Request) void {
        _ = route;
        request.setHeader("Access-Control-Allow-Origin", "*") catch return;
        request.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS") catch return;
        request.setStatus(zap.StatusCode.no_content);
        request.markAsFinished(true);
    }

    /// Handle POST HTTP method for unmask requests
    fn handlePostRequest(route: *zap.Endpoint, request: zap.Request) void {
        // Get our handler instance from the endpoint
        const self: *UnmaskRouteHandler = @fieldParentPtr("route", route);

        // Create a new arena for this specific request
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const request_allocator = arena.allocator();

        // Check request body
        if (request.body == null) {
            sendJsonError(request, "Missing request body", .bad_request);
            return;
        }

        // Parse JSON request
        const unmask_request = parseJsonUnmaskRequest(request_allocator, request.body.?) catch {
            sendJsonError(request, "Invalid JSON", .bad_request);
            return;
        };

        // Check for [MASK] token
        if (std.mem.indexOf(u8, unmask_request.text, "[MASK]") == null) {
            sendJsonError(request, "No [MASK] token found in text", .bad_request);
            return;
        }

        // Process the unmask request and measure time
        const start_time = std.time.milliTimestamp();

        // Lock the mutex before processing to ensure thread safety
        self.mutex.lock();
        defer self.mutex.unlock();

        const predictions = self.processUnmaskRequest(request_allocator, unmask_request.text) catch |err| {
            log.err("Error while processing unmask: {}", .{err});
            sendJsonError(request, "Error while processing unmask", .internal_server_error);
            return;
        };

        const processing_time = std.time.milliTimestamp() - start_time;

        // Create and send response
        const response = UnmaskResponse{
            .text = unmask_request.text,
            .predictions = predictions,
            .processing_time_ms = @intCast(processing_time),
        };

        sendJsonResponse(request, response, .ok);
    }

    /// Process masked language modeling request \n
    /// Based on huggingface/transformers fill_mask pipeline
    fn processUnmaskRequest(self: *UnmaskRouteHandler, allocator: std.mem.Allocator, text: []const u8) ![]TokenPrediction {
        var tokenizer_decoder = try self.tokenizer.decoder();
        defer tokenizer_decoder.deinit();

        const pad_token = self.tokenizer.tokenToId("[PAD]") orelse return error.NoSuchToken;
        const mask_token = self.tokenizer.tokenToId("[MASK]") orelse return error.NoSuchToken;

        // Tokenize input text
        const tokens = try tokenizeText(allocator, self.tokenizer, text);
        defer allocator.free(tokens);

        // Find "[MASK]" positions
        const mask_positions = try findMaskPositions(allocator, tokens, mask_token);
        defer allocator.free(mask_positions);

        // Prepare input tensors
        const input_tokens = try prepareModelInputs(allocator, tokens, self.seq_len, pad_token);
        defer allocator.free(input_tokens);

        // Create input tensors (on the accelerator)
        const input_shape = zml.Shape.init(.{ .b = 1, .s = self.seq_len }, .i64);
        const input_ids_tensor = try zml.Buffer.fromSlice(self.bert_module.platform(), input_shape.dims(), input_tokens);
        defer input_ids_tensor.deinit();

        // Model inference (retrieve indices)
        var inference_timer = try std.time.Timer.start();
        var topk = self.bert_module.call(.{input_ids_tensor});
        defer zml.aio.unloadBuffers(&topk);
        const inference_time = inference_timer.read();

        // Create the prediction results
        var predictions = std.ArrayList(TokenPrediction).init(allocator);

        // Transfer the result to host memory (CPU)
        var indices_host_buffer = try topk.indices.toHostAlloc(allocator);
        defer indices_host_buffer.deinit(allocator);
        var values_host_buffer = try topk.values.toHostAlloc(allocator);
        defer values_host_buffer.deinit(allocator);

        // We consider only the first occurrence of [MASK], which has five predictions
        const pred_offset = mask_positions[0] * 5;
        const prediction_tokens = indices_host_buffer.items(i32)[pred_offset..][0..5];
        const prediction_scores = values_host_buffer.items(f32)[pred_offset..][0..5];

        // Log timing information
        log.info("⏱️\tModel inference in {d}ms", .{inference_time / std.time.ns_per_ms});

        // Create the prediction results
        for (prediction_tokens, prediction_scores) |token_id, score| {
            const token_text = try tokenizer_decoder.next(@intCast(token_id));
            if (token_text) |word| {
                try predictions.append(.{
                    .token = try allocator.dupe(u8, word),
                    .score = score,
                });
            }
        }

        return predictions.toOwnedSlice();
    }
};

/// Handler for health check route
const HealthCheckRouteHandler = struct {
    route: zap.Endpoint = undefined,

    pub fn init() HealthCheckRouteHandler {
        return .{
            .route = zap.Endpoint.init(.{
                .path = "/api/healthz",
                .get = handleGetRequest,
                .options = handleOptionsRequest,
                .post = handleMethodNotAllowed,
                .put = handleMethodNotAllowed,
                .delete = handleMethodNotAllowed,
                .patch = handleMethodNotAllowed,
            }),
        };
    }

    pub fn getRoute(self: *HealthCheckRouteHandler) *zap.Endpoint {
        return &self.route;
    }

    fn handleOptionsRequest(route: *zap.Endpoint, request: zap.Request) void {
        _ = route;
        request.setHeader("Access-Control-Allow-Origin", "*") catch return;
        request.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS") catch return;
        request.setStatus(.no_content);
        request.markAsFinished(true);
    }

    fn handleGetRequest(route: *zap.Endpoint, request: zap.Request) void {
        _ = route;
        request.setStatus(.no_content);
        request.markAsFinished(true);
    }
};

// ---- Utility Functions ----

/// Parse JSON request helper
fn parseJsonUnmaskRequest(allocator: std.mem.Allocator, body: []const u8) !UnmaskRequest {
    const result = try std.json.parseFromSlice(UnmaskRequest, allocator, body, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    });
    return result.value;
}

/// Send JSON response helper
fn sendJsonResponse(request: zap.Request, data: anytype, status: zap.StatusCode) void {
    var json_buffer: [2048]u8 = undefined;
    const json_data = zap.stringifyBuf(&json_buffer, data, .{});

    if (json_data) |response| {
        request.setStatus(status);
        request.setContentType(.JSON) catch return;
        request.sendBody(response) catch return;
    } else {
        log.err("Error: Failed to serialize JSON response", .{});
        request.setStatus(.internal_server_error);
        request.sendBody("Error: Failed to serialize JSON response") catch return;
    }
}

/// Error response helper
fn sendJsonError(request: zap.Request, error_message: []const u8, status: zap.StatusCode) void {
    log.warn("Error: {s} (status: {})", .{ error_message, @intFromEnum(status) });

    const error_response = ErrorResponse{
        .error_message = error_message,
        .code = @intFromEnum(status),
    };

    sendJsonResponse(request, error_response, status);
}

/// Default handler for requests that don't match any route
fn handleNotFound(request: zap.Request) void {
    request.setStatus(.not_found);
    sendJsonError(request, "Resource not found", .not_found);
}

/// Method not allowed handler
fn handleMethodNotAllowed(route: *zap.Endpoint, request: zap.Request) void {
    _ = route;
    sendJsonError(request, "Method not allowed", .method_not_allowed);
}

/// Tokenize text for ModernBERT input
fn tokenizeText(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, text: []const u8) ![]const u32 {
    var tokens = std.ArrayList(u32).init(allocator);
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const bos = tokenizer.tokenToId("[CLS]") orelse return error.NoSuchToken;
    const eos = tokenizer.tokenToId("[SEP]") orelse return error.NoSuchToken;

    try tokens.append(bos);
    try tokens.appendSlice(try encoder.encode(text));
    try tokens.append(eos);

    return tokens.toOwnedSlice();
}

/// Find positions of mask tokens in the input
fn findMaskPositions(allocator: std.mem.Allocator, tokens: []const u32, mask_token: u32) ![]usize {
    var mask_positions = std.ArrayList(usize).init(allocator);
    defer mask_positions.deinit();

    for (tokens, 0..) |token, i| {
        if (token == mask_token) {
            try mask_positions.append(i);
        }
    }

    if (mask_positions.items.len == 0) {
        log.err("Input text must contain `[MASK]`", .{});
        return error.InvalidInput;
    }

    if (mask_positions.items.len > 1) {
        log.warn("Currently only supporting one [MASK] per input", .{});
    }

    return mask_positions.toOwnedSlice();
}

/// Prepare input tokens for the model
fn prepareModelInputs(allocator: std.mem.Allocator, tokens: []const u32, seq_len: i64, pad_token: u32) ![]u32 {
    const input_ids = try allocator.alloc(u32, @intCast(seq_len));

    // Fill with padding tokens
    @memset(input_ids, pad_token);

    // Copy tokens into the padded array
    for (tokens, 0..) |token, i| {
        input_ids[i] = @intCast(token);
    }

    return input_ids;
}

/// Print CLI usage and exit
fn printUsageAndExit(stderr: anytype) noreturn {
    stderr.print("usage: ", .{}) catch {};
    clap.usage(stderr, clap.Help, &cli_params) catch {};
    stderr.print("\n", .{}) catch {};
    std.process.exit(0);
}

// ---- Main Application ----

/// Entry point
pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

/// Main async function
pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;
    const stderr = std.io.getStdErr().writer();

    // Parse command line arguments
    var diag: clap.Diagnostic = .{};
    var cli = clap.parse(clap.Help, &cli_params, cli_parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.report(stderr, err);
        try printUsageAndExit(stderr);
    };
    defer cli.deinit();

    // Show help if requested
    if (cli.args.help != 0) {
        try clap.help(stderr, clap.Help, &cli_params, .{});
        return;
    }

    // Ensure cache directory exists
    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/modernbert/cache");

    // Initialize ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Setup platform and compilation options
    const create_opts_json = cli.args.@"create-options" orelse "{}";
    const create_opts = try std.json.parseFromSliceLeaky(zml.Platform.CreateOptions, allocator, create_opts_json, .{});
    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/modernbert",
        .sharding_enabled = cli.args.sharding orelse true,
    };

    // Auto-select platform
    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    // Check for required model path
    const model_file = cli.args.model orelse {
        stderr.print("Error: missing --model=...\n\n", .{}) catch {};
        printUsageAndExit(stderr);
        unreachable;
    };

    // Open the model file
    var tensor_store = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer tensor_store.deinit();

    // Create memory arena for model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    // Load tokenizer
    var tokenizer = blk: {
        if (cli.args.tokenizer) |tok| {
            log.info("\tLoading tokenizer from {s}", .{tok});
            var timer = try stdx.time.Timer.start();
            defer log.info("✅\tLoaded tokenizer from {s} [{}]", .{ tok, timer.read() });

            break :blk try zml.tokenizer.Tokenizer.fromFile(model_arena, tok);
        } else {
            log.err("Error: missing --tokenizer", .{});
            return;
        }
    };
    defer tokenizer.deinit();

    // Configure ModernBERT model
    const modernbert_options = modernbert.ModernBertOptions{
        .pad_token = tokenizer.tokenToId("[PAD]") orelse return error.NoSuchToken,
        .num_attention_heads = @intCast(cli.args.@"num-attention-heads" orelse 12),
        .tie_word_embeddings = cli.args.@"tie-word-embeddings" orelse false,
        .local_attention = 128,
    };

    // Initialize model from tensor store
    var modern_bert_for_masked_lm = try zml.aio.populateModel(modernbert.ModernBertForMaskedLM, model_arena, tensor_store);
    modern_bert_for_masked_lm.init(modernbert_options);

    log.info("\tModernBERT options: {}", .{modernbert_options});

    // Prepare for model loading
    const seq_len = @as(i64, @intCast(cli.args.@"seq-len" orelse 256));
    const input_shape = zml.Shape.init(.{ .b = 1, .s = seq_len }, .u32);

    var start = try std.time.Timer.start();

    // Load model weights
    log.info("\tLoading ModernBERT weights from {?s}...", .{model_file});
    var bert_weights = try zml.aio.loadBuffers(modernbert.ModernBertForMaskedLM, .{modernbert_options}, tensor_store, model_arena, platform);
    defer zml.aio.unloadBuffers(&bert_weights);
    log.info("✅\tLoaded weights in {d}ms", .{start.read() / std.time.ns_per_ms});

    // Compile the model
    log.info("\tCompiling ModernBERT model...", .{});
    var fut_mod = try asynk.asyncc(zml.compile, .{
        allocator,
        modernbert.ModernBertForMaskedLM.forward,
        .{modernbert_options},
        .{input_shape},
        tensor_store,
        platform,
    });
    var bert_module = (try fut_mod.awaitt()).prepare(bert_weights);
    defer bert_module.deinit();
    log.info("✅\tLoaded weights and compiled model in {d}ms", .{start.read() / std.time.ns_per_ms});

    // Initialize HTTP route handlers
    var unmask_handler = try UnmaskRouteHandler.init(
        allocator,
        tokenizer,
        bert_module,
        seq_len,
    );
    defer unmask_handler.deinit();

    var healthcheck_handler = HealthCheckRouteHandler.init();

    // Create HTTP server with route support
    const port = cli.args.port orelse 3000;
    var server = zap.Endpoint.Listener.init(allocator, .{
        .port = port,
        .on_request = handleNotFound,
        .log = true,
    });
    defer server.deinit();

    // Register routes
    try server.register(healthcheck_handler.getRoute());
    try server.register(unmask_handler.getRoute());

    // Start HTTP server
    try server.listen();

    log.info("✅\tModernBERT server listening on localhost:{d}", .{port});
    log.info("📝\tExample usage: curl -X POST http://localhost:{d}/api/unmask -H \"Content-Type: application/json\" -d '{{\"text\":\"The capital of France is [MASK].\"}}'", .{port});

    zap.start(.{
        .threads = 1,
        .workers = 1,
    });
}
