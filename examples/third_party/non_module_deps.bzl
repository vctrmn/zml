load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _non_module_deps_impl(mctx):

    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/Hejsil/zig-clap.git",
        commit = "068c38f89814079635692c7d0be9f58508c86173",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )


    new_git_repository(
        name = "com_github_zigzap_zap",
        remote = "https://github.com/zigzap/zap.git",
        tag = "v0.9.1",
        build_file = "//:third_party/com_github_zigzap_zap/zap.bazel",
        patch_cmds = [
            """sed -i'.bak' 's/inline static uint8_t seek2ch/inline static uint8_t facil_seek2ch/g' facil.io/lib/facil/redis/resp_parser.h""",
            """sed -i'.bak' 's/static ws_s \\*new_websocket();/static ws_s \\*new_websocket(intptr_t uuid);/g' facil.io/lib/facil/http/websockets.c""",
        ]
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
