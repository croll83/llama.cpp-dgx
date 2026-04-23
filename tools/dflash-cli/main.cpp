// tools/dflash-cli/main.cpp — thin CLI driver over the dflash27b session API.
//
// Argument parsing + daemon / one-shot dispatch. All decode logic lives in
// session.cpp behind the public API declared in dflash27b.h.

#include "dflash27b.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
  #include <io.h>
#else
  #include <unistd.h>
#endif

// ─── tiny I/O helpers ─────────────────────────────────────────────

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t) f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *) out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *) v.data(), v.size() * sizeof(int32_t));
    return (bool) f;
}

// ─── streaming-fd helper (writes int32 LE) ────────────────────────

static int g_stream_fd = -1;

static void stream_emit(int32_t tok) {
    if (g_stream_fd < 0) return;
    int32_t v = tok;
#if defined(_WIN32)
    DWORD written;
    WriteFile((HANDLE)(intptr_t) g_stream_fd, &v, sizeof(v), &written, nullptr);
#else
    ssize_t n = ::write(g_stream_fd, &v, sizeof(v));
    (void) n;
#endif
}

// ─── main ─────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <target.gguf> <draft.safetensors> [<prompt_ids.bin> <n_gen> <out_ids.bin>] [--daemon] ...\n",
            argv[0]);
        return 2;
    }

    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const char * prompt_path = (argc >= 6 && argv[3][0] != '-') ? argv[3] : nullptr;
    int          n_gen       = (argc >= 6 && argv[3][0] != '-') ? std::atoi(argv[4]) : 0;
    const char * out_path    = (argc >= 6 && argv[3][0] != '-') ? argv[5] : nullptr;

    dflash_session_params_t p = dflash_session_default_params();
    bool daemon_mode = false;

    for (int i = 3; i < argc; i++) {
        if      (std::strcmp(argv[i], "--daemon") == 0)        daemon_mode = true;
        else if (std::strcmp(argv[i], "--fast-rollback") == 0) p.fast_rollback = 1;
        else if (std::strcmp(argv[i], "--ddtree") == 0)      { p.ddtree = 1; p.fast_rollback = 1; }
        else if (std::strncmp(argv[i], "--ddtree-budget=", 16) == 0) {
            p.ddtree_budget = std::atoi(argv[i] + 16);
            if (p.ddtree_budget <= 0) p.ddtree_budget = 64;
        }
        else if (std::strncmp(argv[i], "--ddtree-temp=", 14) == 0) {
            p.ddtree_temp = (float) std::atof(argv[i] + 14);
            if (p.ddtree_temp <= 0.0f) p.ddtree_temp = 1.0f;
        }
        else if (std::strcmp(argv[i], "--ddtree-no-chain-seed") == 0) {
            p.ddtree_chain_seed = 0;
        }
        else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            g_stream_fd = std::atoi(argv[i] + 12);
        }
        else if (std::strncmp(argv[i], "--max-ctx=", 10) == 0) {
            p.max_ctx = std::atoi(argv[i] + 10);
        }
        // --seq-verify and --profile-scaling removed in Phase C refactor.
    }

    if (const char * s = std::getenv("DFLASH27B_KV_TBQ")) {
        if (std::atoi(s) != 0) p.kv_tbq = 1;
    }
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        p.prefill_ubatch = std::max(1, std::atoi(s));
    }

    if (!daemon_mode && (!prompt_path || !out_path)) {
        std::fprintf(stderr, "Missing positional arguments for non-daemon mode.\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    std::printf("[cfg] fast_rollback=%d ddtree=%d budget=%d temp=%.2f chain_seed=%d\n",
                p.fast_rollback, p.ddtree, p.ddtree_budget, p.ddtree_temp, p.ddtree_chain_seed);

    dflash_session_t * s = dflash_session_create(target_path, draft_path, &p, backend);
    if (!s) {
        std::fprintf(stderr, "session create: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    auto run_one = [&](const int32_t * prompt_ids, int n_prompt,
                       int n_gen_req, int append_mode,
                       std::vector<int32_t> & out_ids) -> bool {
        struct ctx_s { std::vector<int32_t> * out; } cctx{ &out_ids };
        auto cb = [](int32_t tok, void * uc) -> int {
            auto & oc = *((ctx_s *) uc);
            oc.out->push_back(tok);
            stream_emit(tok);
            return 1; // keep going
        };
        int rc = dflash_session_run(s, prompt_ids, n_prompt, n_gen_req, append_mode,
                                    cb, &cctx);
        return rc >= 0;
    };

    if (daemon_mode) {
        std::printf("[daemon] ready\n");
        std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            // Protocol:
            //   "PATH N_GEN"        (legacy / R)
            //   "R PATH N_GEN"      reset before prefill (full prompt)
            //   "A PATH N_GEN"      append (delta prompt only, keep cache)
            int append = 0;
            const char * parse_from = line.c_str();
            if (line.size() >= 2 && (line[0] == 'R' || line[0] == 'A') && line[1] == ' ') {
                append = (line[0] == 'A') ? 1 : 0;
                parse_from = line.c_str() + 2;
            }
            char ppath[1024];
            int  req_n_gen = 0;
            if (std::sscanf(parse_from, "%1023s %d", ppath, &req_n_gen) != 2) {
                stream_emit(-1);
                continue;
            }
            auto prompt = read_int32_file(ppath);
            if (prompt.empty()) {
                std::fprintf(stderr, "empty prompt\n");
                stream_emit(-1);
                continue;
            }
            std::vector<int32_t> out;
            if (!run_one(prompt.data(), (int) prompt.size(), req_n_gen, append, out)) {
                std::fprintf(stderr, "session_run failed\n");
            }
            stream_emit(-1);
        }
    } else {
        auto prompt = read_int32_file(prompt_path);
        if (prompt.empty()) {
            std::fprintf(stderr, "empty prompt\n");
            dflash_session_destroy(s);
            ggml_backend_free(backend);
            return 1;
        }
        std::vector<int32_t> out_all = prompt;
        std::vector<int32_t> gen;
        if (!run_one(prompt.data(), (int) prompt.size(), n_gen, /*append=*/0, gen)) {
            std::fprintf(stderr, "session_run failed\n");
            dflash_session_destroy(s);
            ggml_backend_free(backend);
            return 1;
        }
        out_all.insert(out_all.end(), gen.begin(), gen.end());
        if (out_path) write_int32_file(out_path, out_all);
    }

    dflash_session_destroy(s);
    ggml_backend_free(backend);
    return 0;
}
