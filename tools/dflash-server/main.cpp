// tools/dflash-server — OpenAI-compatible HTTP server on top of llama-dflash.
//
// Phase 2 of the DFlash integration: one native C++ binary (no Python,
// no external lucebox-hub server_tools.py) that serves /v1/chat/completions
// with DFlash MTP decoding under the hood.
//
// Architecture (Phase 2): spawns `llama-dflash --daemon` as a subprocess and
// drives it via the same protocol server_tools.py uses (prompt file path +
// n_gen over stdin, output tokens over a streaming fd). Phase 3 will merge
// the decode loop in-process and wire DFlash into tools/server proper with
// slot-level prompt caching.

#include "common.h"
#include "chat.h"
#include "llama.h"
#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// common.h already defines a top-level `json` alias; use the fully qualified
// name locally to avoid clashing.
using njson = nlohmann::ordered_json;

// ─── subprocess: llama-dflash --daemon ─────────────────────────────

struct dflash_daemon {
    pid_t pid       = -1;
    int   stdin_w   = -1;   // parent → child stdin
    int   stream_r  = -1;   // child → parent: int32 token stream (fd 5 in child)
    std::mutex mtx;

    bool start(const std::string & bin,
               const std::string & target,
               const std::string & draft,
               int   budget,
               int   max_ctx) {
        int stdin_pipe[2]  = {-1, -1}; // parent_w=1, child_r=0
        int stream_pipe[2] = {-1, -1}; // parent_r=0, child_w=1
        if (pipe(stdin_pipe) != 0 || pipe(stream_pipe) != 0) {
            fprintf(stderr, "pipe() failed: %s\n", strerror(errno));
            return false;
        }

        pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork() failed: %s\n", strerror(errno));
            return false;
        }

        if (pid == 0) {
            // child
            dup2(stdin_pipe[0], STDIN_FILENO);
            // leave stdout/stderr attached so we see daemon logs
            // stream_pipe[1] must end up at fd 5
            if (stream_pipe[1] != 5) {
                dup2(stream_pipe[1], 5);
                close(stream_pipe[1]);
            }
            close(stdin_pipe[0]);  close(stdin_pipe[1]);
            close(stream_pipe[0]);

            char budget_s[32], max_ctx_s[32];
            snprintf(budget_s, sizeof(budget_s), "--ddtree-budget=%d", budget);
            snprintf(max_ctx_s, sizeof(max_ctx_s), "--max-ctx=%d", max_ctx);

            // placeholder positional args (prompt + n_gen + out) are required
            // by the CLI parser but unused in daemon mode.
            const char * placeholder_prompt = "/dev/null";
            const char * placeholder_ngen   = "0";
            const char * placeholder_out    = "/dev/null";

            // llama-dflash TARGET DRAFT PROMPT N_GEN OUT [flags...] --daemon
            //   --fast-rollback --ddtree --ddtree-budget=B --max-ctx=N --stream-fd=5
            execlp(bin.c_str(), bin.c_str(),
                   target.c_str(), draft.c_str(),
                   placeholder_prompt, placeholder_ngen, placeholder_out,
                   "--daemon", "--fast-rollback", "--ddtree",
                   budget_s, max_ctx_s, "--stream-fd=5",
                   (char *) nullptr);
            perror("execlp");
            _exit(127);
        }

        // parent
        close(stdin_pipe[0]);
        close(stream_pipe[1]);
        stdin_w  = stdin_pipe[1];
        stream_r = stream_pipe[0];

        // give the daemon a moment to load; the caller can still race if the
        // model load exceeds this, but the protocol tolerates it (reads
        // block).
        return true;
    }

    // Generate up to n_gen tokens for the given prompt ids. Returns the list
    // of tokens emitted (may be shorter on early stop / sentinel).
    std::vector<int32_t> generate(const std::vector<int32_t> & ids, int n_gen) {
        std::lock_guard<std::mutex> lock(mtx);

        // Materialize the prompt as a temp file; the daemon reads int32 LE.
        char tmp_path[] = "/tmp/dflash-prompt-XXXXXX";
        int tmp_fd = mkstemp(tmp_path);
        if (tmp_fd < 0) return {};
        ssize_t total = 0;
        const ssize_t want = (ssize_t) ids.size() * 4;
        while (total < want) {
            ssize_t w = write(tmp_fd, (const char *)ids.data() + total, want - total);
            if (w <= 0) { close(tmp_fd); unlink(tmp_path); return {}; }
            total += w;
        }
        close(tmp_fd);

        // Dispatch to the daemon.
        char cmd[1024];
        int n = snprintf(cmd, sizeof(cmd), "%s %d\n", tmp_path, n_gen);
        if (write(stdin_w, cmd, n) != n) {
            unlink(tmp_path);
            return {};
        }

        // Collect tokens until we have n_gen of them or hit the -1 sentinel.
        std::vector<int32_t> out;
        out.reserve(n_gen);
        while ((int) out.size() < n_gen) {
            int32_t tok = 0;
            ssize_t r = read(stream_r, &tok, 4);
            if (r != 4) break;          // daemon died or pipe closed
            if (tok == -1) break;       // sentinel: error / EOS
            out.push_back(tok);
        }

        unlink(tmp_path);
        return out;
    }

    void stop() {
        if (pid > 0) {
            if (stdin_w >= 0) close(stdin_w);    // EOF triggers the daemon loop exit
            int status = 0;
            waitpid(pid, &status, 0);
        }
        if (stream_r >= 0) close(stream_r);
        pid = -1;
        stdin_w = stream_r = -1;
    }
};

// ─── globals ──────────────────────────────────────────────────────

struct server_state {
    std::string                 alias = "dark-opus";
    llama_model *               model = nullptr;
    const llama_vocab *         vocab = nullptr;
    common_chat_templates_ptr   tmpls;
    std::vector<llama_token>    stop_ids;
    dflash_daemon               daemon;
};

static server_state g;
static std::atomic<bool> g_shutdown{false};

// ─── helpers ──────────────────────────────────────────────────────

static std::string role_str(const njson & m) {
    return m.value("role", std::string(""));
}

static std::string content_str(const njson & m) {
    if (!m.contains("content") || m["content"].is_null()) return "";
    if (m["content"].is_string()) return m["content"].get<std::string>();
    // Multi-part content (images/tool chunks) not supported in Phase 2 — flatten.
    std::string s;
    for (const auto & part : m["content"]) {
        if (part.is_object() && part.contains("text")) s += part["text"].get<std::string>();
    }
    return s;
}

static std::string detokenize_range(const llama_vocab * vocab,
                                    const std::vector<int32_t> & ids,
                                    const std::vector<llama_token> & stops) {
    std::string out;
    char buf[512];
    for (int32_t id : ids) {
        if (std::find(stops.begin(), stops.end(), (llama_token) id) != stops.end()) break;
        int n = llama_token_to_piece(vocab, (llama_token) id, buf, sizeof(buf), 0, /*special*/ false);
        if (n > 0) out.append(buf, n);
    }
    return out;
}

// Try to match a known special token by its printable form. Used to seed the
// stop list so the model can end its turn without us running it past EOS.
static void maybe_add_stop(const llama_vocab * vocab,
                           std::vector<llama_token> & stops,
                           const std::vector<std::string> & names) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_vocab; i++) {
        char buf[64];
        int n = llama_token_to_piece(vocab, (llama_token) i, buf, sizeof(buf), 0, /*special*/ true);
        if (n <= 0) continue;
        std::string tok(buf, n);
        for (const auto & name : names) {
            if (tok == name) {
                stops.push_back((llama_token) i);
                break;
            }
        }
    }
}

// Parse <think>...</think> reasoning blocks out of the model output (Qwen3.5
// style). Mirrors server_tools.py parse_reasoning behavior when thinking is
// disabled: we never produced <think> in the prompt so any leading reasoning
// is unusual, but the 27B Abliterated can still emit it spontaneously.
static void split_reasoning(const std::string & text,
                            std::string & content,
                            std::string & reasoning) {
    const std::string open  = "<think>";
    const std::string close = "</think>";
    auto o = text.find(open);
    auto c = text.find(close);
    if (o != std::string::npos && c != std::string::npos && c > o) {
        reasoning = text.substr(o + open.size(), c - (o + open.size()));
        content   = text.substr(0, o) + text.substr(c + close.size());
        // trim leading whitespace in both
        auto trim = [](std::string & s) {
            size_t i = 0; while (i < s.size() && (s[i] == '\n' || s[i] == ' ' || s[i] == '\t')) i++;
            s.erase(0, i);
        };
        trim(reasoning);
        trim(content);
    } else {
        content = text;
    }
}

// ─── HTTP handlers ────────────────────────────────────────────────

static void handle_models(const httplib::Request &, httplib::Response & res) {
    njson body = {
        {"object", "list"},
        {"data", njson::array({
            {
                {"id", g.alias},
                {"object", "model"},
                {"owned_by", "dflash"},
            }
        })},
    };
    res.set_content(body.dump(), "application/json");
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res) {
    njson body;
    try {
        body = njson::parse(req.body);
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(njson({{"error", {{"message", e.what()}, {"type", "invalid_request"}}}}).dump(),
                        "application/json");
        return;
    }

    // Build chat template inputs.
    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = false;   // default off, matches v4 --reasoning off
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;

    for (const auto & m : body.value("messages", njson::array())) {
        common_chat_msg msg;
        msg.role    = role_str(m);
        msg.content = content_str(m);
        inputs.messages.push_back(msg);
    }

    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const auto & kw = body["chat_template_kwargs"];
        if (kw.contains("enable_thinking")) inputs.enable_thinking = kw["enable_thinking"].get<bool>();
    }

    common_chat_params tmpl_params;
    try {
        tmpl_params = common_chat_templates_apply(g.tmpls.get(), inputs);
    } catch (const std::exception & e) {
        res.status = 500;
        res.set_content(njson({{"error", {{"message", e.what()}, {"type", "template_error"}}}}).dump(),
                        "application/json");
        return;
    }

    // Tokenize prompt.
    std::vector<llama_token> ids = common_tokenize(g.vocab, tmpl_params.prompt, /*add_special*/ true, /*parse_special*/ true);
    std::vector<int32_t> ids32(ids.begin(), ids.end());

    const int max_tokens = body.value("max_tokens", 512);
    if (max_tokens <= 0) {
        res.status = 400;
        res.set_content(njson({{"error", {{"message", "max_tokens must be > 0"}, {"type", "invalid_request"}}}}).dump(),
                        "application/json");
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto out_ids = g.daemon.generate(ids32, max_tokens);
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    std::string full = detokenize_range(g.vocab, out_ids, g.stop_ids);

    // User stop sequences.
    for (const auto & s : body.value("stop", njson::array())) {
        if (!s.is_string()) continue;
        auto p = full.find(s.get<std::string>());
        if (p != std::string::npos) full.resize(p);
    }

    std::string content_s, reasoning_s;
    split_reasoning(full, content_s, reasoning_s);

    njson msg = {{"role", "assistant"}, {"content", content_s}};
    if (!reasoning_s.empty()) msg["reasoning_content"] = reasoning_s;

    njson resp = {
        {"id", std::string("chatcmpl-") + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t0.time_since_epoch()).count())},
        {"object", "chat.completion"},
        {"created", (int64_t) time(nullptr)},
        {"model", g.alias},
        {"choices", njson::array({
            {
                {"index", 0},
                {"message", msg},
                {"finish_reason", "stop"},
            }
        })},
        {"usage", {
            {"prompt_tokens",     (int) ids.size()},
            {"completion_tokens", (int) out_ids.size()},
            {"total_tokens",      (int) (ids.size() + out_ids.size())},
        }},
    };

    fprintf(stderr, "[req] prompt=%zu out=%zu in %.2fs -> %.1f tok/s\n",
            ids.size(), out_ids.size(), dt,
            dt > 0 ? out_ids.size() / dt : 0.0);

    res.set_content(resp.dump(), "application/json");
}

// ─── arg parsing ──────────────────────────────────────────────────

struct cli_args {
    std::string host    = "0.0.0.0";
    int         port    = 8000;
    std::string target;
    std::string draft;
    std::string bin     = "llama-dflash";
    std::string alias   = "dark-opus";
    int         budget  = 22;
    int         max_ctx = 16384;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --target PATH --draft PATH [options]\n"
        "  --target PATH           target GGUF model (required)\n"
        "  --draft PATH            DFlash draft safetensors file or dir (required)\n"
        "  --bin PATH              llama-dflash binary path (default: llama-dflash)\n"
        "  --host HOST             HTTP host (default: 0.0.0.0)\n"
        "  --port N                HTTP port (default: 8000)\n"
        "  --alias NAME            model alias in /v1/models (default: dark-opus)\n"
        "  --budget N              DDTree node budget (default: 22)\n"
        "  --max-ctx N             target max context (default: 16384)\n",
        argv0);
}

static bool parse_args(int argc, char ** argv, cli_args & a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](int i) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return nullptr; }
            return argv[i + 1];
        };
        if (k == "--target" || k == "-m") { auto v = need(i); if (!v) return false; a.target = v; i++; }
        else if (k == "--draft")          { auto v = need(i); if (!v) return false; a.draft = v; i++; }
        else if (k == "--bin")            { auto v = need(i); if (!v) return false; a.bin = v; i++; }
        else if (k == "--host")           { auto v = need(i); if (!v) return false; a.host = v; i++; }
        else if (k == "--port")           { auto v = need(i); if (!v) return false; a.port = std::atoi(v); i++; }
        else if (k == "--alias")          { auto v = need(i); if (!v) return false; a.alias = v; i++; }
        else if (k == "--budget")         { auto v = need(i); if (!v) return false; a.budget = std::atoi(v); i++; }
        else if (k == "--max-ctx")        { auto v = need(i); if (!v) return false; a.max_ctx = std::atoi(v); i++; }
        else if (k == "-h" || k == "--help") { print_usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (a.target.empty() || a.draft.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ─── main ─────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) return 2;

    g.alias = args.alias;

    // Load target only for tokenizer + chat template (no weights on GPU here).
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only         = true;

    fprintf(stderr, "loading vocab from %s ...\n", args.target.c_str());
    g.model = llama_model_load_from_file(args.target.c_str(), mparams);
    if (!g.model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    g.vocab = llama_model_get_vocab(g.model);
    g.tmpls = common_chat_templates_init(g.model, /*chat_template_override=*/ "");

    // Seed stop token list with a few Qwen-family specials.
    maybe_add_stop(g.vocab, g.stop_ids, {"<|im_end|>", "<|endoftext|>"});
    // EOS token from metadata.
    if (llama_vocab_eos(g.vocab) != LLAMA_TOKEN_NULL) g.stop_ids.push_back(llama_vocab_eos(g.vocab));
    std::sort(g.stop_ids.begin(), g.stop_ids.end());
    g.stop_ids.erase(std::unique(g.stop_ids.begin(), g.stop_ids.end()), g.stop_ids.end());
    fprintf(stderr, "stop ids: ");
    for (auto id : g.stop_ids) fprintf(stderr, "%d ", id);
    fprintf(stderr, "\n");

    // Spawn daemon.
    fprintf(stderr, "spawning %s --daemon ...\n", args.bin.c_str());
    if (!g.daemon.start(args.bin, args.target, args.draft, args.budget, args.max_ctx)) {
        fprintf(stderr, "failed to start daemon\n");
        return 1;
    }

    // HTTP server.
    httplib::Server srv;
    srv.set_payload_max_length(16 * 1024 * 1024);
    srv.Get ("/v1/models",             handle_models);
    srv.Get ("/health",                [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    srv.Post("/v1/chat/completions",   handle_chat_completions);

    // Ctrl-C cleanup: kill the daemon child before we exit.
    auto sig_handler = +[](int) {
        g_shutdown.store(true);
        g.daemon.stop();
        llama_model_free(g.model);
        llama_backend_free();
        _exit(0);
    };
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    fprintf(stderr, "llama-dflash-server listening on %s:%d (alias=%s, budget=%d, max_ctx=%d)\n",
            args.host.c_str(), args.port, args.alias.c_str(), args.budget, args.max_ctx);
    srv.listen(args.host, args.port);

    g.daemon.stop();
    llama_model_free(g.model);
    llama_backend_free();
    return 0;
}
