// Loads a DFlash draft model from a GGUF file (community quants like
// spiritbuun/Qwen3.6-27B-DFlash-GGUF). Mirrors load_draft_safetensors but
// uses llama.cpp's GGUF API. Tensor types come straight from the file
// (typically Q8_0 for projections, F32 for norms) — ggml_mul_mat handles
// Q8_0 src0 natively, so no conversion is needed.
//
// GGUF tensor name → DraftWeights field:
//   dflash_fc.weight                       → fc                 [25600, 5120] Q8_0
//   dflash_hidden_norm.weight              → hidden_norm        [5120]        F32
//   output_norm.weight                     → out_norm           [5120]        F32
//   blk.<i>.attn_norm.weight               → layers[i].attn_norm
//   blk.<i>.post_attention_norm.weight     → layers[i].ffn_norm
//   blk.<i>.attn_q.weight                  → layers[i].wq
//   blk.<i>.attn_k.weight                  → layers[i].wk
//   blk.<i>.attn_v.weight                  → layers[i].wv
//   blk.<i>.attn_output.weight             → layers[i].wo
//   blk.<i>.attn_q_norm.weight             → layers[i].q_norm
//   blk.<i>.attn_k_norm.weight             → layers[i].k_norm
//   blk.<i>.ffn_gate.weight                → layers[i].w_gate
//   blk.<i>.ffn_up.weight                  → layers[i].w_up
//   blk.<i>.ffn_down.weight                → layers[i].w_down
//
// SWA hyperparameters read from GGUF KV:
//   dflash-draft.attention.sliding_window           (uint32)
//   dflash-draft.attention.sliding_window_pattern   ([bool])

#include "internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd    = -1;
#endif
    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) { err = "CreateFileA failed"; return false; }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) { err = "GetFileSizeEx failed"; return false; }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { err = "CreateFileMappingA failed"; return false; }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { err = "MapViewOfFile failed"; return false; }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open failed: " + std::string(std::strerror(errno)); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat failed"; return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap failed"; addr = nullptr; return false; }
#endif
        return true;
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Look up a tensor in the meta_ctx and bind it to a field. Returns false
// (and sets last_error) when the GGUF file is missing the named tensor.
bool bind_tensor(ggml_context * ctx,
                 const char * name,
                 ggml_tensor ** out_field) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        set_last_error(std::string("GGUF: missing tensor '") + name + "'");
        return false;
    }
    *out_field = t;
    return true;
}

} // namespace

bool load_draft_gguf(const std::string & path,
                     ggml_backend_t       backend,
                     DraftWeights &       out) {
    // ── 1. Open with GGUF API (no_alloc=true → tensors created in meta_ctx
    //       with their declared type/shape; data not yet uploaded). ─────────
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;

    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx || !meta_ctx) {
        set_last_error("GGUF: gguf_init_from_file failed: " + path);
        if (gctx) gguf_free(gctx);
        return false;
    }

    // ── 2. Validate architecture key (defensive — graph hardcodes 5 layers
    //       × Qwen3.6 hidden=5120). ───────────────────────────────────────
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("GGUF: missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "dflash-draft") {
            set_last_error(std::string("GGUF: unexpected arch '") + arch + "' (expected dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // ── 3. Reuse the gguf-owned meta_ctx as our DraftWeights ctx. The
    //       tensors gguf created already have correct types/shapes; we just
    //       grab pointers by name and let ggml_backend_alloc_ctx_tensors
    //       upload them. ─────────────────────────────────────────────────
    const int n_layers = DFLASH27B_DRAFT_LAYERS;
    out.ctx = meta_ctx;
    out.backend = backend;
    out.layers.assign(n_layers, DraftLayer{});

    bool ok = true;
    ok &= bind_tensor(out.ctx, "dflash_fc.weight",          &out.fc);
    ok &= bind_tensor(out.ctx, "dflash_hidden_norm.weight", &out.hidden_norm);
    ok &= bind_tensor(out.ctx, "output_norm.weight",        &out.out_norm);
    if (!ok) { gguf_free(gctx); return false; }

    for (int il = 0; il < n_layers; il++) {
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "blk.%d.", il);
        std::string p = pfx;
        DraftLayer & L = out.layers[il];
        ok &= bind_tensor(out.ctx, (p + "attn_norm.weight"          ).c_str(), &L.attn_norm);
        ok &= bind_tensor(out.ctx, (p + "post_attention_norm.weight").c_str(), &L.ffn_norm);
        ok &= bind_tensor(out.ctx, (p + "attn_q.weight"             ).c_str(), &L.wq);
        ok &= bind_tensor(out.ctx, (p + "attn_k.weight"             ).c_str(), &L.wk);
        ok &= bind_tensor(out.ctx, (p + "attn_v.weight"             ).c_str(), &L.wv);
        ok &= bind_tensor(out.ctx, (p + "attn_output.weight"        ).c_str(), &L.wo);
        ok &= bind_tensor(out.ctx, (p + "attn_q_norm.weight"        ).c_str(), &L.q_norm);
        ok &= bind_tensor(out.ctx, (p + "attn_k_norm.weight"        ).c_str(), &L.k_norm);
        ok &= bind_tensor(out.ctx, (p + "ffn_gate.weight"           ).c_str(), &L.w_gate);
        ok &= bind_tensor(out.ctx, (p + "ffn_up.weight"             ).c_str(), &L.w_up);
        ok &= bind_tensor(out.ctx, (p + "ffn_down.weight"           ).c_str(), &L.w_down);
        if (!ok) { gguf_free(gctx); return false; }
    }

    // ── 4. SWA metadata: read sliding_window + per-layer pattern from KV. ──
    out.swa_window = 0;
    out.layer_is_swa.clear();
    {
        int64_t swid = gguf_find_key(gctx, "dflash-draft.attention.sliding_window");
        if (swid >= 0) {
            out.swa_window = (int) gguf_get_val_u32(gctx, swid);
        }
        int64_t pid = gguf_find_key(gctx, "dflash-draft.attention.sliding_window_pattern");
        if (pid >= 0 && out.swa_window > 0) {
            const size_t n = gguf_get_arr_n(gctx, pid);
            const bool * arr = (const bool *) gguf_get_arr_data(gctx, pid);
            out.layer_is_swa.reserve(n);
            for (size_t i = 0; i < n; i++) out.layer_is_swa.push_back(arr[i]);
        }
        // Fallback: if pattern array is absent but window > 0, treat all SWA.
        if (out.swa_window > 0 && out.layer_is_swa.empty()) {
            out.layer_is_swa.assign(out.layers.size(), true);
        }
    }

    // ── 5. Allocate backend buffer + upload tensor bytes from the file. ──
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("GGUF: ggml_backend_alloc_ctx_tensors failed (draft)");
        gguf_free(gctx);
        return false;
    }

    Mmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) {
        set_last_error("GGUF: " + err);
        gguf_free(gctx);
        return false;
    }
    const size_t  data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gctx);
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(out.ctx, tname);
        if (!t) continue;  // gguf created a tensor we don't bind (shouldn't happen, defensive)
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("GGUF: tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *) mm.addr + off, 0, sz);
    }

    gguf_free(gctx);

    std::fprintf(stderr,
        "draft (gguf): swa_window = %d (0 = non-causal), layer_pattern_size = %zu\n",
        out.swa_window, out.layer_is_swa.size());
    return true;
}

} // namespace dflash27b
