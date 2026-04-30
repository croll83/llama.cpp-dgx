"""Pure-PyTorch reference implementations for the GDN chunk kernels.

Generates random inputs with a fixed seed, computes the expected outputs
via a naive (slow but obviously-correct) algorithm, and serialises both
to a binary fixture file consumed by the CUDA test harness
(`tests/test_gdn_chunk.cu`).

Layouts mirror the CUDA kernels exactly:
  - g, beta, g_cumsum:      [B, T, H]              row-major fp32
  - K:                       [B, T, H, DK]          row-major bf16
  - V:                       [B, T, H, DV]          row-major bf16
  - A_sol:                   [B, T, H, S]           row-major bf16
  - h_initial / h_final:     [B, H, DK, DV]         row-major bf16
  - h_per_chunk:             [B, num_chunks, H, DK, DV] row-major bf16

Conventions:
  - chunk-local cumsum: each chunk's cumsum starts at 0 (no inter-chunk
    carry). chunk_size must divide T or be padded; we test with T=128
    (matches H=4, S=64, two chunks).
  - KKT solve: solves (I - L) X = diag(beta) per chunk where
        L[i, j] = beta[i] * <K[i], K[j]> * exp(g_cum[j-1] - g_cum[i])  for j < i
        L[i, j] = 0                                                     for j >= i
        (g_cum[-1] := 0)
    Output X is strict-lower-tri-with-diag, also in [B, T, H, S] layout.
  - prepare_h: SSM state advance per (B, H), iterating chunks sequentially:
        for each chunk c:
            h_per_chunk[c] = h            (state at the START of chunk c)
            for each token t in chunk:
                u_t  = beta_t * (v_t - K_t @ h)        # [DV]
                h    = exp(g_cum[t] - g_cum[t-1]) * h  # decay
                h    = h + outer(K_t, u_t)             # rank-1 update
        h_final = h
    g_cum[t-1] is taken within the chunk (chunk-local), so for t=0 inside
    each chunk we use g_prev = 0.
"""

import os
import struct
import sys

import torch


def chunk_local_cumsum_ref(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """g: [B, T, H] fp32. Returns [B, T, H] = chunk-local inclusive cumsum."""
    B, T, H = g.shape
    out = torch.zeros_like(g)
    num_chunks = (T + chunk_size - 1) // chunk_size
    for c in range(num_chunks):
        a = c * chunk_size
        b = min(a + chunk_size, T)
        out[:, a:b, :] = torch.cumsum(g[:, a:b, :], dim=1)
    return out


def kkt_solve_ref(K: torch.Tensor,
                  beta: torch.Tensor,
                  g_cumsum: torch.Tensor,
                  chunk_size: int) -> torch.Tensor:
    """
    K: [B, T, H, DK] (any dtype)
    beta, g_cumsum: [B, T, H] fp32
    Returns A_sol: [B, T, H, S=chunk_size] fp32 (caller can cast to bf16).
    """
    B, T, H, DK = K.shape
    S = chunk_size
    K_f = K.float()
    A_sol = torch.zeros((B, T, H, S), dtype=torch.float32, device=K.device)

    num_chunks = (T + S - 1) // S
    for c in range(num_chunks):
        a = c * S
        b = min(a + S, T)
        S_eff = b - a

        K_chunk    = K_f[:, a:b, :, :]
        beta_chunk = beta[:, a:b, :]
        g_chunk    = g_cumsum[:, a:b, :]

        KK = torch.einsum('bihd,bkhd->bhik', K_chunk, K_chunk)

        g_perm = g_chunk.permute(0, 2, 1)
        g_prev = torch.zeros_like(g_perm)
        g_prev[..., 1:] = g_perm[..., :-1]
        g_diff = g_prev.unsqueeze(-2) - g_perm.unsqueeze(-1)
        beta_perm = beta_chunk.permute(0, 2, 1).unsqueeze(-1)
        L_full = beta_perm * KK * torch.exp(g_diff)
        mask = torch.tril(torch.ones(S_eff, S_eff, device=K.device), diagonal=-1)
        L = L_full * mask

        X = torch.zeros((B, H, S_eff, S_eff), dtype=torch.float32, device=K.device)
        diag_beta = beta_chunk.permute(0, 2, 1)
        for i in range(S_eff):
            X[..., i, i] = diag_beta[..., i]
            if i > 0:
                X[..., i, :i] = (L[..., i:i+1, :i] @ X[..., :i, :i]).squeeze(-2)

        X_perm = X.permute(0, 2, 1, 3)
        A_sol[:, a:b, :, :S_eff] = X_perm

    return A_sol


def prepare_h_ref(K: torch.Tensor,
                   V: torch.Tensor,
                   g_cumsum: torch.Tensor,
                   beta: torch.Tensor,
                   h_initial: torch.Tensor,
                   chunk_size: int):
    """
    K: [B, T, H, DK] (bf16 or fp32)
    V: [B, T, H, DV] (bf16 or fp32)
    g_cumsum, beta: [B, T, H] fp32
    h_initial: [B, H, DK, DV] (bf16 or fp32)
    Returns:
        h_per_chunk: [B, num_chunks, H, DK, DV] fp32  (state at chunk START)
        h_final:     [B, H, DK, DV] fp32
    Math (per (B, H)):
        h = h_initial
        for c in chunks:
            h_per_chunk[c] = h.clone()
            for t in chunk:
                v_t   = V[t]
                k_t   = K[t]
                u_t   = beta_t * (v_t - k_t @ h)         # [DV]
                alpha = exp(g_cum[t] - g_cum[t-1])       # chunk-local g_prev=0 at t=0
                h     = alpha * h + outer(k_t, u_t)
        h_final = h
    """
    B, T, H, DK = K.shape
    DV = V.shape[-1]
    S = chunk_size
    num_chunks = (T + S - 1) // S
    K_f = K.float()
    V_f = V.float()
    h_per_chunk = torch.zeros((B, num_chunks, H, DK, DV), dtype=torch.float32, device=K.device)
    h = h_initial.float().clone()  # [B, H, DK, DV]
    for c in range(num_chunks):
        a = c * S
        b = min(a + S, T)
        h_per_chunk[:, c] = h
        for t_local in range(b - a):
            t_g = a + t_local
            k_t = K_f[:, t_g, :, :]                          # [B, H, DK]
            v_t = V_f[:, t_g, :, :]                          # [B, H, DV]
            beta_t = beta[:, t_g, :]                         # [B, H]
            g_t    = g_cumsum[:, t_g, :]                     # [B, H]
            if t_local == 0:
                g_prev = torch.zeros_like(g_t)               # chunk-local cumsum starts at 0
            else:
                g_prev = g_cumsum[:, t_g - 1, :]
            alpha = torch.exp(g_t - g_prev)                  # [B, H]

            # k_t @ h where h is [B, H, DK, DV] → [B, H, DV]
            kh = torch.einsum('bhd,bhde->bhe', k_t, h)
            u_t = beta_t.unsqueeze(-1) * (v_t - kh)          # [B, H, DV]
            # Decay h
            h = alpha.unsqueeze(-1).unsqueeze(-1) * h
            # Rank-1 update
            h = h + torch.einsum('bhd,bhe->bhde', k_t, u_t)
    return h_per_chunk, h


def fused_fwd_ref(Q: torch.Tensor,
                  K: torch.Tensor,
                  V: torch.Tensor,
                  A_sol: torch.Tensor,
                  g_cumsum: torch.Tensor,
                  h_per_chunk: torch.Tensor,
                  chunk_size: int) -> torch.Tensor:
    """Compute the GDN output O.

    Args:
      Q:           [B, T, H, DK] (any dtype, typically bf16)
      K:           [B, T, H, DK]
      V:           [B, T, H, DV]
      A_sol:       [B, T, H, S]   strict-lower-tri-with-diag (rows of A in chunk)
      g_cumsum:    [B, T, H] fp32 chunk-local
      h_per_chunk: [B, num_chunks, H, DK, DV] fp32 (or bf16) — state at chunk START
    Returns:
      O:           [B, T, H, DV] fp32 (caller can cast to bf16)
    """
    B, T, H, DK = Q.shape
    DV = V.shape[-1]
    S = chunk_size
    num_chunks = (T + S - 1) // S
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    A_f = A_sol.float()
    h_f = h_per_chunk.float()
    O = torch.zeros((B, T, H, DV), dtype=torch.float32, device=Q.device)

    for c in range(num_chunks):
        a = c * S
        b = min(a + S, T)
        S_eff = b - a

        Q_chunk  = Q_f[:, a:b]                          # [B, S, H, DK]
        K_chunk  = K_f[:, a:b]
        V_chunk  = V_f[:, a:b]
        A_chunk  = A_f[:, a:b, :, :S_eff]               # [B, S, H, S_eff]
        gc_chunk = g_cumsum[:, a:b]                     # [B, S, H]
        h_c      = h_f[:, c]                            # [B, H, DK, DV]

        # V_eff[t, d] = V[t, d] - (K[t] @ h_c)[d]
        Kh    = torch.einsum('bthd,bhde->bthe', K_chunk, h_c)   # [B, S, H, DV]
        V_eff = V_chunk - Kh                                     # [B, S, H, DV]

        # U[i, d] = sum_j A[i, j] * V_eff[j, d]   (per (B, H))
        # A_chunk: [B, S(i), H, S(j)] → permute to [B, H, S, S]
        A_perm = A_chunk.permute(0, 2, 1, 3)            # [B, H, S, S]
        V_perm = V_eff.permute(0, 2, 1, 3)              # [B, H, S, DV]
        U_perm = A_perm @ V_perm                        # [B, H, S, DV]

        # Intra-chunk attention:
        #   attn[t, k] = (q_t · k_k) * exp(g_cum[t] - g_cum[k])  for k <= t else 0
        QK = torch.einsum('bthd,bkhd->bhtk', Q_chunk, K_chunk)   # [B, H, S, S]
        gc_perm = gc_chunk.permute(0, 2, 1)                       # [B, H, S]
        decay = torch.exp(gc_perm.unsqueeze(-1) - gc_perm.unsqueeze(-2))  # [B, H, S(t), S(k)]
        mask = torch.tril(torch.ones(S_eff, S_eff, device=Q.device))      # k <= t
        attn = QK * decay * mask                                            # [B, H, S, S]

        O_intra = attn @ U_perm                                             # [B, H, S, DV]

        # Inter-chunk: o_inter[t, d] = exp(gc[t]) * (q_t @ h_c)[d]
        Qh = torch.einsum('bthd,bhde->bthe', Q_chunk, h_c)                   # [B, S, H, DV]
        Qh_perm = Qh.permute(0, 2, 1, 3)                                      # [B, H, S, DV]
        O_inter = torch.exp(gc_perm).unsqueeze(-1) * Qh_perm                  # [B, H, S, DV]

        O_chunk = (O_intra + O_inter).permute(0, 2, 1, 3)                    # [B, S, H, DV]
        O[:, a:b] = O_chunk
    return O


def write_fixture(path: str,
                  B: int, T: int, H: int, DK: int, DV: int, S: int,
                  g: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  beta: torch.Tensor, g_cumsum: torch.Tensor,
                  A_sol: torch.Tensor,
                  h_initial: torch.Tensor,
                  h_per_chunk: torch.Tensor,
                  h_final: torch.Tensor,
                  O: torch.Tensor) -> None:
    """Binary layout consumed by tests/test_gdn_chunk.cu:

        6-byte ints (LE):  B  T  H  DK  DV  S
        float32 array (B*T*H):                       g
        bfloat16 array (B*T*H*DK), as uint16 LE:    Q
        bfloat16 array (B*T*H*DK), as uint16 LE:    K
        bfloat16 array (B*T*H*DV), as uint16 LE:    V
        float32 array (B*T*H):                       beta
        float32 array (B*T*H):                       g_cumsum_expected
        bfloat16 array (B*T*H*S), as uint16 LE:     A_sol_expected
        bfloat16 array (B*H*DK*DV), as uint16 LE:   h_initial
        bfloat16 array (B*nc*H*DK*DV), as uint16:   h_per_chunk_expected
        bfloat16 array (B*H*DK*DV), as uint16 LE:   h_final_expected
        bfloat16 array (B*T*H*DV), as uint16 LE:    O_expected
    """
    num_chunks = (T + S - 1) // S
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", B, T, H, DK, DV, S))
        f.write(g.contiguous().cpu().numpy().astype("<f4").tobytes())

        Q_u16 = Q.contiguous().cpu().view(torch.uint16).numpy()
        f.write(Q_u16.astype("<u2").tobytes())

        K_u16 = K.contiguous().cpu().view(torch.uint16).numpy()
        f.write(K_u16.astype("<u2").tobytes())

        V_u16 = V.contiguous().cpu().view(torch.uint16).numpy()
        f.write(V_u16.astype("<u2").tobytes())

        f.write(beta.contiguous().cpu().numpy().astype("<f4").tobytes())
        f.write(g_cumsum.contiguous().cpu().numpy().astype("<f4").tobytes())

        A_sol_u16 = A_sol.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
        f.write(A_sol_u16.astype("<u2").tobytes())

        h0_u16 = h_initial.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
        f.write(h0_u16.astype("<u2").tobytes())

        hpc_u16 = h_per_chunk.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
        f.write(hpc_u16.astype("<u2").tobytes())

        hf_u16 = h_final.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
        f.write(hf_u16.astype("<u2").tobytes())

        O_u16 = O.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
        f.write(O_u16.astype("<u2").tobytes())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdn_chunk_fixture.bin"

    torch.manual_seed(20260430)
    B, T, H, DK, DV = 2, 128, 4, 128, 128
    S = 64

    g = torch.randn(B, T, H, dtype=torch.float32) * 0.1

    # Q (not L2-normalised in our convention; raw transformation output).
    Q = (torch.randn(B, T, H, DK, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    # K must be L2-normalised so KKT solve doesn't blow up (matches the
    # GDN wrapper's behaviour in production).
    K_raw = torch.randn(B, T, H, DK, dtype=torch.float32)
    K     = (K_raw / (K_raw.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)

    # V is not normalised in production.
    V = (torch.randn(B, T, H, DV, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    beta  = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32))

    # h_initial: zero for the first test (most realistic for a fresh prefill).
    h_initial = torch.zeros((B, H, DK, DV), dtype=torch.bfloat16)

    g_cumsum = chunk_local_cumsum_ref(g, S)
    A_sol    = kkt_solve_ref(K, beta, g_cumsum, S)
    h_per_chunk, h_final = prepare_h_ref(K, V, g_cumsum, beta, h_initial, S)
    O = fused_fwd_ref(Q, K, V, A_sol, g_cumsum, h_per_chunk, S)

    write_fixture(out_path, B, T, H, DK, DV, S,
                  g, Q, K, V, beta, g_cumsum, A_sol,
                  h_initial, h_per_chunk, h_final, O)

    print(f"wrote {out_path}: B={B} T={T} H={H} DK={DK} DV={DV} S={S}")
    print(f"  g_cumsum stats:    min={g_cumsum.min():.4f} max={g_cumsum.max():.4f} mean={g_cumsum.mean():.4f}")
    print(f"  A_sol stats:       min={A_sol.min():.4f} max={A_sol.max():.4f} abs.mean={A_sol.abs().mean():.4f}")
    print(f"  h_per_chunk stats: min={h_per_chunk.min():.4f} max={h_per_chunk.max():.4f} abs.mean={h_per_chunk.abs().mean():.4f}")
    print(f"  h_final stats:     min={h_final.min():.4f} max={h_final.max():.4f} abs.mean={h_final.abs().mean():.4f}")
    print(f"  O stats:           min={O.min():.4f} max={O.max():.4f} abs.mean={O.abs().mean():.4f}")


if __name__ == "__main__":
    main()
