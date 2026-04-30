"""Pure-PyTorch reference implementations for the GDN chunk kernels.

Generates random inputs with a fixed seed, computes the expected outputs
via a naive (slow but obviously-correct) algorithm, and serialises both
to a binary fixture file consumed by the CUDA test harness
(`tests/test-gdn-chunk.cu`).

Layouts mirror the CUDA kernels exactly. See test-gdn-chunk.cu for the
binary layout (header order is critical).

Usage:
  python3 gdn_chunk_ref.py [out.bin] [--shape B,T,H,DK,DV,S]

Default shape is the small unit-test shape (B=2 T=128 H=4 DK=128 DV=128 S=64).
For production-size profiling fixture pass --shape 1,192,16,128,128,64 (or
similar — must match an instantiated kernel template).
"""

import argparse
import os
import struct
import sys

import torch


def chunk_local_cumsum_ref(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H = g.shape
    out = torch.zeros_like(g)
    num_chunks = (T + chunk_size - 1) // chunk_size
    for c in range(num_chunks):
        a = c * chunk_size
        b = min(a + chunk_size, T)
        out[:, a:b, :] = torch.cumsum(g[:, a:b, :], dim=1)
    return out


def kkt_solve_ref(K, beta, g_cumsum, chunk_size):
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


def prepare_h_ref(K, V, g_cumsum, beta, h_initial, chunk_size):
    B, T, H, DK = K.shape
    DV = V.shape[-1]
    S = chunk_size
    num_chunks = (T + S - 1) // S
    K_f = K.float(); V_f = V.float()
    h_per_chunk = torch.zeros((B, num_chunks, H, DK, DV), dtype=torch.float32, device=K.device)
    h = h_initial.float().clone()
    for c in range(num_chunks):
        a = c * S
        b = min(a + S, T)
        h_per_chunk[:, c] = h
        for t_local in range(b - a):
            t_g = a + t_local
            k_t = K_f[:, t_g, :, :]
            v_t = V_f[:, t_g, :, :]
            beta_t = beta[:, t_g, :]
            g_t    = g_cumsum[:, t_g, :]
            if t_local == 0:
                g_prev = torch.zeros_like(g_t)
            else:
                g_prev = g_cumsum[:, t_g - 1, :]
            alpha = torch.exp(g_t - g_prev)
            kh = torch.einsum('bhd,bhde->bhe', k_t, h)
            u_t = beta_t.unsqueeze(-1) * (v_t - kh)
            h = alpha.unsqueeze(-1).unsqueeze(-1) * h
            h = h + torch.einsum('bhd,bhe->bhde', k_t, u_t)
    return h_per_chunk, h


def fused_fwd_ref(Q, K, V, A_sol, g_cumsum, h_per_chunk, chunk_size):
    B, T, H, DK = Q.shape
    DV = V.shape[-1]
    S = chunk_size
    num_chunks = (T + S - 1) // S
    Q_f = Q.float(); K_f = K.float(); V_f = V.float()
    A_f = A_sol.float(); h_f = h_per_chunk.float()
    O = torch.zeros((B, T, H, DV), dtype=torch.float32, device=Q.device)
    for c in range(num_chunks):
        a = c * S
        b = min(a + S, T)
        S_eff = b - a
        Q_chunk  = Q_f[:, a:b]; K_chunk  = K_f[:, a:b]; V_chunk  = V_f[:, a:b]
        A_chunk  = A_f[:, a:b, :, :S_eff]
        gc_chunk = g_cumsum[:, a:b]
        h_c      = h_f[:, c]
        Kh    = torch.einsum('bthd,bhde->bthe', K_chunk, h_c)
        V_eff = V_chunk - Kh
        A_perm = A_chunk.permute(0, 2, 1, 3)
        V_perm = V_eff.permute(0, 2, 1, 3)
        U_perm = A_perm @ V_perm
        QK = torch.einsum('bthd,bkhd->bhtk', Q_chunk, K_chunk)
        gc_perm = gc_chunk.permute(0, 2, 1)
        decay = torch.exp(gc_perm.unsqueeze(-1) - gc_perm.unsqueeze(-2))
        mask = torch.tril(torch.ones(S_eff, S_eff, device=Q.device))
        attn = QK * decay * mask
        O_intra = attn @ U_perm
        Qh = torch.einsum('bthd,bhde->bthe', Q_chunk, h_c)
        Qh_perm = Qh.permute(0, 2, 1, 3)
        O_inter = torch.exp(gc_perm).unsqueeze(-1) * Qh_perm
        O_chunk = (O_intra + O_inter).permute(0, 2, 1, 3)
        O[:, a:b] = O_chunk
    return O


def write_fixture(path, B, T, H, DK, DV, S,
                  g, Q, K, V, beta, g_cumsum, A_sol,
                  h_initial, h_per_chunk, h_final, O):
    num_chunks = (T + S - 1) // S
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", B, T, H, DK, DV, S))
        f.write(g.contiguous().cpu().numpy().astype("<f4").tobytes())
        for arr in (Q, K, V):
            u16 = arr.contiguous().cpu().view(torch.uint16).numpy()
            f.write(u16.astype("<u2").tobytes())
        f.write(beta.contiguous().cpu().numpy().astype("<f4").tobytes())
        f.write(g_cumsum.contiguous().cpu().numpy().astype("<f4").tobytes())
        for arr in (A_sol.to(torch.bfloat16),
                    h_initial.to(torch.bfloat16),
                    h_per_chunk.to(torch.bfloat16),
                    h_final.to(torch.bfloat16),
                    O.to(torch.bfloat16)):
            u16 = arr.contiguous().cpu().view(torch.uint16).numpy()
            f.write(u16.astype("<u2").tobytes())


def parse_shape(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError("--shape needs 6 ints: B,T,H,DK,DV,S")
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path", nargs="?", default="/tmp/gdn_chunk_fixture.bin")
    ap.add_argument("--shape", default="2,128,4,128,128,64",
                     help="B,T,H,DK,DV,S (default unit-test shape)")
    ap.add_argument("--seed", type=int, default=20260430)
    args = ap.parse_args()

    B, T, H, DK, DV, S = parse_shape(args.shape)
    print(f"shape: B={B} T={T} H={H} DK={DK} DV={DV} S={S}  → {args.out_path}")

    torch.manual_seed(args.seed)
    g = torch.randn(B, T, H, dtype=torch.float32) * 0.1
    Q = (torch.randn(B, T, H, DK, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    K_raw = torch.randn(B, T, H, DK, dtype=torch.float32)
    K     = (K_raw / (K_raw.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    V = (torch.randn(B, T, H, DV, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32))
    h_initial = torch.zeros((B, H, DK, DV), dtype=torch.bfloat16)

    g_cumsum = chunk_local_cumsum_ref(g, S)
    A_sol    = kkt_solve_ref(K, beta, g_cumsum, S)
    h_per_chunk, h_final = prepare_h_ref(K, V, g_cumsum, beta, h_initial, S)
    O = fused_fwd_ref(Q, K, V, A_sol, g_cumsum, h_per_chunk, S)

    write_fixture(args.out_path, B, T, H, DK, DV, S,
                  g, Q, K, V, beta, g_cumsum, A_sol,
                  h_initial, h_per_chunk, h_final, O)
    print(f"  g_cumsum:    min={g_cumsum.min():.4f} max={g_cumsum.max():.4f}")
    print(f"  A_sol:       abs.mean={A_sol.abs().mean():.4f}")
    print(f"  h_per_chunk: abs.mean={h_per_chunk.abs().mean():.4f}")
    print(f"  h_final:     abs.mean={h_final.abs().mean():.4f}")
    print(f"  O:           abs.mean={O.abs().mean():.4f}")


if __name__ == "__main__":
    main()
