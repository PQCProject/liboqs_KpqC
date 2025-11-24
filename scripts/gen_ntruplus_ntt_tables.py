#!/usr/bin/env python3
"""Generate Good-Thomas NTT tables for NTRU+ (64×9) and verify NTT equivalence."""

import math
import random
from typing import List, Tuple

Q = 3457
N = 576
N1 = 64
N2 = 9
ROOT = 7  # primitive root mod Q

if math.gcd(N1, N2) != 1:
    raise SystemExit("N must factor into coprime components")

OMEGA = pow(ROOT, (Q - 1) // N, Q)
INV_N1 = pow(N1, -1, N2)
INV_N2 = pow(N2, -1, N1)

W1_EXP = (N2 * INV_N2) % N  # produces primitive N1-th root
W2_EXP = (N1 * INV_N1) % N  # produces primitive N2-th root
OMEGA1 = pow(OMEGA, W1_EXP, Q)
OMEGA2 = pow(OMEGA, W2_EXP, Q)

R = 1 << 16
QINV = 12929  # q^{-1} mod 2^16

def montgomery_reduce(a: int) -> int:
    """Montgomery reduction matching C implementation."""
    t = (a & 0xFFFF) * QINV & 0xFFFF
    return (a - t * Q) >> 16

def fqmul(a: int, b: int) -> int:
    return montgomery_reduce(a * b)

def barrett_reduce(a: int) -> int:
    v = ((1 << 26) + Q // 2) // Q
    t = ((v * a + (1 << 25)) >> 26) * Q
    return a - t

def to_montgomery(x: int) -> int:
    return (x % Q) * R % Q

def from_montgomery(x: int) -> int:
    return montgomery_reduce(x)

def crt_index(i1: int, i2: int) -> int:
    """Index satisfying idx ≡ i1 (mod N1) and idx ≡ i2 (mod N2)."""
    return (i1 + N1 * ((i2 - i1) * INV_N1 % N2)) % N

def idx_to_pair(idx: int) -> Tuple[int, int]:
    return idx % N1, idx % N2

def ntt_reference(a: List[int]) -> List[int]:
    res = [0] * N
    for k in range(N):
        acc = 0
        omega_k = pow(OMEGA, k, Q)
        for j in range(N):
            acc = (acc + a[j] * pow(omega_k, j, Q)) % Q
        res[k] = acc
    return res

def cooley_radix2_ntt(a: List[int], root: int) -> List[int]:
    n = len(a)
    res = list(a)
    len_ = n // 2
    k = 1
    while len_ >= 1:
        step = len_ * 2
        for start in range(0, n, step):
            zeta = pow(root, k, Q)
            k += 1
            for j in range(start, start + len_):
                t = (res[j + len_] * zeta) % Q
                res[j + len_] = (res[j] - t) % Q
                res[j] = (res[j] + t) % Q
        len_ //= 2
    return res

def naive_dft(a: List[int], root: int) -> List[int]:
    n = len(a)
    res = [0] * n
    for k in range(n):
        acc = 0
        for j in range(n):
            acc = (acc + a[j] * pow(root, (j * k) % n, Q)) % Q
        res[k] = acc
    return res

def good_thomas_ntt(a: List[int]) -> List[int]:
    mat = [[0] * N2 for _ in range(N1)]
    for idx, val in enumerate(a):
        i1, i2 = idx_to_pair(idx)
        mat[i1][i2] = val
    # length-64 transforms per column
    for i2 in range(N2):
        col = [mat[i][i2] for i in range(N1)]
        col = naive_dft(col, OMEGA1)
        for i1 in range(N1):
            mat[i1][i2] = col[i1]
    # length-9 transforms per row
    for i1 in range(N1):
        row = naive_dft(mat[i1], OMEGA2)
        mat[i1] = row
    # combine back to 1-D vector
    res = [0] * N
    for idx in range(N):
        i1, i2 = idx_to_pair(idx)
        res[idx] = mat[i1][i2]
    return res

def gen_tables() -> Tuple[List[int], List[int], List[int]]:
    # TODO: implement actual table generation
    return [], [], []

def main():
    tests = 5
    for i in range(tests):
        vec = [random.randrange(Q) for _ in range(N)]
        ref = ntt_reference(vec)
        cand = good_thomas_ntt(vec)
        if any((ref[i] - cand[i]) % Q for i in range(N)):
            print("Mismatch detected at test", i)
            return
    print("Good-Thomas reference matches for", tests, "random inputs")

if __name__ == "__main__":
    main()
