# KpqC-liboqs

This repository is a fork of
👉 https://github.com/minjoo97/liboqs_KpqC,
preserving the original integration of Korean Post-Quantum Cryptography (KpqC) algorithms into liboqs, while adding performance optimizations specifically for the NTRU+ KEM-576 implementation.

The primary objectives of this project are:<br>
•	To enable the use of Korean PQC algorithms within the liboqs framework, and<br>
•	To significantly improve the performance of NTRU+ KEM-576 by removing computational bottlenecks and applying targeted optimizations.

## 📜 License

This project is licensed under the **MIT License**.

As this repository is a fork of [liboqs](https://github.com/open-quantum-safe/liboqs)
(and further derived from https://github.com/minjoo97/liboqs_KpqC),
the original MIT license and copyright notices from the upstream
projects are preserved and included.

Please see `LICENSE.txt` for full details.



---

## 🚀 What’s Improved (NTRU+ 576 Only)


**1. NEON SIMD Vectorization**

   •	Implementation Path: src/kem/ntru_plus/KpqClean_ver2_NTRU_PLUS_KEM576_neon<br>
   •	Applies 128-bit NEON SIMD parallelization to operations such as poly_cbd, NTT, and poly_baseinv<br>
   •	Uses vld4q-based de-interleaving to process 4-coefficient blocks in an 8-way parallel manner<br>


**2. Montgomery Batch Inversion**

   •	Implementation Path:
   src/kem/ntru_plus/KpqClean_ver2_NTRU_PLUS_KEM576_clean_montgomery-batch-normalization<br>
   •	Reduces the number of expensive fqinv operations in KeyGen from 144 calls to a single call<br>
   •	Implements batch inversion using determinant accumulation → single inversion → reverse reconstruction


**3. Function Inlining**
   •	Implementation Path:
   src/kem/ntru_plus/KpqClean_ver2_NTRU_PLUS_KEM576_clean_montgomery-batch-normalization<br>
   •	Converts montgomery_reduce and barrett_reduce into static inline functions<br>
   •	Eliminates CALL/RET overhead for frequently used modular arithmetic routines
   
---

## 🙏 Acknowledgements

This project is a fork of the Open Quantum Safe (OQS) project’s `liboqs`. We thank all OQS contributors for their foundational work.

The OQS project is supported by the [Post-Quantum Cryptography Alliance (PQCA)](https://pqca.org/) under the Linux Foundation.
