# Performance Log

| Date | Scenario | Optimization(s) | Iter/s | Notes |
|------|----------|-----------------|--------|-------|
| Earlier | MNIST (~2k samples), batch=32, hidden=128 | Baseline (pre-optimisation) | ~12 | Provided by user before enabling any compiler flags. |
| Earlier | MNIST (~2k samples), batch=32, hidden=128 | `-O3 -march=native -ffast-math` | ~70 | First improvement after turning on compiler optimisations. |
| Today | MNIST (~2k samples), batch=32, hidden=128 | `-O3` flags + buffer reuse | ~390 | After dense/activation layers reuse their internal buffers. |
| Today | MNIST (~2k samples), batch=32, hidden=128 | `-O3` + buffer reuse + OpenBLAS (threads=1) | ~1,200 | Built with `make USE_BLAS=1` after installing `libopenblas-dev`; throughput measured via `iters/s` in training logs. |
| Today | XOR (4 samples), batch=1, hidden=4, seed=1 | `-O3` flags + buffer reuse | 1,431,229 | `./build/nnc --dataset xor --epochs 5000 --log-steps 1000 --seed 1` (no validation). |

Add new rows whenever we tweak performance-critical code. Aim to keep scenarios comparable (same dataset, batch size, and hardware).
