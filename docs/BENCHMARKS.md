# Benchmarks and profiling

This file holds the detail behind the Benchmarks section of the [README](../README.md): the profile of the 0.2.1 release on an Intel i7-8665U, and a check on a second machine. The criterion tables are in the README.

## Profiling

Everything in this section was measured with Linux `perf` on the i7-8665U (4 cores, 8 threads, 8 MB L3, a 15 W laptop chip) for 0.2.1, and the 0.2.1 entry in [CHANGELOG.md](../CHANGELOG.md) records its headline numbers. The profile has not been re-run since.

### Commands

`examples/profile.rs` is the workload. With no arguments it sorts 10M random `u32` and 10M random `f32` thirty times each, copying an unsorted master buffer before every sort and timing only the sort. `profile [TYPE] [N] [ITERS] [ALGO]` narrows it: `TYPE` is `u32`, `f32` or `all`, `N` is the element count, `ITERS` is the number of sorts per type, and `ALGO` is `ts` for turbosort or `std` for `sort_unstable` (`sort_unstable_by(f32::total_cmp)` for `f32`). The build keeps frame pointers, because DWARF unwinding loses the AVX2 hot loops under `perf script`:

```sh
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling --example profile
perf record --call-graph fp -F 997 -e cycles:u -- target/profiling/examples/profile
perf script --inline | inferno-collapse-perf | inferno-flamegraph > docs/flamegraph.svg
```

### Where the cycles go

On the default workload the recorded profile put 73% of cycles in the scatter passes and 22% in the histogram scans. The rest was the harness's per-iteration copy (3%) and startup. The sorted-input check did not register, because on random data it stops within the first few elements.

The committed [flamegraph](flamegraph.svg) does not show this split. No turbosort function appears in it: 70.6% of it is glibc's `__memmove_avx_unaligned_erms` and 29.3% is `[unknown]` frames. It needs regenerating with the commands above.

### Cache behaviour

A `perf stat` sweep of `examples/profile` on random `u32` gave the table below. Each time is the mean of 30, 10 and 3 back-to-back sorts at 1M, 10M and 100M keys, so the times differ slightly from the criterion tables in the README; the ratios agree. The counters cover the whole process, including data generation and the per-iteration copy, identically for both sorts. Per-key counters do not depend on clock speed, which makes them the stable metric on a 15 W chip. The exact event list passed to `perf stat` was not recorded.

| Random `u32` | 1M | 10M | 100M |
| --- | --- | --- | --- |
| turbosort | 5.6 ms | 84 ms | 0.80 s |
| `sort_unstable` | 15.7 ms | 188 ms | 2.53 s |
| speedup | 2.8x | 2.2x | 3.2x |
| turbosort LLC misses per key | 0.32 | 0.95 | 0.99 |
| turbosort instructions per key | 53 | 55 | 61 |
| `sort_unstable` instructions per key | 180 | 210 | 245 |
| turbosort IPC | 2.1 | 1.9 | 1.9 |
| `sort_unstable` IPC | 2.9 | 2.8 | 2.8 |

LLC is the last-level cache, the L3 on this chip, and IPC is instructions per cycle.

The table shows two regimes. At 1M keys the data plus the scratch buffer (8 MB together) partly fit in the 8 MB L3, so only a third of keys miss it. From 10M up every key costs about one LLC miss, and the sort runs against memory with IPC near 1.9. The standard library's `sort_unstable`, a quicksort variant (ipnsort in current Rust), is the mirror image: it is compute-bound at IPC 2.8, and its instructions per key grow with log n (180 to 245 from 1M to 100M) while the radix sort stays near 55. That is why the gap widens to about 3x at 100M keys. Running the pair in the opposite order gave 2.97x, so the gap is not a thermal effect. The ratio was best at the largest size tested.

Recording cache misses instead of cycles (`perf record -e cache-misses:u` at 100M keys) put 88% of the sort's LLC misses in the scatter pass. The histogram scans and the copy-back are sequential streams, which the hardware prefetcher covers. dTLB misses (misses in the cache of virtual-to-physical page translations) stayed under 0.007 per key at every size with plain 4 KiB pages. The scatter writes 256 destination streams, each one sequential, so the set of pages in use stays near 256 however large the array grows.

## A second machine

These runs were made on 2026-09-27 against the 0.2.1 code (commit 403204f) on a shared cloud VM: a KVM guest with 4 vCPUs of an Intel Xeon at 2.80 GHz, 1 MiB of L2 per core and a 33 MiB L3 shared with other tenants, using Rust 1.94.1. They take fewer samples than the laptop run, on a host shared with other tenants, so treat them as a check on the direction of the README's numbers.

```sh
# Run A: every type at 1M, 10 samples per point
cargo bench --bench sort_bench --features parallel -- --sample-size 10 --warm-up-time 1 --measurement-time 2 '/1000000$'
# Run B: u32 and u64 at 4K and 1M, 20 samples per point
cargo bench --bench sort_bench -- --sample-size 20 --warm-up-time 2 --measurement-time 4 '^turbosort/u(32|64)/.*/(4096|1000000)$'
# Run C: u32 at 1M, turbosort and sort_unstable only, 30 samples per point
cargo bench --bench sort_bench -- --sample-size 30 --warm-up-time 2 --measurement-time 5 '^turbosort/u32/(turbosort|std_unstable)/1000000$'
```

| Run | Type | Size | std | turbosort | voracious | vs std | vs voracious |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | `u32` | 1M | 20.5 ms | 12.6 ms | 29.9 ms | 1.62x | 2.37x |
| A | `u64` | 1M | 22.4 ms | 69.2 ms | 42.7 ms | 0.32x | 0.62x |
| A | `f32` | 1M | 28.5 ms | 15.3 ms | 22.3 ms | 1.87x | 1.46x |
| A | `i32` | 1M | 20.2 ms | 20.9 ms | 29.7 ms | 0.97x | 1.42x |
| B | `u32` | 4K | 43.9 µs | 27.3 µs | 36.8 µs | 1.61x | 1.35x |
| B | `u32` | 1M | 19.5 ms | 22.1 ms | 28.6 ms | 0.88x | 1.29x |
| B | `u64` | 4K | 43.6 µs | 61.5 µs | 42.1 µs | 0.71x | 0.69x |
| B | `u64` | 1M | 21.9 ms | 72.6 ms | 40.3 ms | 0.30x | 0.56x |
| C | `u32` | 1M | 20.3 ms | 23.5 ms | not run | 0.86x | not run |

Each time is criterion's point estimate, and each ratio is the other sort's time divided by turbosort's. In run A, `sort_parallel` on 1M `u32` took 20.9 ms, against 19.8 ms for the serial sort and 20.8 ms for `sort_unstable`.

At 4K, where the data fits in cache, the ratios against `sort_unstable` match the laptop's to two digits (1.61x for `u32`, 0.71x for `u64`). At 1M they do not, and they varied: `sort_unstable` on `u32` stayed between 19.5 and 20.5 ms across the three runs, while turbosort's time moved from 12.6 ms to 23.5 ms. These runs do not show why. The `u64` loss at 1M held in both runs that measured it. A three-sort run of `examples/profile` on the same VM (`profile u32 1000000 3 ts`, then with `std`) gave 20.74 ms per sort for turbosort and 23.17 ms for `sort_unstable`.
