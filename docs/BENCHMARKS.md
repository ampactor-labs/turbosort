# Benchmarks and profiling

This file holds the detail behind the Benchmarks section of the [README](../README.md): the profile of the 0.2.1 release on an Intel i7-8665U, a check on a second machine, and before-and-after runs for the changes since 0.2.1. The 0.2.1 criterion tables are in the README.

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

## Since 0.2.1

These runs were made on 2026-10-01 on a shared cloud VM: a KVM guest with 4 vCPUs of an Intel Xeon at 2.10 GHz (AVX2 and AVX-512), 48 KiB of L1d and 2 MiB of L2 per core and a shared 260 MiB L3, using Rust 1.97.0. "Before" is commit 0ccf74e, the 0.2.1 code with later README changes, with the new `benches/sort_bench.rs` copied in so that both sides run the same benchmark file; "after" is the code with the changes listed under 0.2.2 in [CHANGELOG.md](../CHANGELOG.md). Each side is one criterion run at 20 samples per point, and each ratio is `sort_unstable`'s time from the same run divided by turbosort's. The host is shared, so absolute times drift between runs; compare the ratios.

```sh
# Run in each tree; the second command is the random-input groups.
cargo bench --bench sort_bench -- '^patterns/' --sample-size 20 --warm-up-time 1 --measurement-time 2
cargo bench --bench sort_bench -- '^turbosort/(u32|u64|i32|f32)/(turbosort|std_unstable)/(16|128|512|4096|65536|1000000)$' --sample-size 20 --warm-up-time 1 --measurement-time 3
```

### Patterned inputs

The `patterns` group: sorted and reversed input, four distinct values in rotation, a pipe organ (ascending then descending), a sawtooth of eight ascending runs, and a random permutation of 0..n.

| Input | Size | Before | After | Before vs `sort_unstable` | After vs `sort_unstable` |
| --- | --- | --- | --- | --- | --- |
| `u32`, sorted | 512 | 5.06 µs | 81 ns | 0.05x | 2.98x |
| `u32`, reversed | 512 | 5.53 µs | 142 ns | 0.05x | 1.91x |
| `u32`, four distinct values | 512 | 36.1 µs | 1.89 µs | 0.06x | 0.84x |
| `u32`, pipe organ | 512 | 7.56 µs | 1.91 µs | 0.52x | 2.25x |
| `u32`, sawtooth | 512 | 43 µs | 1.8 µs | 0.11x | 1.78x |
| `u32`, random permutation | 512 | 3.19 µs | 1.83 µs | 1.17x | 2.33x |
| `u64`, sorted | 512 | 2.34 µs | 122 ns | 0.12x | 1.77x |
| `u64`, reversed | 512 | 2.21 µs | 194 ns | 0.14x | 1.28x |
| `u64`, four distinct values | 512 | 41.5 µs | 1.75 µs | 0.05x | 1.05x |
| `u64`, pipe organ | 512 | 39.8 µs | 4.15 µs | 0.10x | 1.03x |
| `u64`, sawtooth | 512 | 34.9 µs | 3.65 µs | 0.11x | 1.08x |
| `u64`, random permutation | 512 | 4.82 µs | 3.72 µs | 0.89x | 0.98x |
| `u32`, sorted | 65,536 | 23.9 µs | 9.4 µs | 1.03x | 2.80x |
| `u32`, reversed | 65,536 | 865 µs | 16.5 µs | 0.04x | 1.67x |
| `u32`, four distinct values | 65,536 | 229 µs | 252 µs | 0.91x | 0.89x |
| `u32`, pipe organ | 65,536 | 1 ms | 969 µs | 1.14x | 1.14x |
| `u32`, sawtooth | 65,536 | 1.06 ms | 1.11 ms | 1.03x | 0.76x |
| `u32`, random permutation | 65,536 | 796 µs | 758 µs | 1.26x | 1.48x |
| `u64`, sorted | 65,536 | 40.9 µs | 17.6 µs | 0.84x | 1.46x |
| `u64`, reversed | 65,536 | 1.12 ms | 28.8 µs | 0.03x | 1.27x |
| `u64`, four distinct values | 65,536 | 390 µs | 415 µs | 0.53x | 0.54x |
| `u64`, pipe organ | 65,536 | 717 µs | 805 µs | 1.43x | 1.50x |
| `u64`, sawtooth | 65,536 | 639 µs | 555 µs | 1.42x | 1.71x |
| `u64`, random permutation | 65,536 | 964 µs | 924 µs | 1.05x | 1.24x |

At 512 elements the ordered and repetitive patterns were 0.05x to 0.52x of `sort_unstable` before: the quicksorts in the 17 to 512 tier handed the whole slice to insertion sort when median-of-three picked the minimum. The reversed rows at 65,536 ran every radix pass before and are reversed in one scan now. The other 65,536-element rows run the same radix sort on both sides, so their differences are run-to-run noise (the `u32` sawtooth moved from 1.03x to 0.76x with no code change). Four distinct values stays below `sort_unstable` at both widths, because the standard library partitions equal keys away while the radix sort runs every live pass.

### Write combining and constant bytes

Two later changes, measured with the same patterns benchmark on the same VM, as the mean point estimates criterion recorded. Equal-sized buckets (a permutation of 0..n, a pipe organ) made the plain scatter's write positions contend for a few cache sets, and the byte passes now stage writes in cache lines for that shape. Separately, the radix sort now histograms only the bytes that vary, which is what four small distinct values gain from.

| Input | Size | Before | After | After vs `sort_unstable` |
| --- | --- | --- | --- | --- |
| `u32`, random permutation | 65,536 | 758 µs | 344 µs | 3.14x |
| `u32`, pipe organ | 65,536 | 969 µs | 692 µs | 1.70x |
| `u32`, sawtooth | 65,536 | 1.11 ms | 827 µs | 1.28x |
| `u32`, four distinct values | 65,536 | 252 µs | 168 µs | 1.23x |
| `u64`, four distinct values | 65,536 | 415 µs | 176 µs | 1.33x |

8-byte keys of this size take the 11-bit path, which does not combine, so their permutation, pipe-organ and sawtooth rows did not change.

### Random inputs

The existing groups, minus voracious and 10M. `u32`, `i32` and `f32` at 4K and up run the same radix code on both sides, so their rows show the noise; `u64` changed because the radix sort now diverts, and 512 `u32` because of the network-and-merge tier.

| Type | Size | Before | After | Before vs `sort_unstable` | After vs `sort_unstable` |
| --- | --- | --- | --- | --- | --- |
| `u32` | 16 | 42 ns | 23 ns | 1.37x | 2.04x |
| `u32` | 128 | 298 ns | 294 ns | 2.25x | 2.51x |
| `u32` | 512 | 3.15 µs | 2.06 µs | 1.46x | 2.15x |
| `u32` | 4,096 | 23.1 µs | 24 µs | 1.68x | 1.65x |
| `u32` | 65,536 | 410 µs | 430 µs | 2.52x | 2.35x |
| `u32` | 1M | 11.6 ms | 10.3 ms | 1.64x | 1.88x |
| `i32` | 128 | 294 ns | 304 ns | 2.48x | 2.56x |
| `i32` | 4,096 | 24.2 µs | 21.2 µs | 1.65x | 1.85x |
| `i32` | 1M | 10.1 ms | 9.81 ms | 2.13x | 1.91x |
| `f32` | 128 | 286 ns | 294 ns | 4.04x | 3.92x |
| `f32` | 4,096 | 38.9 µs | 32.3 µs | 1.68x | 1.97x |
| `f32` | 1M | 9.32 ms | 9.67 ms | 3.22x | 2.91x |
| `u64` | 128 | 795 ns | 680 ns | 0.89x | 1.07x |
| `u64` | 4,096 | 65.7 µs | 39.1 µs | 0.63x | 1.18x |
| `u64` | 1M | 31.8 ms | 21 ms | 0.65x | 0.90x |

### Batches of short slices

The groups above sort one input over and over, so at small sizes the branch predictor learns the comparisons that insertion sort and quicksort make, which flatters them. The `small_batches` group sorts 16,384 random `u32` cut into slices of one length, so every slice is new to the predictor. Each side is one run at 20 samples, with 0.2.1 at commit 0ccf74e as above:

```sh
cargo bench --bench sort_bench -- '^small_batches/' --sample-size 20 --warm-up-time 1 --measurement-time 3
```

| Slice length | `sort_unstable`, per slice | 0.2.1, per slice | 0.2.2, per slice | 0.2.1 vs `sort_unstable` | 0.2.2 vs `sort_unstable` |
| --- | --- | --- | --- | --- | --- |
| 4 | 11.6 ns | 31.8 ns | 7.8 ns | 0.36x | 1.49x |
| 8 | 51.5 ns | 32.0 ns | 6.8 ns | 1.82x | 7.55x |
| 16 | 182 ns | 44.8 ns | 14.7 ns | 3.77x | 12.4x |
| 17 | 196 ns | 120 ns | 42.3 ns | 1.65x | 4.62x |
| 32 | 158 ns | 109 ns | 41.3 ns | 1.45x | 3.84x |
| 64 | 379 ns | 161 ns | 103 ns | 2.20x | 3.68x |
| 128 | 801 ns | 308 ns | 261 ns | 2.79x | 3.07x |
| 512 | 4.58 µs | 5.22 µs | 1.90 µs | 0.84x | 2.41x |

The `sort_unstable` column is from the 0.2.2 run. Three changes account for the short lengths. The networks used to copy keys one at a time into a stack buffer and read it back with vector loads, and a vector load that overlaps several smaller stores waits for them to reach the cache; keys now go straight into registers. Depending on the type and the calling code, the compiler inlined the radix sort into `sort`, and then every call probed 40 to 48 KiB of stack for its histograms before looking at the length (0.2.1 did for `f32`; during 0.2.2's development it did for `u32` until the fix); the radix cores are no longer inlined. And slices of 2 and 3 use insertion sort, which beats a network's fixed cost of about 7 ns there. At 512 the quicksort that 0.2.1 ran lost to `sort_unstable` on fresh inputs, although it won the single-input benchmark (1.46x above).

### sort_by_key

`sort_by_key` and `sort_by_key_with_buffer` are new, so there is no 0.2.1 side; the baseline is the standard library's stable `sort_by_key`, which keeps equal keys in order as both of them do. The `by_key` group sorts 32-byte records (a key and 24 bytes of payload) by a random key, from one run at 10 samples on the same VM; the buffered series reuses one `SortByKeyBuffer` for every iteration:

```sh
cargo bench --bench sort_bench -- '^by_key/' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

| Key | Records | Standard library | `sort_by_key` | Buffer reused | `sort_by_key` vs standard library | Buffer reused vs standard library |
| --- | --- | --- | --- | --- | --- | --- |
| `u32` | 1,000 | 17.4 µs | 9.3 µs | 10.5 µs | 1.87x | 1.66x |
| `u32` | 65,536 | 3.40 ms | 0.91 ms | 1.06 ms | 3.74x | 3.22x |
| `u32` | 1M | 83.1 ms | 73.2 ms | 39.3 ms | 1.14x | 2.11x |
| `u64` | 1,000 | 17.5 µs | 12.1 µs | 12.1 µs | 1.45x | 1.45x |
| `u64` | 65,536 | 3.35 ms | 1.49 ms | 1.51 ms | 2.26x | 2.22x |
| `u64` | 1M | 87.6 ms | 92.9 ms | 64.1 ms | 0.94x | 1.37x |

An earlier run of the allocating version measured 1.84x, 2.97x and 1.32x for `u32` and 1.28x, 2.01x and 0.95x for `u64`, so differences of 10 to 20% between runs are noise; the gap at a million records is not. There, timing each phase in a scratch harness put 75% (`u64`) to 85% (`u32`) of the time in the radix passes over the entries and in moving the records into their new order, which write 48 to 64 MB of freshly allocated memory per call; the buffered version writes into memory it has already touched. The same harness, sorting batches of distinct record arrays, measured `u32` keys at 2.1x at a million, and found the losses the README lists: few distinct keys (0.6x for `u64` at a million), and input already sorted (0.7x) or reversed (0.5x), where the standard library finishes in one pass over the records.

