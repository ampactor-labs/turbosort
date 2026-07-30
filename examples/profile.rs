//! Profiling workload for perf/flamegraph runs.
//!
//! With no arguments: sorts 10M random u32 and 10M random f32, thirty times
//! each, cloning from an unsorted master buffer every iteration so the sort
//! never sees its own output. The per-type wrappers are `#[inline(never)]`
//! so each shows up as its own tower in a flamegraph.
//!
//! Build with frame pointers: dwarf unwinding loses the AVX2 hot loops under
//! `perf script`, so `--call-graph fp` is the recipe that works.
//!
//! ```sh
//! RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling --example profile
//! perf record --call-graph fp -F 997 -e cycles:u -- target/profiling/examples/profile
//! perf script --inline | inferno-collapse-perf | inferno-flamegraph > docs/flamegraph.svg
//! ```
//!
//! Arguments narrow the workload for `perf stat` sweeps:
//!
//! ```sh
//! profile [TYPE] [N] [ITERS] [ALGO]
//! #  TYPE   u32 | f32 | all      (default all)
//! #  N      element count        (default 10000000)
//! #  ITERS  sorts per type       (default 30)
//! #  ALGO   ts | std             (default ts; std = sort_unstable baseline)
//! ```
//!
//! Timing starts after the clone, so only the sort itself is measured.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum Algo {
    Turbosort,
    Std,
}

#[inline(never)]
fn profile_u32(master: &[u32], scratch: &mut Vec<u32>, iters: usize, algo: Algo) -> f64 {
    let mut total_s = 0.0;
    for _ in 0..iters {
        scratch.clear();
        scratch.extend_from_slice(master);
        let t = Instant::now();
        match algo {
            Algo::Turbosort => turbosort::sort(scratch),
            Algo::Std => scratch.sort_unstable(),
        }
        total_s += t.elapsed().as_secs_f64();
    }
    total_s
}

#[inline(never)]
fn profile_f32(master: &[f32], scratch: &mut Vec<f32>, iters: usize, algo: Algo) -> f64 {
    let mut total_s = 0.0;
    for _ in 0..iters {
        scratch.clear();
        scratch.extend_from_slice(master);
        let t = Instant::now();
        match algo {
            Algo::Turbosort => turbosort::sort(scratch),
            // total_cmp matches turbosort's total order, so both algorithms
            // do equivalent work on NaN-free data.
            Algo::Std => scratch.sort_unstable_by(f32::total_cmp),
        }
        total_s += t.elapsed().as_secs_f64();
    }
    total_s
}

fn report(label: &str, n: usize, iters: usize, total_s: f64) {
    let per_sort_ms = total_s / iters as f64 * 1e3;
    let mkeys_per_s = n as f64 * iters as f64 / total_s / 1e6;
    println!("{label}: {iters} sorts of {n} elements, {per_sort_ms:.2} ms/sort, {mkeys_per_s:.0} Mkeys/s");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ty = args.first().map(String::as_str).unwrap_or("all");
    let n: usize = args
        .get(1)
        .map(|a| a.parse().expect("N must be an integer"))
        .unwrap_or(10_000_000);
    let iters: usize = args
        .get(2)
        .map(|a| a.parse().expect("ITERS must be an integer"))
        .unwrap_or(30);
    let algo = match args.get(3).map(String::as_str) {
        None | Some("ts") => Algo::Turbosort,
        Some("std") => Algo::Std,
        Some(other) => panic!("unknown ALGO {other:?}, expected ts or std"),
    };
    assert!(
        matches!(ty, "u32" | "f32" | "all"),
        "unknown TYPE {ty:?}, expected u32, f32, or all"
    );

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    if ty == "u32" || ty == "all" {
        let master: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
        let mut scratch = Vec::with_capacity(n);
        let total_s = profile_u32(&master, &mut scratch, iters, algo);
        report("u32", n, iters, total_s);
        // Keep results observable so the optimizer cannot discard the sorts.
        assert!(scratch.windows(2).all(|w| w[0] <= w[1]));
    }

    if ty == "f32" || ty == "all" {
        let master: Vec<f32> = (0..n).map(|_| rng.gen_range(-1e6f32..1e6)).collect();
        let mut scratch = Vec::with_capacity(n);
        let total_s = profile_f32(&master, &mut scratch, iters, algo);
        report("f32", n, iters, total_s);
        assert!(scratch.windows(2).all(|w| w[0] <= w[1]));
    }
}
