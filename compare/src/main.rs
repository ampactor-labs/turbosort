//! turbosort against the standard library, voracious and vqsort (Google
//! Highway's vectorized quicksort, AVX-512 where the CPU has it).
//!
//! For each length, about 4M random keys are cut into slices of that
//! length and every slice is sorted, so short sorts see fresh data rather
//! than one input the branch predictor has learned. Each figure is
//! nanoseconds per key, the best of nine rounds, with the median in
//! parentheses.
//!
//! ```sh
//! cargo run --release            # u32, u64 and f32
//! cargo run --release -- u32     # one type
//! ```

use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use voracious_radix_sort::RadixSort;

const TOTAL: usize = 1 << 22;
const ROUNDS: usize = 9;
const LENGTHS: [usize; 8] = [8, 16, 128, 512, 4096, 65_536, 1_000_000, 10_000_000];

/// Best and median nanoseconds per key for sorting `src` in slices of `len`.
fn time<T: Copy>(src: &[T], len: usize, mut sort: impl FnMut(&mut [T])) -> (f64, f64, Vec<T>) {
    let mut work = src.to_vec();
    let mut ns = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        work.copy_from_slice(src);
        let start = Instant::now();
        for slice in work.chunks_mut(len) {
            sort(slice);
        }
        ns.push(start.elapsed().as_nanos() as f64 / src.len() as f64);
    }
    ns.sort_by(f64::total_cmp);
    (ns[0], ns[ROUNDS / 2], work)
}

/// Total-order bits, so floats compare like their sorts order them.
trait Bits: Copy {
    fn bits(self) -> u64;
}
impl Bits for u32 {
    fn bits(self) -> u64 {
        self as u64
    }
}
impl Bits for u64 {
    fn bits(self) -> u64 {
        self
    }
}
impl Bits for f32 {
    fn bits(self) -> u64 {
        let b = self.to_bits();
        (if b >> 31 == 1 { !b } else { b | 1 << 31 }) as u64
    }
}

fn run<T>(
    name: &str,
    gen: impl Fn(&mut StdRng) -> T,
    std_sort: impl Fn(&mut [T]) + Copy,
    voracious: impl Fn(&mut [T]) + Copy,
) where
    T: Bits + turbosort::SortableKey + vqsort_rs::VqsortItem,
{
    let mut rng = StdRng::seed_from_u64(0x7EB0);
    println!("\n## {name}: ns per key, best of {ROUNDS} (median)\n");
    println!("| Length | std | turbosort | voracious | vqsort | turbosort vs std | turbosort vs vqsort |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for &len in &LENGTHS {
        let n = TOTAL.max(len) / len * len;
        let src: Vec<T> = (0..n).map(|_| gen(&mut rng)).collect();
        let keys = |v: &[T]| v.iter().map(|x| x.bits()).collect::<Vec<u64>>();
        let (s, s_med, expected) = time(&src, len, std_sort);
        let (t, t_med, got) = time(&src, len, |c: &mut [T]| turbosort::sort(c));
        assert!(
            keys(&got) == keys(&expected),
            "turbosort disagrees with std at {len}"
        );
        let (v, v_med, got) = time(&src, len, voracious);
        assert!(
            keys(&got) == keys(&expected),
            "voracious disagrees with std at {len}"
        );
        let (q, q_med, got) = time(&src, len, |c: &mut [T]| vqsort_rs::sort(c));
        assert!(
            keys(&got) == keys(&expected),
            "vqsort disagrees with std at {len}"
        );
        println!(
            "| {len} | {s:.2} ({s_med:.2}) | {t:.2} ({t_med:.2}) | {v:.2} ({v_med:.2}) | {q:.2} ({q_med:.2}) | {:.2}x | {:.2}x |",
            s / t,
            q / t
        );
    }
}

fn main() {
    let only = std::env::args().nth(1);
    let want = |name: &str| only.as_deref().map_or(true, |o| o == name);
    if want("u32") {
        run(
            "u32",
            |r| r.gen::<u32>(),
            |c: &mut [u32]| c.sort_unstable(),
            |c: &mut [u32]| c.voracious_sort(),
        );
    }
    if want("u64") {
        run(
            "u64",
            |r| r.gen::<u64>(),
            |c: &mut [u64]| c.sort_unstable(),
            |c: &mut [u64]| c.voracious_sort(),
        );
    }
    if want("f32") {
        run(
            "f32",
            |r| r.gen_range(-1e6f32..1e6),
            |c: &mut [f32]| c.sort_unstable_by(f32::total_cmp),
            |c: &mut [f32]| c.voracious_sort(),
        );
    }
}
