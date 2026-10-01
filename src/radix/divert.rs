//! Diverting LSD: radix-sort only the most significant digits, then finish.
//!
//! Sorting the top digits that carry about log2(n) bits leaves the keys in
//! order up to short runs that share every sorted digit. One scan then sorts
//! each run, which costs less than the low-digit passes it replaces. For
//! random 64-bit keys that is three byte passes instead of eight at 1M
//! elements.
//!
//! The digit count comes from the histograms the passes already need, so the
//! choice costs nothing. If it is wrong (digits that look independent but are
//! correlated, or skewed ones), the runs come out longer and the finishing
//! sort takes them: a slower sort, never a wrong one.

use crate::arch::scalar;
use crate::key::{SortableKey, UnsignedKey};

/// Bits beyond log2(n) that the sorted digits should carry. With two, about
/// one key in eight shares a run with another, and most runs are one key.
const SLACK_BITS: u32 = 2;

/// Runs up to this long are finished by insertion sort; longer runs by the
/// comparison sort.
const SHORT_RUN: usize = 16;

/// Diverting must skip at least this many digits to pay for the finishing
/// scan. Skipping two byte passes of a 4-byte key measured 5% slower at 1K
/// to 4K keys than running them; 8-byte keys skip four to six.
const MIN_SKIP: usize = 3;

/// How many of the lowest `n_live` live digits can be left unsorted.
///
/// `max_bucket(i)` is the largest bucket of the `i`-th live digit, least
/// significant first. Walking down from the most significant digit, each is
/// credited with log2(len / max_bucket) bits, the bits it is sure to split
/// keys by, until the credit reaches log2(len) + [`SLACK_BITS`]. Returns 0
/// unless that skips at least [`MIN_SKIP`] digits; the walk stops once that
/// is out of reach, so most calls look at no bucket at all.
pub fn skippable_digits(
    n_live: usize,
    len: usize,
    mut max_bucket: impl FnMut(usize) -> usize,
) -> usize {
    let log_len = log2_eighths(len);
    let need = log_len + 8 * SLACK_BITS;
    let mut have = 0;
    for i in (MIN_SKIP..n_live).rev() {
        // max <= len, and log2_eighths is monotonic, so this cannot underflow.
        have += log_len - log2_eighths(max_bucket(i));
        if have >= need {
            return i;
        }
    }
    0
}

/// Sort each run of keys that agree on every bit from `shift` up.
///
/// Call after the digits from `shift` up have been radix-sorted: the runs are
/// then in order, and each only needs sorting within itself.
pub fn finish_runs<T: SortableKey>(slice: &mut [T], shift: u32) {
    // An all-ones mask makes wide_digit the key shifted down. On 32-bit
    // targets the usize truncation can merge two adjacent runs, which only
    // means sorting a longer run.
    let prefix = |x: &T| x.to_radix_key().wide_digit(shift, usize::MAX);
    let Some(first) = slice.first() else {
        return;
    };
    // One prefix per key, compared with the run's; most runs are one key,
    // so the branch that ends a run is the predictable one.
    let mut start = 0;
    let mut run_prefix = prefix(first);
    for i in 1..slice.len() {
        let p = prefix(&slice[i]);
        if p != run_prefix {
            sort_run(&mut slice[start..i]);
            start = i;
            run_prefix = p;
        }
    }
    sort_run(&mut slice[start..]);
}

#[inline]
fn sort_run<T: SortableKey>(run: &mut [T]) {
    if run.len() > SHORT_RUN {
        scalar::comparison_sort(run);
    } else if run.len() > 1 {
        scalar::insertion_sort(run);
    }
}

/// log2(x) in eighths of a bit. The three bits after the leading one stand in
/// for the fraction (log2(1 + f) >= f), so it never overestimates, is at most
/// a quarter bit low, and is exact at powers of two. `x` must be non-zero.
fn log2_eighths(x: usize) -> u32 {
    debug_assert!(x > 0);
    let int = usize::BITS - 1 - x.leading_zeros();
    let frac = if int >= 3 {
        (x >> (int - 3)) & 7
    } else {
        (x << (3 - int)) & 7
    };
    8 * int + frac as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_eighths_brackets_the_true_value() {
        let mut prev = 0;
        for x in 1..5000usize {
            let est = log2_eighths(x);
            let exact = 8.0 * (x as f64).log2();
            assert!(est as f64 <= exact + 1e-9, "x={x}: {est} > {exact}");
            assert!(est as f64 > exact - 2.0, "x={x}: {est} too low for {exact}");
            assert!(est >= prev, "not monotonic at {x}");
            prev = est;
        }
        assert_eq!(log2_eighths(1 << 20), 160);
    }

    #[test]
    fn uniform_bytes_skip_the_low_digits() {
        // 1M keys with eight uniform bytes: three top bytes carry 24 bits,
        // enough for log2(1M) + 2 = 22, so the five low bytes are skipped.
        let len = 1 << 20;
        let max = len / 256 + 64;
        assert_eq!(skippable_digits(8, len, |_| max), 5);
    }

    #[test]
    fn no_skip_below_min_skip() {
        let len = 1 << 16;
        // Four uniform bytes, 64K keys: 18 bits needed, three bytes sorted,
        // only one skippable, so sort them all.
        assert_eq!(skippable_digits(4, len, |_| len / 256), 0);
        // 1K keys: two bytes carry the 12 bits needed, but two skipped
        // digits are fewer than MIN_SKIP.
        assert_eq!(skippable_digits(4, 1024, |_| 4), 0);
        // The same keys widened to eight bytes skip six.
        assert_eq!(skippable_digits(8, 1024, |_| 4), 6);
        // A digit that barely splits the keys earns almost no credit.
        assert_eq!(skippable_digits(8, len, |_| len - 1), 0);
        assert_eq!(skippable_digits(0, len, |_| unreachable!()), 0);
        assert_eq!(skippable_digits(MIN_SKIP, len, |_| unreachable!()), 0);
    }

    #[test]
    fn finish_runs_sorts_within_runs_only() {
        // Sorted on the high byte, shuffled within each high-byte run.
        let mut data: Vec<u16> = Vec::new();
        for hi in 0..20u16 {
            let run = (hi as usize * 7) % 40 + 1; // lengths 1..=40, some > SHORT_RUN
            for i in 0..run {
                data.push(hi << 8 | ((i * 37 + hi as usize) % 251) as u16);
            }
        }
        let mut expected = data.clone();
        expected.sort_unstable();
        finish_runs(&mut data, 8);
        assert_eq!(data, expected);
    }
}
