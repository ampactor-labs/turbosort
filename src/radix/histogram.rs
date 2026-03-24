//! Histogram utilities for radix sort.

/// Check if a pass can be skipped (all elements have the same digit).
///
/// A pass where one bin has count == total length means that byte position
/// is constant across all keys — the scatter would be a no-op.
#[inline]
pub fn is_pass_trivial(histogram: &[usize], total: usize) -> bool {
    histogram.contains(&total)
}
