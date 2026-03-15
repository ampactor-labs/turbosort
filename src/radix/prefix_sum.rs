//! Exclusive prefix sum for radix sort offset computation.
//!
//! Converts a histogram (counts per digit) into an offset table where
//! `offsets[digit]` = starting position for elements with that digit.

/// Compute exclusive prefix sum in-place.
///
/// After this call, `histogram[i]` contains the sum of all original values
/// at indices `0..i`. The original counts are consumed.
///
/// # Example
///
/// ```text
/// Input:  [3, 1, 0, 2]
/// Output: [0, 3, 4, 4]  (exclusive scan)
/// ```
pub fn exclusive_prefix_sum(histogram: &mut [usize]) {
    let mut sum = 0;
    for count in histogram.iter_mut() {
        let c = *count;
        *count = sum;
        sum += c;
    }
}
