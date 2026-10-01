//! Scalar fallback algorithms: insertion sort and a comparison sort.
//!
//! Used on all platforms as the baseline and on architectures without SIMD support.

use crate::key::SortableKey;

/// Insertion sort on transformed keys. Optimal for n ≤ 16.
///
/// Operates in-place by comparing radix keys, but swaps original elements
/// to maintain the type-correct output.
///
/// # Performance
///
/// O(n²) worst case, but fast for small n due to minimal overhead and
/// branch-predictor-friendly access patterns.
#[inline]
pub fn insertion_sort<T: SortableKey>(slice: &mut [T]) {
    for i in 1..slice.len() {
        let key_i = slice[i].to_radix_key();
        let mut j = i;
        while j > 0 && slice[j - 1].to_radix_key() > key_i {
            slice.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Unstable comparison sort on the radix keys, for slices that neither the
/// SIMD networks nor the radix sort take.
///
/// This is `core`'s `sort_unstable_by_key`: it allocates nothing and is
/// O(n log n) on every input pattern.
pub fn comparison_sort<T: SortableKey>(slice: &mut [T]) {
    slice.sort_unstable_by_key(|x| x.to_radix_key());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_sort_basic() {
        let mut data = vec![5i32, 3, 8, 1, 9, 2, 7, 4, 6];
        insertion_sort(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn comparison_sort_basic() {
        let mut data = vec![
            5u32, 3, 8, 1, 9, 2, 7, 4, 6, 10, 15, 12, 11, 14, 13, 16, 17, 18,
        ];
        comparison_sort(&mut data);
        let mut expected = data.clone();
        expected.sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn comparison_sort_all_equal() {
        let mut data = vec![42u32; 100];
        comparison_sort(&mut data);
        assert!(data.iter().all(|&x| x == 42));
    }

    #[test]
    fn comparison_sort_signed() {
        let mut data = vec![3i32, -1, 4, -1, 5, -9, 2, -6, 5, 3];
        comparison_sort(&mut data);
        let mut expected = data.clone();
        expected.sort();
        assert_eq!(data, expected);
    }
}
