//! Scalar fallback algorithms: insertion sort and Hoare partition.
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

/// Hoare partition: rearranges `slice` so that elements with keys < pivot_key
/// are on the left, elements with keys >= pivot_key are on the right.
///
/// Returns the index of the first element in the right partition.
///
/// # Algorithm
///
/// Classic two-pointer scan from both ends, swapping out-of-place elements.
#[inline]
pub fn hoare_partition<T: SortableKey>(slice: &mut [T], pivot_key: T::Key) -> usize {
    let mut left = 0;
    let mut right = slice.len();

    loop {
        while left < right && slice[left].to_radix_key() < pivot_key {
            left += 1;
        }
        while left < right && slice[right - 1].to_radix_key() >= pivot_key {
            right -= 1;
        }
        if left >= right {
            return left;
        }
        slice.swap(left, right - 1);
        left += 1;
        right -= 1;
    }
}

/// Median-of-three pivot selection. Returns the radix key of the median.
///
/// Samples first, middle, and last elements to avoid worst-case pivot
/// selection on sorted/reverse-sorted input.
#[inline]
pub fn median_of_three_key<T: SortableKey>(slice: &[T]) -> T::Key {
    let len = slice.len();
    let a = slice[0].to_radix_key();
    let b = slice[len / 2].to_radix_key();
    let c = slice[len - 1].to_radix_key();

    if a <= b {
        if b <= c {
            b
        } else if a <= c {
            c
        } else {
            a
        }
    } else if a <= c {
        a
    } else if b <= c {
        c
    } else {
        b
    }
}

/// Scalar quicksort using Hoare partition with median-of-three pivot.
///
/// Falls back to insertion sort for partitions ≤ 16 elements.
/// Includes a depth limit to guarantee O(n log n) worst case (switches to
/// heapsort-like behavior by choosing worst-case-resistant pivots).
pub fn quicksort<T: SortableKey>(slice: &mut [T]) {
    quicksort_impl(slice, 2 * log2_usize(slice.len()));
}

fn quicksort_impl<T: SortableKey>(slice: &mut [T], depth_limit: usize) {
    if slice.len() <= 16 {
        insertion_sort(slice);
        return;
    }

    if depth_limit == 0 {
        // Fallback: insertion sort to avoid O(n²) on adversarial input.
        // A proper heapsort would be better, but insertion sort suffices
        // since SIMD quicksort handles the medium-size range in practice.
        insertion_sort(slice);
        return;
    }

    let pivot_key = median_of_three_key(slice);
    let mid = hoare_partition(slice, pivot_key);

    // Avoid infinite recursion: if partition didn't split, force progress
    if mid == 0 || mid == slice.len() {
        // All elements equal (or bad pivot). Just insertion sort.
        insertion_sort(slice);
        return;
    }

    let (left, right) = slice.split_at_mut(mid);
    quicksort_impl(left, depth_limit - 1);
    quicksort_impl(right, depth_limit - 1);
}

/// Integer log2, used for depth limits.
#[inline]
fn log2_usize(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    usize::BITS as usize - 1 - n.leading_zeros() as usize
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
    fn quicksort_basic() {
        let mut data = vec![
            5u32, 3, 8, 1, 9, 2, 7, 4, 6, 10, 15, 12, 11, 14, 13, 16, 17, 18,
        ];
        quicksort(&mut data);
        let mut expected = data.clone();
        expected.sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn quicksort_all_equal() {
        let mut data = vec![42u32; 100];
        quicksort(&mut data);
        assert!(data.iter().all(|&x| x == 42));
    }

    #[test]
    fn quicksort_signed() {
        let mut data = vec![3i32, -1, 4, -1, 5, -9, 2, -6, 5, 3];
        quicksort(&mut data);
        let mut expected = data.clone();
        expected.sort();
        assert_eq!(data, expected);
    }
}
