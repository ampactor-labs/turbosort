//! SIMD-accelerated radix sort for primitive types.
//!
//! `turbosort` provides O(n) LSD radix sort with SIMD acceleration for the 10
//! primitive numeric types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`,
//! `i64`, `f32`, `f64`.
//!
//! # Features
//!
//! - **`std`** (default): enables heap allocation for internal buffers.
//! - **`alloc`**: like `std` but for `no_std` + allocator environments.
//! - **`parallel`**: enables `sort_parallel()` via rayon for huge arrays.
//!
//! # Usage
//!
//! ```
//! let mut data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6];
//! turbosort::sort(&mut data);
//! assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
//! ```
//!
//! For allocation-sensitive code, use [`sort_with_buffer`]:
//!
//! ```
//! let mut data = [5u32, 3, 8, 1, 9];
//! let mut buf = [0u32; 5];
//! turbosort::sort_with_buffer(&mut data, &mut buf);
//! assert_eq!(data, [1, 3, 5, 8, 9]);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod key;

mod arch;
mod dispatch;
#[cfg(target_arch = "x86_64")]
mod lut;
#[cfg(feature = "parallel")]
mod parallel;
mod radix;
mod small;
mod tiny;

pub use key::SortableKey;

/// Sort a mutable slice of any [`SortableKey`] type in ascending order.
///
/// Automatically selects the optimal algorithm based on input size:
/// - n ≤ 16: sorting network
/// - 17 ≤ n ≤ 512: SIMD sorting networks up to 128 elements, then quicksort
///   with network leaves (scalar quicksort without AVX2)
/// - n > 512: LSD radix sort (requires `alloc` feature); already-sorted
///   inputs are detected in one scan and returned unchanged
///
/// Without the `alloc` feature, arrays > 512 elements fall back to quicksort.
///
/// # Examples
///
/// ```
/// let mut v = vec![3i32, -1, 4, -1, 5, -9];
/// turbosort::sort(&mut v);
/// assert_eq!(v, vec![-9, -1, -1, 3, 4, 5]);
/// ```
///
/// ```
/// let mut v = vec![1.0f32, f32::NAN, -0.0, 0.0, f32::NEG_INFINITY];
/// turbosort::sort(&mut v);
/// assert_eq!(v[0], f32::NEG_INFINITY);
/// assert!(v[1].to_bits() == (-0.0f32).to_bits()); // -0.0 < +0.0
/// assert!(v[4].is_nan()); // NaN sorted to end
/// ```
#[inline]
pub fn sort<T: SortableKey>(slice: &mut [T]) {
    dispatch::sort(slice);
}

/// Sort a mutable slice using a caller-provided buffer.
///
/// The buffer must be at least as long as the slice. This avoids internal
/// allocation, making it suitable for `no_std`, embedded, or latency-sensitive
/// contexts.
///
/// # Panics
///
/// Panics if `buffer.len() < slice.len()`.
///
/// # Examples
///
/// ```
/// let mut data = [42u64, 7, 99, 1, 0];
/// let mut buf = [0u64; 5];
/// turbosort::sort_with_buffer(&mut data, &mut buf);
/// assert_eq!(data, [0, 1, 7, 42, 99]);
/// ```
#[inline]
pub fn sort_with_buffer<T: SortableKey>(slice: &mut [T], buffer: &mut [T]) {
    dispatch::sort_with_buffer(slice, buffer);
}

/// Sort a mutable slice using parallel LSD radix sort.
///
/// Uses rayon to distribute work across multiple cores. Falls back to
/// single-threaded radix sort for arrays below 131K elements.
///
/// Requires the `parallel` feature.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "parallel")]
/// # {
/// let mut data: Vec<u32> = (0..1_000_000).rev().collect();
/// turbosort::sort_parallel(&mut data);
/// assert_eq!(data, (0..1_000_000).collect::<Vec<u32>>());
/// # }
/// ```
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[inline]
pub fn sort_parallel<T: SortableKey + Send + Sync>(slice: &mut [T])
where
    T::Key: Send + Sync,
{
    parallel::sort(slice);
}
