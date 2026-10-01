//! Radix sort for primitive types, with SIMD sorting networks for short slices.
//!
//! `turbosort` sorts slices of the 10 primitive numeric types (`u8`, `u16`,
//! `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`) with an O(n) LSD
//! radix sort above 512 elements, and with AVX2 or NEON sorting networks
//! below that for 4-byte types.
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

// Unit tests link std in every configuration; without the `std` feature
// AVX2 detection stays off, so those runs cover the scalar paths on x86.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod key;

mod arch;
mod dispatch;
#[cfg(feature = "parallel")]
mod parallel;
mod presorted;
mod radix;
mod small;
mod tiny;

pub use key::SortableKey;

/// Sort a mutable slice of any [`SortableKey`] type in ascending order.
///
/// Picks an algorithm by input size:
/// - n ≤ 16: a sorting network for 4-byte keys on AVX2 (from 4 elements) or
///   NEON, `core`'s unstable sort otherwise
/// - 17 ≤ n ≤ 512: with AVX2 and a 4-byte key, register networks up to 128
///   elements and merged 128-element blocks above that; otherwise `core`'s
///   unstable sort
/// - n > 512: LSD radix sort (requires `alloc` feature)
///
/// Above 16 elements, input that is already sorted, ascending or descending,
/// is detected in one scan and finished without sorting, and `u8`/`i8` use a
/// counting sort from 48 elements.
///
/// Without the `alloc` feature, arrays over 512 elements use `core`'s
/// unstable sort, which needs no scratch memory.
///
/// # Examples
///
/// ```
/// let mut v = vec![3i32, -1, 4, -1, 5, -9];
/// turbosort::sort(&mut v);
/// assert_eq!(v, vec![-9, -1, -1, 3, 4, 5]);
/// ```
///
/// Floats sort in the order of `f32::total_cmp`. `f32::NAN` has its sign bit
/// clear and sorts last; a NaN with the sign bit set (on x86, `0.0 / 0.0`
/// computed at run time gives one) sorts first.
///
/// ```
/// let mut v = vec![1.0f32, f32::NAN, -0.0, 0.0, f32::NEG_INFINITY];
/// turbosort::sort(&mut v);
/// assert_eq!(v[0], f32::NEG_INFINITY);
/// assert!(v[1].to_bits() == (-0.0f32).to_bits()); // -0.0 < +0.0
/// assert!(v[4].is_nan()); // f32::NAN is a positive NaN
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
/// Uses rayon to distribute work across multiple cores. Below 131,072
/// elements it runs the serial [`sort`] instead.
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

/// Compiles every Rust code block in the README under `cargo test`, so a
/// drifting example fails the build instead of misleading a reader.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
