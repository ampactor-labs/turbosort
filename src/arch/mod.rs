//! Architecture-specific backend selection.
//!
//! At compile time, cfg gates select which SIMD backends are available.
//! At runtime, CPUID checks pick the best available instruction set.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

use crate::key::SortableKey;

/// Sort a small slice (n ≤ 16) using the best available backend.
#[inline]
pub fn sort_tiny<T: SortableKey>(slice: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_u32_type::<T>() {
            unsafe { x86_64::avx2::sort_tiny_u32_keys_generic(slice) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_neon_u32_type::<T>() {
            unsafe { aarch64::neon::sort_tiny_u32_keys_generic(slice) };
            return;
        }
    }
    scalar::insertion_sort(slice);
}

/// Sort a medium slice (17..=512) using the best available backend.
#[inline]
pub fn sort_small<T: SortableKey>(slice: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_u32_type::<T>() {
            unsafe { x86_64::avx2::quicksort_u32_keys_generic(slice) };
            return;
        }
    }
    scalar::quicksort(slice);
}

/// Check if AVX2 is available and the type has a 4-byte key (u32/i32/f32).
#[cfg(target_arch = "x86_64")]
#[inline]
fn is_avx2_u32_type<T: SortableKey>() -> bool {
    x86_64::avx2::is_available()
        && core::mem::size_of::<T>() == 4
        && core::mem::size_of::<T::Key>() == 4
}

/// Check if the type has a 4-byte key for NEON acceleration.
#[cfg(target_arch = "aarch64")]
#[inline]
fn is_neon_u32_type<T: SortableKey>() -> bool {
    core::mem::size_of::<T>() == 4 && core::mem::size_of::<T::Key>() == 4
}
