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
        if try_avx2_tiny(slice) {
            return;
        }
    }
    scalar::insertion_sort(slice);
}

/// Try to sort tiny arrays with AVX2. Returns true if handled.
#[cfg(target_arch = "x86_64")]
#[inline]
fn try_avx2_tiny<T: SortableKey>(slice: &mut [T]) -> bool {
    if !x86_64::avx2::is_available() {
        return false;
    }

    // AVX2 sorting network: 4-byte types with 4-byte keys (u32, i32, f32)
    if core::mem::size_of::<T>() == 4 && core::mem::size_of::<T::Key>() == 4 {
        unsafe { x86_64::avx2::sort_tiny_u32_keys_generic(slice) };
        return true;
    }

    false
}

/// Sort a medium slice (17..=512) using the best available backend.
#[inline]
pub fn sort_small<T: SortableKey>(slice: &mut [T]) {
    scalar::quicksort(slice);
}
