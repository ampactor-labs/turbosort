//! Traits for mapping primitive types to their unsigned radix-sortable keys.
//!
//! Every sortable primitive implements [`SortableKey`], which converts the type
//! to/from an unsigned integer that preserves sort order. This lets the radix
//! sort operate uniformly on unsigned keys regardless of the original type.
//!
//! # Transforms
//!
//! - **Unsigned integers:** identity (already radix-sortable).
//! - **Signed integers:** XOR the sign bit to flip negative/positive ordering.
//! - **Floats:** IEEE 754 trick — positive: flip sign bit; negative: flip all bits.
//!   This maps the IEEE 754 total order (`f32::total_cmp`) to unsigned integer
//!   order: -0.0 < +0.0, NaNs with the sign bit clear sort after +inf, and
//!   NaNs with it set sort before -inf.

use core::mem;

/// Unsigned integer keys, as the radix sort sees them.
///
/// Implementors must be `Copy`, convertible to/from byte arrays, and support
/// the bitwise operations needed for radix digit extraction.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait UnsignedKey: Copy + Ord + private::Sealed + Sized {
    /// Number of bytes in the key (determines radix pass count).
    const BYTES: usize = mem::size_of::<Self>();

    /// Extract the `pass`-th radix digit (byte) from this key.
    ///
    /// Pass 0 extracts the least-significant byte, pass `BYTES - 1` the most-significant.
    fn radix_digit(self, pass: usize) -> u8;

    /// Extract a digit of arbitrary width: the bits at `shift..shift+width`,
    /// where `mask == (1 << width) - 1`. `shift` must be less than the key width.
    fn wide_digit(self, shift: u32, mask: usize) -> usize;

    /// Construct a key from a single byte value. Only meaningful for one-byte
    /// keys, where the digit is the whole key (the counting-sort path).
    fn from_digit(digit: u8) -> Self;

    /// The key zero-extended to 64 bits.
    fn to_u64(self) -> u64;
}

/// Trait mapping a sortable primitive to its unsigned radix key.
///
/// # Sealed
///
/// Only the 10 primitive types (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64)
/// implement this trait. Custom types should use `slice::sort_unstable` instead.
pub trait SortableKey: Copy + private::Sealed + Sized {
    /// The unsigned key type that preserves sort order.
    type Key: UnsignedKey;

    /// Convert this value to its radix-sortable unsigned key.
    fn to_radix_key(self) -> Self::Key;

    /// Convert a radix-sortable unsigned key back to the original type.
    fn from_radix_key(key: Self::Key) -> Self;
}

mod private {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

// --- UnsignedKey impls ---

macro_rules! impl_unsigned_key {
    ($($ty:ty),*) => {
        $(
            impl UnsignedKey for $ty {
                #[inline(always)]
                fn radix_digit(self, pass: usize) -> u8 {
                    (self >> (pass * 8)) as u8
                }

                #[inline(always)]
                fn wide_digit(self, shift: u32, mask: usize) -> usize {
                    // The digit occupies the low bits after the shift, so the
                    // usize truncation on 32-bit targets cannot lose digit bits.
                    ((self >> shift) as usize) & mask
                }

                #[inline(always)]
                fn from_digit(digit: u8) -> Self {
                    digit as $ty
                }

                #[inline(always)]
                fn to_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    };
}

impl_unsigned_key!(u8, u16, u32, u64);

// --- SortableKey impls: unsigned integers (identity) ---

macro_rules! impl_sortable_unsigned {
    ($($ty:ty),*) => {
        $(
            impl SortableKey for $ty {
                type Key = Self;

                #[inline(always)]
                fn to_radix_key(self) -> Self { self }

                #[inline(always)]
                fn from_radix_key(key: Self) -> Self { key }
            }
        )*
    };
}

impl_sortable_unsigned!(u8, u16, u32, u64);

// --- SortableKey impls: signed integers (flip sign bit) ---

macro_rules! impl_sortable_signed {
    ($signed:ty, $unsigned:ty) => {
        impl SortableKey for $signed {
            type Key = $unsigned;

            #[inline(always)]
            fn to_radix_key(self) -> $unsigned {
                // XOR with sign-bit mask: flips ordering so that
                // i::MIN → 0, 0 → SIGN_BIT, i::MAX → u::MAX
                (self as $unsigned) ^ (1 << (<$unsigned>::BITS - 1))
            }

            #[inline(always)]
            fn from_radix_key(key: $unsigned) -> Self {
                (key ^ (1 << (<$unsigned>::BITS - 1))) as Self
            }
        }
    };
}

impl_sortable_signed!(i8, u8);
impl_sortable_signed!(i16, u16);
impl_sortable_signed!(i32, u32);
impl_sortable_signed!(i64, u64);

// --- SortableKey impls: floats (IEEE 754 radix trick) ---

impl SortableKey for f32 {
    type Key = u32;

    #[inline(always)]
    fn to_radix_key(self) -> u32 {
        let bits = self.to_bits();
        // Positive floats (sign bit 0): flip sign bit → maps to upper half of u32.
        // Negative floats (sign bit 1): flip ALL bits → maps to lower half, reversed.
        // NaN: exponent 0xFF, so a positive NaN lands after +inf and a negative
        // one (sign bit set, as x86 produces for 0.0/0.0) before -inf.
        if bits & 0x8000_0000 == 0 {
            bits ^ 0x8000_0000
        } else {
            !bits
        }
    }

    #[inline(always)]
    fn from_radix_key(key: u32) -> Self {
        let bits = if key & 0x8000_0000 == 0 {
            !key
        } else {
            key ^ 0x8000_0000
        };
        f32::from_bits(bits)
    }
}

impl SortableKey for f64 {
    type Key = u64;

    #[inline(always)]
    fn to_radix_key(self) -> u64 {
        let bits = self.to_bits();
        if bits & 0x8000_0000_0000_0000 == 0 {
            bits ^ 0x8000_0000_0000_0000
        } else {
            !bits
        }
    }

    #[inline(always)]
    fn from_radix_key(key: u64) -> Self {
        let bits = if key & 0x8000_0000_0000_0000 == 0 {
            !key
        } else {
            key ^ 0x8000_0000_0000_0000
        };
        f64::from_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that to_radix_key preserves ordering for all signed/float types.
    #[test]
    fn signed_key_ordering() {
        // i32: MIN < -1 < 0 < 1 < MAX
        let vals: Vec<i32> = vec![i32::MIN, -1, 0, 1, i32::MAX];
        let keys: Vec<u32> = vals.iter().map(|v| v.to_radix_key()).collect();
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "key ordering broken: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn float_key_ordering() {
        let vals: Vec<f32> = vec![
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let keys: Vec<u32> = vals.iter().map(|v| v.to_radix_key()).collect();
        for w in keys.windows(2) {
            assert!(
                w[0] < w[1],
                "float key ordering broken: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn roundtrip_signed() {
        for v in [i32::MIN, -1, 0, 1, i32::MAX] {
            assert_eq!(v, i32::from_radix_key(v.to_radix_key()));
        }
    }

    #[test]
    fn roundtrip_float() {
        for v in [f32::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY] {
            let rt = f32::from_radix_key(v.to_radix_key());
            assert_eq!(v.to_bits(), rt.to_bits(), "roundtrip failed for {v}");
        }
    }
}
