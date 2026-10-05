//! Stable sorting by a key ([`crate::sort_by_key`] and
//! [`crate::sort_by_key_with_buffer`]).
//!
//! Each key is computed once and stored with its element's position. An LSD
//! radix sort over those entries, which is stable, sorts the key bytes that
//! vary; positions are never sorted, because the entries start in position
//! order. Wide keys divert as in the main radix sort: when the top bytes
//! nearly order the keys, the rest are finished per run. The elements are
//! then moved into the sorted order through a buffer, as bitwise moves, so
//! they need not be `Copy` or `Clone`. All of that memory lives in a
//! [`SortByKeyBuffer`], which a caller that sorts often keeps between calls.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use crate::key::{SortableKey, UnsignedKey};
use crate::radix::{divert, prefix_sum};

/// Below this length [`sort_by_key`] is the standard library's stable sort,
/// which beats allocating buffers, building entries and moving every
/// element there.
const RADIX_MIN: usize = 128;

/// Below this length [`sort_by_key_with_buffer`] insertion-sorts its entries
/// instead of running radix passes. Through a reused buffer the two tied at
/// 64 random keys; insertion was 2-4x faster at 16 to 32 and lost from 96.
const INSERTION_MAX: usize = 64;

/// A key and the position of its element.
#[derive(Clone, Copy)]
struct Entry<K> {
    key: K,
    pos: u32,
}

/// Reusable memory for [`sort_by_key_with_buffer`](crate::sort_by_key_with_buffer).
///
/// It holds two arrays of a key and a 4-byte position per element and room
/// for the elements themselves, and grows to the longest slice it has
/// sorted. Kept between calls, it makes repeated sorts allocation-free. It
/// never holds elements between calls: dropping it drops none.
pub struct SortByKeyBuffer<T, K: SortableKey> {
    entries: Vec<Entry<K::Key>>,
    scratch: Vec<Entry<K::Key>>,
    // Always length 0; only its capacity is used.
    moved: Vec<T>,
}

impl<T, K: SortableKey> SortByKeyBuffer<T, K> {
    /// An empty buffer; the first sort allocates what it needs.
    pub const fn new() -> Self {
        SortByKeyBuffer {
            entries: Vec::new(),
            scratch: Vec::new(),
            moved: Vec::new(),
        }
    }

    /// A buffer that can sort `len` elements without allocating.
    pub fn with_capacity(len: usize) -> Self {
        SortByKeyBuffer {
            entries: Vec::with_capacity(len),
            scratch: Vec::with_capacity(len),
            moved: Vec::with_capacity(len),
        }
    }

    /// How many elements it can sort without allocating.
    pub fn capacity(&self) -> usize {
        self.entries
            .capacity()
            .min(self.scratch.capacity())
            .min(self.moved.capacity())
    }
}

impl<T, K: SortableKey> Default for SortByKeyBuffer<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, K: SortableKey> fmt::Debug for SortByKeyBuffer<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SortByKeyBuffer")
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// See [`crate::sort_by_key`].
pub fn sort_by_key<T, K, F>(slice: &mut [T], mut key: F)
where
    K: SortableKey,
    F: FnMut(&T) -> K,
{
    if slice.len() < RADIX_MIN {
        slice.sort_by_key(|x| key(x).to_radix_key());
    } else {
        sort_by_key_with_buffer(slice, &mut SortByKeyBuffer::new(), key);
    }
}

/// See [`crate::sort_by_key_with_buffer`].
pub fn sort_by_key_with_buffer<T, K, F>(
    slice: &mut [T],
    buffer: &mut SortByKeyBuffer<T, K>,
    mut key: F,
) where
    K: SortableKey,
    F: FnMut(&T) -> K,
{
    let len = slice.len();
    if len < 2 {
        return;
    }
    if len > u32::MAX as usize {
        slice.sort_by_key(|x| key(x).to_radix_key());
        return;
    }
    let entries = &mut buffer.entries;
    entries.clear();
    entries.extend(slice.iter().zip(0..).map(|(x, pos)| Entry {
        key: key(x).to_radix_key(),
        pos,
    }));

    if entries.windows(2).all(|w| w[0].key <= w[1].key) {
        return;
    }
    if entries.windows(2).all(|w| w[0].key >= w[1].key) {
        // Reversing sorts the keys but puts equal keys in reverse order;
        // reversing each run of them again restores it. The run at
        // entries[start..i] lands at slice[len - i..len - start].
        slice.reverse();
        let mut start = 0;
        for i in 1..=len {
            if i == len || entries[i].key != entries[start].key {
                if i - start > 1 {
                    slice[len - i..len - start].reverse();
                }
                start = i;
            }
        }
        return;
    }

    if len < INSERTION_MAX {
        insertion_sort(entries);
    } else {
        sort_entries(entries, &mut buffer.scratch);
    }
    move_into_order(slice, &buffer.entries, &mut buffer.moved);
}

/// Sort a few `entries` stably by key: each one moves past the larger keys
/// before it, never past an equal one.
fn insertion_sort<K: UnsignedKey>(entries: &mut [Entry<K>]) {
    for i in 1..entries.len() {
        let e = entries[i];
        let mut j = i;
        while j > 0 && entries[j - 1].key > e.key {
            entries[j] = entries[j - 1];
            j -= 1;
        }
        entries[j] = e;
    }
}

/// Sort `entries`, which are in position order, stably by key. `scratch`
/// is resized to match and may come back holding the other half of the
/// ping-pong.
fn sort_entries<K: UnsignedKey>(entries: &mut Vec<Entry<K>>, scratch: &mut Vec<Entry<K>>) {
    let len = entries.len();
    let mut hist = [[0usize; 256]; 8];
    for e in entries.iter() {
        for (byte, h) in hist.iter_mut().enumerate().take(K::BYTES) {
            h[e.key.radix_digit(byte) as usize] += 1;
        }
    }
    // A byte varies unless the first key's bucket holds every key.
    let first = entries[0].key;
    let mut live = [0usize; 8];
    let mut n_live = 0;
    for byte in 0..K::BYTES {
        if hist[byte][first.radix_digit(byte) as usize] < len {
            live[n_live] = byte;
            n_live += 1;
        }
    }
    let skip = divert::skippable_digits(n_live, len, |i| {
        hist[live[i]].iter().copied().max().unwrap_or(0)
    });

    scratch.clear();
    scratch.resize(
        len,
        Entry {
            key: K::from_digit(0),
            pos: 0,
        },
    );
    for &byte in &live[skip..n_live] {
        let mut offsets = hist[byte];
        prefix_sum::exclusive_prefix_sum(&mut offsets);
        for e in entries.iter() {
            let digit = e.key.radix_digit(byte) as usize;
            scratch[offsets[digit]] = *e;
            offsets[digit] += 1;
        }
        core::mem::swap(entries, scratch);
    }

    if skip > 0 {
        // Runs that agree on every sorted byte are still in position order;
        // sorting each by key and position finishes them stably.
        let shift = 8 * live[skip] as u32;
        let top = |e: &Entry<K>| e.key.to_u64() >> shift;
        let mut start = 0;
        for i in 1..=len {
            if i == len || top(&entries[i]) != top(&entries[start]) {
                if i - start > 1 {
                    entries[start..i].sort_unstable_by_key(|e| (e.key, e.pos));
                }
                start = i;
            }
        }
    }
}

/// Rearrange `slice` so that position `i` holds the element that was at
/// `entries[i].pos`, through `moved`, which stays empty.
fn move_into_order<T, K>(slice: &mut [T], entries: &[Entry<K>], moved: &mut Vec<T>) {
    let len = slice.len();
    debug_assert_eq!(entries.len(), len);
    debug_assert!(moved.is_empty());
    moved.reserve(len);
    let src = slice.as_mut_ptr();
    let dst = moved.as_mut_ptr();
    // SAFETY: the positions are a permutation of 0..len (one entry was made
    // per element, and sorting only permutes entries), so each element is
    // read once into `moved`'s spare capacity, and copying it back leaves
    // every element in `slice` exactly once. Nothing between the first read
    // and the last write can panic. `moved` keeps length 0, so it never owns
    // or drops an element.
    unsafe {
        for (i, e) in entries.iter().enumerate() {
            core::ptr::copy_nonoverlapping(src.add(e.pos as usize), dst.add(i), 1);
        }
        core::ptr::copy_nonoverlapping(dst, src, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Sort `(key, position)` pairs by key and check the result against the
    /// standard library's stable sort.
    fn check<K: SortableKey + core::fmt::Debug>(keys: Vec<K>) {
        let mut items: Vec<(K, usize)> = keys.into_iter().zip(0..).collect();
        let mut expected = items.clone();
        expected.sort_by_key(|&(k, _)| k.to_radix_key());
        sort_by_key(&mut items, |&(k, _)| k);
        let positions = |v: &[(K, usize)]| v.iter().map(|&(_, i)| i).collect::<Vec<_>>();
        assert_eq!(
            positions(&items),
            positions(&expected),
            "len={}",
            items.len()
        );
    }

    #[test]
    fn matches_the_standard_stable_sort() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        // Miri runs the lengths around the threshold only.
        let lens: &[usize] = if cfg!(miri) {
            &[0, 1, 2, 127, 128, 129, 300]
        } else {
            &[0, 1, 2, 127, 128, 129, 1000, 5000]
        };
        for &len in lens {
            // Few distinct keys, so stability shows.
            check(
                (0..len)
                    .map(|_| (xorshift(&mut state) % 7) as u32)
                    .collect(),
            );
            check((0..len).map(|_| xorshift(&mut state) as u32).collect());
            check((0..len).map(|_| xorshift(&mut state) as i8).collect());
            check(
                (0..len)
                    .map(|_| (xorshift(&mut state) % 5) as i16 - 2)
                    .collect(),
            );
            check((0..len).map(|_| xorshift(&mut state) as i64).collect());
            check(
                (0..len)
                    .map(|_| f64::from_bits(xorshift(&mut state)))
                    .collect(),
            );
            check(
                (0..len)
                    .map(|_| [-1.5f32, -0.0, 0.0, 2.0][(xorshift(&mut state) % 4) as usize])
                    .collect(),
            );
        }
    }

    #[test]
    fn wide_keys_divert_and_finish_stably() {
        // Random 64-bit keys divert after their top bytes; so do keys whose
        // top bytes are random and whose low bytes take few values, which
        // leaves runs of equal top bytes holding equal keys.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let lens: &[usize] = if cfg!(miri) {
            &[300]
        } else {
            &[300, 3000, 70_000]
        };
        for &len in lens {
            check((0..len).map(|_| xorshift(&mut state)).collect::<Vec<u64>>());
            check(
                (0..len)
                    .map(|_| (xorshift(&mut state) & !0xFFFF) | (xorshift(&mut state) % 3))
                    .collect::<Vec<u64>>(),
            );
            check(
                (0..len)
                    .map(|_| (xorshift(&mut state) % 9) << 40)
                    .collect::<Vec<u64>>(),
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 70,000 keys; the other tests cover the same code
    fn long_runs_after_diverting_stay_stable() {
        // 200 keys share their top bytes among otherwise random keys, so the
        // radix sort still diverts and leaves them in one run, long enough
        // that only a stable finish keeps their three values in order.
        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut keys: Vec<u64> = (0..70_000).map(|_| xorshift(&mut state)).collect();
        let len = keys.len() as u64;
        for _ in 0..200 {
            let i = (xorshift(&mut state) % len) as usize;
            keys[i] = 0xABCD_EF01_2345_0000 | (xorshift(&mut state) % 3);
        }
        check(keys);
    }

    #[test]
    fn ordered_inputs() {
        // Ascending input is left alone; descending input with duplicates
        // must keep equal keys in order, not reverse them.
        let len = 2000u32;
        check((0..len).map(|i| i / 3).collect());
        check((0..len).rev().map(|i| i / 3).collect());
        check((0..len).rev().collect());
        check(vec![7u64; 1000]);
    }

    #[test]
    fn elements_need_not_be_copy() {
        let mut state = 0x0F0F_0F0F_1234_5678u64;
        let mut items: Vec<(alloc::string::String, u32)> = (0..600)
            .map(|i| {
                let k = (xorshift(&mut state) % 50) as u32;
                (alloc::format!("item {i}"), k)
            })
            .collect();
        let mut expected = items.clone();
        expected.sort_by_key(|(_, k)| *k);
        sort_by_key(&mut items, |(_, k)| *k);
        assert_eq!(items, expected);

        // The same through one buffer, on both sides of RADIX_MIN.
        let mut buffer = SortByKeyBuffer::new();
        for len in [50, 100, 600] {
            let mut items: Vec<(alloc::string::String, u32)> = (0..len)
                .map(|i| {
                    let k = (xorshift(&mut state) % 20) as u32;
                    (alloc::format!("item {i}"), k)
                })
                .collect();
            let mut expected = items.clone();
            expected.sort_by_key(|(_, k)| *k);
            sort_by_key_with_buffer(&mut items, &mut buffer, |(_, k)| *k);
            assert_eq!(items, expected, "len={len}");
        }
    }

    #[test]
    fn one_buffer_across_lengths_and_orders() {
        // Lengths that grow and shrink through one buffer, across the
        // insertion and radix paths, with random, few-valued, ascending and
        // descending keys.
        let mut state = 0x5DEE_CE66_D1CE_4E5Bu64;
        let mut buffer = SortByKeyBuffer::new();
        let lens: &[usize] = if cfg!(miri) {
            &[0, 1, 2, 3, 40, 63, 64, 300, 5]
        } else {
            &[0, 1, 2, 3, 40, 63, 64, 127, 128, 3000, 5, 70_000, 200]
        };
        for &len in lens {
            for case in 0..4 {
                let keys: Vec<u64> = (0..len)
                    .map(|i| match case {
                        0 => xorshift(&mut state),
                        1 => xorshift(&mut state) % 5,
                        2 => i as u64 / 3,
                        _ => (len - i) as u64 / 3,
                    })
                    .collect();
                let mut items: Vec<(u64, usize)> = keys.into_iter().zip(0..).collect();
                let mut expected = items.clone();
                expected.sort_by_key(|&(k, _)| k);
                sort_by_key_with_buffer(&mut items, &mut buffer, |&(k, _)| k);
                assert_eq!(items, expected, "len={len} case={case}");
            }
        }
        let longest = lens.iter().copied().max().unwrap();
        assert!(buffer.capacity() >= longest);
    }
}
