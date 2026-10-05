//! Parallel LSD radix sort using rayon.
//!
//! # Algorithm
//!
//! 1. **Global histograms:** one parallel scan computes every pass's digit
//!    counts (counts are order-independent). The per-chunk counts from this
//!    scan double as the first pass's scatter offsets, so the first pass
//!    needs no separate histogram read.
//! 2. **Per pass:** derive per-chunk scatter offsets from the global prefix
//!    sums, then scatter in parallel — each chunk writes disjoint destination
//!    ranges, so no synchronization is needed. Later passes re-scan their
//!    source for per-chunk counts (the previous scatter moved elements across
//!    chunk boundaries).
//!
//! A fused variant that accumulated next-pass histograms during the scatter
//! was measured and rejected: the extra per-element accumulation into a
//! chunks-by-bins table competes with the scatter's random writes for cache
//! and loses more than the saved sequential read is worth.
//!
//! Eight-byte keys use 11-bit digits (six passes) like the serial wide path;
//! smaller keys use byte digits. As in the serial cores, digits that never
//! vary are not counted, and when the top digits alone nearly order the keys
//! the low passes are replaced by a finishing scan (see `radix::divert`),
//! here split across threads at run boundaries.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use rayon::prelude::*;

use crate::key::{SortableKey, UnsignedKey};
use crate::radix::{divert, histogram, prefix_sum};

/// Keys the serial head of the varying-bits scan reads before the rest, if
/// still needed, is scanned in parallel.
const SERIAL_SCAN: usize = 4096;

/// Minimum array size before parallel sort kicks in.
const PARALLEL_THRESHOLD: usize = 131_072;

/// Sort a slice using parallel LSD radix sort.
///
/// Below 131,072 elements this is the serial [`crate::sort`].
pub fn sort<T: SortableKey + Send + Sync>(slice: &mut [T])
where
    T::Key: Send + Sync,
{
    if slice.len() < PARALLEL_THRESHOLD {
        crate::dispatch::sort(slice);
        return;
    }
    if crate::presorted::finish_presorted(slice) {
        return;
    }
    if T::Key::BYTES == 1 {
        crate::radix::counting_sort(slice);
        return;
    }

    let mut buffer: Vec<T> = Vec::with_capacity(slice.len());
    // SAFETY: the Vec's capacity is valid for slice.len() writes; T is Copy so
    // nothing needs dropping, and the Vec's length stays 0.
    unsafe {
        match T::Key::BYTES {
            2 => sort_core::<T, 256, 2>(slice, buffer.as_mut_ptr()),
            4 => sort_core::<T, 256, 4>(slice, buffer.as_mut_ptr()),
            8 => sort_core::<T, 2048, 6>(slice, buffer.as_mut_ptr()),
            _ => unreachable!("SortableKey key widths are 1, 2, 4, or 8 bytes"),
        }
    }
}

/// Parallel radix passes over `BINS`-ary digits.
///
/// # Safety
///
/// `scratch` must be valid for `slice.len()` writes and must not alias
/// `slice`. It may be uninitialized — every pass writes all positions.
unsafe fn sort_core<T, const BINS: usize, const PASSES: usize>(slice: &mut [T], scratch: *mut T)
where
    T: SortableKey + Send + Sync,
    T::Key: Send + Sync,
{
    let len = slice.len();
    let bits = BINS.trailing_zeros();
    let mask = BINS - 1;
    let chunk_size = len.div_ceil(rayon::current_num_threads().max(1));
    let num_chunks = len.div_ceil(chunk_size);
    let chunk_range = |t: usize| (t * chunk_size, ((t + 1) * chunk_size).min(len));

    // Step 0: which digits vary at all. Random keys settle it within the
    // serial head; otherwise the rest of the scan runs in parallel.
    let head = len.min(SERIAL_SCAN);
    let varying = histogram::varying_bits(&slice[..head]).map(|head_bits| {
        let first = slice[0].to_radix_key().to_u64();
        head_bits
            | slice[head..]
                .par_chunks(chunk_size)
                .map(|c| {
                    c.iter()
                        .fold(0, |acc, x| acc | (x.to_radix_key().to_u64() ^ first))
                })
                .reduce(|| 0, |a, b| a | b)
    });
    let counted: Vec<usize> = (0..PASSES)
        .filter(|&p| varying.map_or(true, |v| (v >> (p as u32 * bits)) & mask as u64 != 0))
        .collect();

    // Step 1: per-chunk histograms of the counted digits, in one parallel scan
    // of the original data. Summed they give the global (order-independent)
    // counts; sliced per pass they are the first pass's per-chunk counts.
    let all_pass_hists: Vec<Vec<usize>> = (0..num_chunks)
        .into_par_iter()
        .map(|t| {
            let (start, end) = chunk_range(t);
            let mut hist = vec![0usize; PASSES * BINS];
            if counted.len() == PASSES {
                // Every digit varies: the fixed-count loop unrolls.
                for elem in &slice[start..end] {
                    let key = elem.to_radix_key();
                    for pass in 0..PASSES {
                        hist[pass * BINS + key.wide_digit(pass as u32 * bits, mask)] += 1;
                    }
                }
            } else {
                for elem in &slice[start..end] {
                    let key = elem.to_radix_key();
                    for &pass in &counted {
                        hist[pass * BINS + key.wide_digit(pass as u32 * bits, mask)] += 1;
                    }
                }
            }
            hist
        })
        .collect();

    let mut global = vec![0usize; PASSES * BINS];
    for ch in &all_pass_hists {
        for (g, c) in global.iter_mut().zip(ch.iter()) {
            *g += c;
        }
    }

    // Passes where the digit actually varies; the rest are no-op scatters.
    let live: Vec<usize> = counted
        .iter()
        .copied()
        .filter(|&p| !histogram::is_pass_trivial(&global[p * BINS..(p + 1) * BINS], len))
        .collect();
    let skip = divert::skippable_digits(live.len(), len, |i| {
        let pass = live[i];
        global[pass * BINS..(pass + 1) * BINS]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    });
    let passes = &live[skip..];

    let first = match passes.first() {
        Some(&p) => p,
        None => return,
    };
    let mut chunk_hists: Vec<[usize; BINS]> = all_pass_hists
        .iter()
        .map(|h| {
            let mut arr = [0usize; BINS];
            arr.copy_from_slice(&h[first * BINS..(first + 1) * BINS]);
            arr
        })
        .collect();
    drop(all_pass_hists);

    let mut in_scratch = false;

    for (li, &pass) in passes.iter().enumerate() {
        let shift = pass as u32 * bits;

        let (src_addr, dst_addr) = if in_scratch {
            (scratch as usize, slice.as_mut_ptr() as usize)
        } else {
            (slice.as_ptr() as usize, scratch as usize)
        };

        // Later passes: per-chunk counts of the current source (the previous
        // scatter moved elements across chunk boundaries).
        if li > 0 {
            // SAFETY: src points at len initialized elements (the previous
            // scatter wrote every position); chunks are disjoint reads.
            chunk_hists = (0..num_chunks)
                .into_par_iter()
                .map(|t| {
                    let src = src_addr as *const T;
                    let (start, end) = chunk_range(t);
                    let mut hist = [0usize; BINS];
                    for i in start..end {
                        let key = unsafe { *src.add(i) }.to_radix_key();
                        hist[key.wide_digit(shift, mask)] += 1;
                    }
                    hist
                })
                .collect();
        }

        // Per-chunk scatter offsets: global exclusive prefix sum, advanced by
        // the chunks before this one.
        let mut running = [0usize; BINS];
        running.copy_from_slice(&global[pass * BINS..(pass + 1) * BINS]);
        prefix_sum::exclusive_prefix_sum(&mut running);
        let mut chunk_offsets: Vec<[usize; BINS]> = Vec::with_capacity(num_chunks);
        for ch in &chunk_hists {
            chunk_offsets.push(running);
            for (r, c) in running.iter_mut().zip(ch.iter()) {
                *r += c;
            }
        }

        // SAFETY: the per-chunk offset tables partition 0..len, so writers
        // touch disjoint destination positions; src is only read. Both
        // pointers outlive the parallel iterator, which joins before this
        // scope ends.
        chunk_offsets
            .into_par_iter()
            .enumerate()
            .for_each(|(t, mut off)| {
                let src = src_addr as *const T;
                let dst = dst_addr as *mut T;
                let (start, end) = chunk_range(t);
                for i in start..end {
                    let elem = unsafe { *src.add(i) };
                    let digit = elem.to_radix_key().wide_digit(shift, mask);
                    let pos = off[digit];
                    unsafe { *dst.add(pos) = elem };
                    off[digit] = pos + 1;
                }
            });

        in_scratch = !in_scratch;
    }

    if in_scratch {
        let src_addr = scratch as usize;
        let dst_addr = slice.as_mut_ptr() as usize;
        // SAFETY: chunks are disjoint; scratch is fully initialized (the last
        // scatter wrote every position).
        (0..num_chunks).into_par_iter().for_each(|t| {
            let (start, end) = chunk_range(t);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (src_addr as *const T).add(start),
                    (dst_addr as *mut T).add(start),
                    end - start,
                );
            }
        });
    }

    if skip > 0 {
        finish_runs_parallel(slice, first as u32 * bits, num_chunks);
    }
}

/// [`divert::finish_runs`] across threads: cut `slice` into about `parts`
/// pieces at run boundaries, so no run is split, and finish each piece.
fn finish_runs_parallel<T: SortableKey + Send>(slice: &mut [T], shift: u32, parts: usize) {
    let len = slice.len();
    let prefix = |x: &T| x.to_radix_key().wide_digit(shift, usize::MAX);
    let mut cuts = Vec::with_capacity(parts + 1);
    cuts.push(0);
    for t in 1..parts {
        let mut cut = (t * len / parts).max(cuts[t - 1]);
        while cut > 0 && cut < len && prefix(&slice[cut]) == prefix(&slice[cut - 1]) {
            cut += 1;
        }
        cuts.push(cut);
    }
    cuts.push(len);

    let mut pieces = Vec::with_capacity(parts);
    let mut rest = slice;
    for w in cuts.windows(2) {
        let (piece, tail) = rest.split_at_mut(w[1] - w[0]);
        pieces.push(piece);
        rest = tail;
    }
    pieces
        .into_par_iter()
        .for_each(|piece| divert::finish_runs(piece, shift));
}
