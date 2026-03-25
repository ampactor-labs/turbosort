//! Parallel LSD radix sort using rayon.
//!
//! Achieves near-linear multicore scaling for arrays >131K elements.
//!
//! # Algorithm
//!
//! 1. **Histogram:** Parallel per-chunk scan (chunks are independent, read-only).
//! 2. **Global prefix sum:** Compute per-thread scatter offsets.
//! 3. **Parallel scatter:** Each thread writes its chunk to disjoint positions
//!    in the output buffer (no synchronization needed).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use rayon::prelude::*;

use crate::key::{SortableKey, UnsignedKey};
use crate::radix::{histogram, prefix_sum};

/// Minimum array size before parallel sort kicks in.
const PARALLEL_THRESHOLD: usize = 131_072;

/// Sort a slice using parallel LSD radix sort.
///
/// Falls back to single-threaded radix sort for arrays below 131K elements.
pub fn sort<T: SortableKey + Send + Sync>(slice: &mut [T])
where
    T::Key: Send + Sync,
{
    if slice.len() < PARALLEL_THRESHOLD {
        crate::radix::sort(slice);
        return;
    }

    // SAFETY: T is one of the 10 primitive numeric types (sealed by SortableKey);
    // all-zero bytes is a valid representation for all of them.
    let mut buffer = vec![unsafe { core::mem::zeroed() }; slice.len()];
    sort_with_buffer(slice, &mut buffer);
}

/// Sort a slice using parallel LSD radix sort with a caller-provided buffer.
pub fn sort_with_buffer<T: SortableKey + Send + Sync>(slice: &mut [T], buffer: &mut [T])
where
    T::Key: Send + Sync,
{
    assert!(buffer.len() >= slice.len());
    let len = slice.len();
    if len <= 1 {
        return;
    }

    let passes = T::Key::BYTES;
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = len.div_ceil(num_threads);

    // Step 1: compute global histograms for all passes in a single scan.
    // Global histograms are pass-independent (digit counts don't change with
    // reordering), so we compute them once up front like the serial version.
    let global_hist: Vec<usize> = {
        let chunk_hists: Vec<Vec<usize>> = (0..num_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(len);
                let mut hist = vec![0usize; passes * 256];
                for elem in &slice[start..end] {
                    let key = elem.to_radix_key();
                    for pass in 0..passes {
                        let digit = key.radix_digit(pass) as usize;
                        hist[pass * 256 + digit] += 1;
                    }
                }
                hist
            })
            .collect();

        let mut global = vec![0usize; passes * 256];
        for ch in &chunk_hists {
            for (g, c) in global.iter_mut().zip(ch.iter()) {
                *g += c;
            }
        }
        global
    };

    // Step 2-3: for each pass, compute per-chunk histograms from current source,
    // derive scatter offsets, and scatter in parallel.
    let mut in_buffer = false;

    for pass in 0..passes {
        let gh = &global_hist[pass * 256..(pass + 1) * 256];

        if histogram::is_pass_trivial(gh, len) {
            continue;
        }

        // Determine current source for this pass
        let src: &[T] = if in_buffer { buffer } else { slice };

        // Per-chunk histograms must be recomputed each pass because the
        // previous scatter rearranged elements across chunks.
        let chunk_hists: Vec<[usize; 256]> = (0..num_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(len);
                let mut hist = [0usize; 256];
                for elem in &src[start..end] {
                    let digit = elem.to_radix_key().radix_digit(pass) as usize;
                    hist[digit] += 1;
                }
                hist
            })
            .collect();

        // Compute per-chunk scatter offsets from global prefix sum
        let mut global_offsets = [0usize; 256];
        global_offsets.copy_from_slice(gh);
        prefix_sum::exclusive_prefix_sum(&mut global_offsets);

        let mut chunk_offsets: Vec<[usize; 256]> = Vec::with_capacity(num_threads);
        let mut running = global_offsets;
        for ch in &chunk_hists {
            chunk_offsets.push(running);
            for d in 0..256 {
                running[d] += ch[d];
            }
        }

        // Encode pointers as usize to cross thread boundary (raw ptrs aren't Send)
        let (src_addr, dst_addr) = if in_buffer {
            (buffer.as_ptr() as usize, slice.as_mut_ptr() as usize)
        } else {
            (slice.as_ptr() as usize, buffer.as_mut_ptr() as usize)
        };

        // SAFETY: Each chunk writes to disjoint positions in dst (guaranteed by
        // the prefix sum partitioning). src is only read. Both pointers are valid
        // for the duration of the scope. usize encoding is safe because rayon::scope
        // joins all threads before returning.
        rayon::scope(|s| {
            for (t, offsets) in chunk_offsets.into_iter().enumerate() {
                let start = t * chunk_size;
                let end = (start + chunk_size).min(len);
                s.spawn(move |_| {
                    let src_ptr = src_addr as *const T;
                    let dst_ptr = dst_addr as *mut T;
                    let mut off = offsets;
                    for i in start..end {
                        let elem = unsafe { *src_ptr.add(i) };
                        let digit = elem.to_radix_key().radix_digit(pass) as usize;
                        let pos = off[digit];
                        unsafe { *dst_ptr.add(pos) = elem };
                        off[digit] = pos + 1;
                    }
                });
            }
        });

        in_buffer = !in_buffer;
    }

    if in_buffer {
        slice.copy_from_slice(buffer);
    }
}
