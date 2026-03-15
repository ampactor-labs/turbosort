//! Permutation lookup tables for AVX2 compress-store emulation.
//!
//! AVX-512 has native `vpcompressd` but AVX2 does not. We emulate it with
//! a 256-entry LUT that maps an 8-bit mask to a permutation vector for
//! `_mm256_permutevar8x32_epi32`.
//!
//! Each mask bit indicates whether the corresponding lane should be packed
//! to the left (bit=1) or right (bit=0). The LUT entry gives the permutation
//! that moves selected lanes to contiguous positions on the left.
//!
//! Total size: 256 × 8 × 4 = 8KB — fits comfortably in L1 cache.

/// Permutation indices for compress-store on 8×u32 lanes.
///
/// `COMPRESS_LUT[mask]` contains 8 lane indices that, when used with
/// `_mm256_permutevar8x32_epi32`, packs the lanes where `mask` has a 1-bit
/// to the left side of the vector.
///
/// The number of elements packed left equals `mask.count_ones()`.
pub static COMPRESS_LUT: [[i32; 8]; 256] = {
    let mut lut = [[0i32; 8]; 256];
    let mut mask: usize = 0;
    while mask < 256 {
        let mut left = 0usize;
        let mut right = 7usize;
        let mut perm = [0i32; 8];
        let mut lane = 0usize;
        while lane < 8 {
            if (mask >> lane) & 1 == 1 {
                perm[left] = lane as i32;
                left += 1;
            } else {
                perm[right] = lane as i32;
                // Avoid underflow on last iteration
                if right > 0 {
                    right -= 1;
                }
            }
            lane += 1;
        }
        lut[mask] = perm;
        mask += 1;
    }
    lut
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_all_selected() {
        // mask 0xFF: all lanes selected → identity permutation
        let perm = COMPRESS_LUT[0xFF];
        assert_eq!(perm, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn lut_none_selected() {
        // mask 0x00: no lanes selected → all go right
        let perm = COMPRESS_LUT[0x00];
        // Left side empty, right side has all lanes (reversed order from right-packing)
        assert_eq!(perm[0..8].iter().copied().collect::<Vec<_>>().len(), 8);
    }

    #[test]
    fn lut_single_lane() {
        // mask 0b00000100: only lane 2 selected
        let perm = COMPRESS_LUT[0x04];
        assert_eq!(perm[0], 2, "lane 2 should be packed to position 0");
    }

    #[test]
    fn lut_popcount_consistency() {
        // For every mask, the first popcount(mask) entries should be the selected lanes in order
        for mask in 0..256usize {
            let perm = COMPRESS_LUT[mask];
            let popcount = mask.count_ones() as usize;
            let mut selected = Vec::new();
            for lane in 0..8 {
                if (mask >> lane) & 1 == 1 {
                    selected.push(lane as i32);
                }
            }
            assert_eq!(
                &perm[..popcount],
                &selected[..],
                "mask {mask:#010b}: first {popcount} entries should be selected lanes in order"
            );
        }
    }
}
