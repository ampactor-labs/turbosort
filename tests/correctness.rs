use proptest::prelude::*;
use turbosort::SortableKey;

// --- Helpers ---

fn reference_sort<T: SortableKey>(data: &mut [T]) {
    data.sort_by_key(|a| a.to_radix_key());
}

// --- Exhaustive tests for small n ---

fn exhaustive_permutations<T: SortableKey + From<u8> + std::fmt::Debug + PartialEq>(max_n: usize) {
    for n in 0..=max_n {
        let base: Vec<T> = (0..n as u8).map(T::from).collect();
        let mut perm = base.clone();
        loop {
            let mut test = perm.clone();
            turbosort::sort(&mut test);
            let mut expected = perm.clone();
            reference_sort(&mut expected);
            assert_eq!(test, expected, "failed on permutation {:?}", perm);
            if !next_permutation(&mut perm) {
                break;
            }
        }
    }
}

fn next_permutation<T: SortableKey>(data: &mut [T]) -> bool {
    let len = data.len();
    if len < 2 {
        return false;
    }
    let mut i = len - 1;
    while i > 0 && data[i - 1].to_radix_key() >= data[i].to_radix_key() {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = len - 1;
    while data[j].to_radix_key() <= data[i - 1].to_radix_key() {
        j -= 1;
    }
    data.swap(i - 1, j);
    data[i..].reverse();
    true
}

#[test]
fn exhaustive_u8() {
    exhaustive_permutations::<u8>(8);
}

#[test]
fn exhaustive_i8() {
    // Use values that span negative and positive
    for n in 0..=6 {
        let base: Vec<i8> = (-3..(-3 + n as i8)).collect();
        let mut perm = base.clone();
        loop {
            let mut test = perm.clone();
            turbosort::sort(&mut test);
            let mut expected = perm.clone();
            reference_sort(&mut expected);
            assert_eq!(test, expected, "i8 failed on {:?}", perm);
            if !next_permutation(&mut perm) {
                break;
            }
        }
    }
}

// --- Boundary size tests ---

#[test]
fn boundary_sizes_u32() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    for size in [0, 1, 2, 3, 15, 16, 17, 128, 511, 512, 513, 1024, 4096] {
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort();
        turbosort::sort(&mut data);
        assert_eq!(data, expected, "failed at size {size}");
    }
}

#[test]
fn boundary_sizes_with_buffer() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(99);
    for size in [0, 1, 2, 16, 17, 512, 513, 2048] {
        let mut data: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        let mut buf = vec![0u32; size];
        let mut expected = data.clone();
        expected.sort();
        turbosort::sort_with_buffer(&mut data, &mut buf);
        assert_eq!(data, expected, "sort_with_buffer failed at size {size}");
    }
}

/// `sort_with_buffer` stays on the allocation-free byte-digit path for 8-byte
/// keys, even at sizes where `sort()` would switch to the wide-digit path.
/// This exercises that byte-8 core at scale.
#[test]
fn large_u64_with_buffer() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(2024);
    for size in [70_000usize, 200_000] {
        let mut data: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let mut buf = vec![0u64; size];
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_with_buffer(&mut data, &mut buf);
        assert_eq!(data, expected, "sort_with_buffer u64 failed at size {size}");
    }
}

/// Structured 64-bit keys at sizes where `sort` takes the 11-bit path and
/// `sort_with_buffer` the byte path, both diverting.
#[test]
fn large_u64_structured() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(77);
    for size in [70_000usize, 200_000] {
        let shapes: [(&str, Vec<u64>); 6] = [
            ("random", (0..size).map(|_| rng.gen()).collect()),
            (
                "low 40 bits",
                (0..size).map(|_| rng.gen::<u64>() >> 24).collect(),
            ),
            (
                "top bit split",
                (0..size)
                    .map(|_| (rng.gen::<u64>() & 1) << 63 | rng.gen::<u64>() >> 8)
                    .collect(),
            ),
            (
                "correlated top bytes",
                (0..size)
                    .map(|_| {
                        let b = rng.gen::<u64>() & 0xFF;
                        b << 56 | b << 48 | b << 40 | rng.gen::<u64>() >> 24
                    })
                    .collect(),
            ),
            (
                "1000 distinct",
                (0..size)
                    .map(|_| rng.gen_range(0..1000u64) * 0x9E37_79B9_7F4A_7C15)
                    .collect(),
            ),
            (
                "heavy tail",
                (0..size)
                    .map(|_| (1.0 / (rng.gen::<f64>() + 1e-9)).powf(1.5) as u64)
                    .collect(),
            ),
        ];
        for (shape, data) in shapes {
            let mut expected = data.clone();
            expected.sort_unstable();

            let mut sorted = data.clone();
            turbosort::sort(&mut sorted);
            assert_eq!(sorted, expected, "sort, {shape}, size {size}");

            let mut sorted = data;
            let mut buf = vec![0u64; size];
            turbosort::sort_with_buffer(&mut sorted, &mut buf);
            assert_eq!(sorted, expected, "sort_with_buffer, {shape}, size {size}");
        }
    }
}

// --- Float edge cases ---

#[test]
fn float_edge_cases_f32() {
    let mut data = vec![
        1.0f32,
        -1.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    turbosort::sort(&mut data);

    // Verify ordering: -inf < MIN(-3.4e38) < -1 < -min_pos(-1.2e-38) < -0 < 0 < min_pos < 1 < MAX < inf < NaN
    assert_eq!(data[0], f32::NEG_INFINITY);
    assert_eq!(data[1], f32::MIN);
    assert_eq!(data[2], -1.0);
    assert_eq!(data[3], -f32::MIN_POSITIVE);
    assert!(data[4].to_bits() == (-0.0f32).to_bits(), "expected -0.0");
    assert!(
        data[5] == 0.0 && !data[5].is_sign_negative(),
        "expected +0.0"
    );
    assert_eq!(data[6], f32::MIN_POSITIVE);
    assert_eq!(data[7], 1.0);
    assert_eq!(data[8], f32::MAX);
    assert_eq!(data[9], f32::INFINITY);
    assert!(data[10].is_nan());
}

#[test]
fn float_edge_cases_f64() {
    let mut data = vec![
        1.0f64,
        -1.0,
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    turbosort::sort(&mut data);

    assert_eq!(data[0], f64::NEG_INFINITY);
    assert_eq!(data[1], -1.0);
    assert!(data[2].to_bits() == (-0.0f64).to_bits());
    assert!(data[3] == 0.0 && !data[3].is_sign_negative());
    assert_eq!(data[4], 1.0);
    assert_eq!(data[5], f64::INFINITY);
    assert!(data[6].is_nan());
}

// --- Signed edge cases ---

#[test]
fn signed_edge_cases() {
    let mut data = vec![i32::MAX, i32::MIN, 0, -1, 1];
    turbosort::sort(&mut data);
    assert_eq!(data, vec![i32::MIN, -1, 0, 1, i32::MAX]);

    let mut data = vec![i64::MAX, i64::MIN, 0, -1, 1];
    turbosort::sort(&mut data);
    assert_eq!(data, vec![i64::MIN, -1, 0, 1, i64::MAX]);

    let mut data = vec![i8::MAX, i8::MIN, 0, -1, 1];
    turbosort::sort(&mut data);
    assert_eq!(data, vec![i8::MIN, -1, 0, 1, i8::MAX]);

    let mut data = vec![i16::MAX, i16::MIN, 0, -1, 1];
    turbosort::sort(&mut data);
    assert_eq!(data, vec![i16::MIN, -1, 0, 1, i16::MAX]);
}

// --- All-equal ---

#[test]
fn all_equal() {
    let mut data = vec![42u32; 1000];
    turbosort::sort(&mut data);
    assert!(data.iter().all(|&x| x == 42));
}

// --- Already sorted / reverse sorted ---

#[test]
fn already_sorted() {
    let mut data: Vec<u32> = (0..1000).collect();
    let expected = data.clone();
    turbosort::sort(&mut data);
    assert_eq!(data, expected);
}

#[test]
fn reverse_sorted() {
    let mut data: Vec<u32> = (0..1000).rev().collect();
    let mut expected = data.clone();
    expected.sort();
    turbosort::sort(&mut data);
    assert_eq!(data, expected);
}

// --- Presorted input, across every tier ---

const PRESORTED_SIZES: [usize; 8] = [17, 100, 128, 129, 512, 513, 5000, 200_000];

/// Non-increasing input with runs of equal keys gets reversed, not sorted.
#[test]
fn reverse_with_duplicates() {
    for n in PRESORTED_SIZES {
        let mut data: Vec<u32> = (0..n as u32).rev().map(|i| i / 3).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort(&mut data);
        assert_eq!(data, expected, "u32 n={n}");

        let mut data: Vec<f64> = (0..n).rev().map(|i| (i / 3) as f64 - 50.0).collect();
        turbosort::sort(&mut data);
        assert!(data.windows(2).all(|w| w[0] <= w[1]), "f64 n={n}");

        let mut data: Vec<i16> = (0..n).rev().map(|i| (i % 60_000) as i16).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort(&mut data);
        assert_eq!(data, expected, "i16 n={n}");
    }
}

/// Sorted except for the very end: the scan must not stop early and call
/// the slice sorted.
#[test]
fn sorted_except_the_tail() {
    for n in PRESORTED_SIZES {
        let mut data: Vec<u64> = (0..n as u64).collect();
        data.swap(n - 2, n - 1);
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort(&mut data);
        assert_eq!(data, expected, "ascending n={n}");

        let mut data: Vec<u64> = (0..n as u64).rev().collect();
        data[n - 1] = u64::MAX;
        let mut expected = data.clone();
        expected.sort_unstable();
        let mut buf = vec![0u64; n];
        turbosort::sort_with_buffer(&mut data, &mut buf);
        assert_eq!(data, expected, "descending n={n}");
    }
}

// --- Proptest ---

macro_rules! proptest_sort {
    ($name:ident, $ty:ty) => {
        mod $name {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(2000))]

                #[test]
                fn small(mut data in proptest::collection::vec(any::<$ty>(), 0..=16)) {
                    let mut expected = data.clone();
                    reference_sort(&mut expected);
                    turbosort::sort(&mut data);
                    prop_assert_eq!(&data, &expected);
                }

                #[test]
                fn medium(mut data in proptest::collection::vec(any::<$ty>(), 17..=512)) {
                    let mut expected = data.clone();
                    reference_sort(&mut expected);
                    turbosort::sort(&mut data);
                    prop_assert_eq!(&data, &expected);
                }

                #[test]
                fn large(mut data in proptest::collection::vec(any::<$ty>(), 513..=10_000)) {
                    let mut expected = data.clone();
                    reference_sort(&mut expected);
                    turbosort::sort(&mut data);
                    prop_assert_eq!(&data, &expected);
                }
            }
        }
    };
}

// Wide keys with structure the uniform strategies never produce: a varying
// number of high bits cleared, and few distinct values. Both change how many
// digits the radix sort diverts on.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn structured_u64(
        shift in 0u32..60,
        modulus in 1u64..2000,
        dup in any::<bool>(),
        mut data in proptest::collection::vec(any::<u64>(), 513..=6000),
    ) {
        for x in data.iter_mut() {
            *x = if dup { *x % modulus * 0x9E37_79B9_7F4A_7C15 } else { *x >> shift };
        }
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort(&mut data);
        prop_assert_eq!(&data, &expected);
    }

    #[test]
    fn structured_f64(
        scale in 0i32..300,
        data in proptest::collection::vec(any::<u64>(), 513..=6000),
    ) {
        let mut floats: Vec<f64> = data
            .iter()
            .map(|x| (*x as i64) as f64 * 2f64.powi(-scale))
            .collect();
        let mut expected = floats.clone();
        reference_sort(&mut expected);
        turbosort::sort(&mut floats);
        prop_assert_eq!(
            floats.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
    }
}

proptest_sort!(prop_u8, u8);
proptest_sort!(prop_u16, u16);
proptest_sort!(prop_u32, u32);
proptest_sort!(prop_u64, u64);
proptest_sort!(prop_i8, i8);
proptest_sort!(prop_i16, i16);
proptest_sort!(prop_i32, i32);
proptest_sort!(prop_i64, i64);
proptest_sort!(prop_f32, f32);
proptest_sort!(prop_f64, f64);

// --- Adversarial patterns ---

fn check_sorted_u32(input: Vec<u32>) {
    let mut data = input.clone();
    let mut expected = input;
    reference_sort(&mut expected);
    turbosort::sort(&mut data);
    assert_eq!(data, expected);
}

const ADVERSARIAL_SIZES: [usize; 8] = [8, 16, 17, 100, 512, 513, 1000, 200_000];

#[test]
fn adversarial_presorted() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32((0..n as u32).collect());
    }
}

#[test]
fn adversarial_reverse() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32((0..n as u32).rev().collect());
    }
}

#[test]
fn adversarial_pipe_organ() {
    for &n in &ADVERSARIAL_SIZES {
        if n < 2 {
            continue;
        }
        check_sorted_u32(
            (0..n / 2)
                .chain((0..n / 2).rev())
                .map(|x| x as u32)
                .collect(),
        );
    }
}

#[test]
fn adversarial_sawtooth_8() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32((0..n).map(|i| (i % 8) as u32).collect());
    }
}

#[test]
fn adversarial_sawtooth_64() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32((0..n).map(|i| (i % 64) as u32).collect());
    }
}

#[test]
fn adversarial_few_unique() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32((0..n).map(|i| (i % 4) as u32).collect());
    }
}

// --- Parallel path (needs n ≥ 131072 to leave the serial fallback) ---

#[cfg(feature = "parallel")]
mod parallel {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const N: usize = 300_000;

    #[test]
    fn parallel_u32_random() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut data: Vec<u32> = (0..N).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn parallel_u64_random() {
        // Exercises the wide-digit (2048-bin) parallel path.
        let mut rng = StdRng::seed_from_u64(11);
        let mut data: Vec<u64> = (0..N).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn parallel_f32_random() {
        let mut rng = StdRng::seed_from_u64(13);
        let mut data: Vec<f32> = (0..N).map(|_| rng.gen_range(-1e6f32..1e6)).collect();
        let mut expected = data.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        turbosort::sort_parallel(&mut data);
        assert_eq!(
            data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn parallel_few_unique() {
        // Trivial high-digit passes get skipped; the fused histograms must
        // hand off across the skip correctly.
        let mut data: Vec<u32> = (0..N).map(|i| (i % 7) as u32).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn parallel_sorted_and_reverse() {
        let mut data: Vec<u32> = (0..N as u32).collect();
        let expected = data.clone();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);

        let mut data: Vec<u32> = (0..N as u32).rev().collect();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn parallel_reverse_with_duplicates() {
        let mut data: Vec<u64> = (0..N as u64).rev().map(|i| i / 5).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn parallel_small_input_uses_serial_sort() {
        for n in [0usize, 1, 16, 17, 512, 513, 5000] {
            let mut data: Vec<i32> = (0..n as i32).map(|i| (i * 7919) % 1013 - 500).collect();
            let mut expected = data.clone();
            expected.sort_unstable();
            turbosort::sort_parallel(&mut data);
            assert_eq!(data, expected, "n={n}");
        }
    }

    #[test]
    fn parallel_u8_counting() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut data: Vec<u8> = (0..N).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort_unstable();
        turbosort::sort_parallel(&mut data);
        assert_eq!(data, expected);
    }
}

#[test]
fn adversarial_m3_killer() {
    for &n in &ADVERSARIAL_SIZES {
        check_sorted_u32(
            (0..n)
                .map(|i| if i % 2 == 0 { i as u32 } else { (n - i) as u32 })
                .collect(),
        );
    }
}
