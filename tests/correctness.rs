use proptest::prelude::*;
use turbosort::SortableKey;

// --- Helpers ---

fn is_sorted<T: SortableKey>(slice: &[T]) -> bool {
    slice
        .windows(2)
        .all(|w| w[0].to_radix_key() <= w[1].to_radix_key())
}

fn reference_sort<T: SortableKey>(data: &mut [T]) {
    data.sort_by(|a, b| a.to_radix_key().cmp(&b.to_radix_key()));
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
