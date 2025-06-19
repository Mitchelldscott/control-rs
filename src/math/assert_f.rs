//! Macros for float comparisons
//!
//! Based on [assert_float_eq](https://docs.rs/assert_float_eq/latest/assert_float_eq/index.html)

/// better assert
#[macro_export]
macro_rules! assert_f64_eq {
    ($a:expr, $b:expr) => {{
        assert_f64_eq!($a, $b, f64::EPSILON)
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b): (f64, f64) = ($a, $b);
        let epsilon = $eps;

        #[allow(clippy::float_cmp)]
        let is_equal = if a == b {
            // Handles infinities and exact equality.
            true
        } else if a == 0.0_f64 || b == 0.0_f64 || (a - b).abs() < f64::MIN_POSITIVE {
            // One of a or b is zero (or both are extremely close to it).
            // Use a scaled absolute error check.
            (a - b).abs() < (epsilon * f64::MIN_POSITIVE)
        } else {
            // Use relative error.
            // The denominator is clamped to f64::MAX to avoid overflow.
            (a - b).abs() / (a.abs() + b.abs()).min(f64::MAX) < epsilon
        };
        assert!(
            is_equal,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, approx diff: `{:?}` > {:?})",
            a,
            b,
            (a - b).abs(),
            epsilon
        );
    }};
}

/// better assert
#[macro_export]
macro_rules! assert_f32_eq {
    ($a:expr, $b:expr) => {{
        assert_f32_eq!($a, $b, f32::EPSILON)
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b): (f32, f32) = ($a, $b);
        let epsilon = $eps;

        #[allow(clippy::float_cmp)]
        let is_equal = if a == b {
            // Handles infinities and exact equality.
            true
        } else if a == 0.0 || b == 0.0 || (a - b).abs() < f32::MIN_POSITIVE {
            // One of a or b is zero (or both are extremely close to it).
            // Use a scaled absolute error check.
            (a - b).abs() < (epsilon * f32::MIN_POSITIVE)
        } else {
            // Use relative error.
            // The denominator is clamped to f64::MAX to avoid overflow.
            (a - b).abs() / (a.abs() - b.abs()).min(f32::MAX) < epsilon
        };
        assert!(
            is_equal,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, approx diff: `{:?}` > {:?})",
            a,
            b,
            (a - b).abs(),
            epsilon
        );
    }};
}

#[test]
fn it_should_not_panic_if_values_are_approx_equal() {
    assert_f32_eq!(64f32.sqrt(), 8f32);
}

#[test]
#[should_panic(expected = "assertion failed: `(left !== right)")]
fn it_should_panic_if_values_are_not_approx_equal() {
    assert_f32_eq!(3f32, 4f32);
}

#[test]
fn compare_with_explicit_eps() {
    assert_f64_eq!(3f64, 4f64, 2f64);
}

#[test]
#[should_panic(expected = "assertion failed: `(left !== right)` (left: `3.0`, right: `4.0`")]
fn bad_compare_with_explicit_eps() {
    assert_f64_eq!(3f64, 4f64, 1e-3f64);
}

// Make sure the value used for epsilon in the assert_eq
// is the same as the value used in the error message.
#[test]
#[should_panic(
    expected = "assertion failed: `(left !== right)` (left: `0.0`, right: `1.5`, approx diff: `1.5` > 1.0)"
)]
fn should_evaluate_eps_only_once() {
    let mut count = 0_f64;
    // `count` will be 1.0 the first time the curly-braced block
    // is evaluated but 2.0 the second time.
    assert_f64_eq!(0_f64, 1.5_f64, {
        count += 1_f64;
        count
    });
}
