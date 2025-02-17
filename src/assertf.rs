//!
//! Macros for float comparisons

/// better assert
#[macro_export]
macro_rules! assert_f64 {
    (eq, $a:expr, $b:expr) => {{
        assert_f64!(eq, $a, $b, f64::epsilon())
    }};
    (eq, $a:expr, $b:expr, $eps:expr) => {{
        use num_traits::{Float, Zero};

        let (a, b) = (&$a, &$b);
        let abs_a = f64::abs(*a);
        let abs_b = f64::abs(*b);
        let diff = f64::abs(*a - *b);
        let epsilon = $eps;

        let eq = {
            if a == b {
                // Handle infinities.
                true
            } else if *a == f64::zero() || *b == f64::zero() || diff < f64::min_positive_value() {
                // One of a or b is zero (or both are extremely close to it,) use absolute error.
                diff < (epsilon * f64::min_positive_value())
            } else {
                // Use relative error.
                (diff / f64::min(abs_a + abs_b, <f64 as num_traits::Float>::max_value())) < epsilon
            }
        };
        assert!(
            eq,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, approx diff: `{:?}` > {:?})",
            *a,
            *b,
            num_traits::Float::abs(*a - *b),
            epsilon
        );
    }};
}

/// better assert
#[macro_export]
macro_rules! assert_f32 {
    (eq, $a:expr, $b:expr) => {{
        use num_traits::real::Real;
        assert_f32!(eq, $a, $b, f32::epsilon())
    }};
    (eq, $a:expr, $b:expr, $eps:expr) => {{
        use num_traits::Zero;
        let (a, b) = (&$a, &$b);
        let abs_a = f32::abs(*a);
        let abs_b = f32::abs(*b);
        let diff = f32::abs(*a - *b);
        let epsilon = $eps;

        let eq = {
            if a == b {
                // Handle infinities.
                true
            } else if *a == f32::zero() || *b == f32::zero() || diff < f32::min_positive_value() {
                // One of a or b is zero (or both are extremely close to it,) use absolute error.
                diff < (epsilon * f32::min_positive_value())
            } else {
                // Use relative error.
                (diff / f32::min(abs_a + abs_b, <f32 as num_traits::Float>::max_value())) < epsilon
            }
        };
        assert!(
            eq,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, approx diff: `{:?}` > {:?})",
            *a,
            *b,
            num_traits::Float::abs(*a - *b),
            epsilon
        );
    }};
}

#[test]
fn it_should_not_panic_if_values_are_approx_equal() {
    assert_f32!(eq, 64f32.sqrt(), 8f32);
}

#[test]
#[should_panic]
fn it_should_panic_if_values_are_not_approx_equal() {
    assert_f32!(eq, 3 as f32, 4 as f32);
}

#[test]
fn compare_with_explicit_eps() {
    assert_f64!(eq, 3f64, 4f64, 2f64);
}

#[test]
#[should_panic]
fn bad_compare_with_explicit_eps() {
    assert_f64!(eq, 3f64, 4f64, 1e-3f64);
}

// Make sure the value used for epsilon in the assert_eq
// is the same as the value used in the error message.
#[test]
#[should_panic(expected = "approx diff: `100.0` > 1.0")]
fn should_evaluate_eps_only_once() {
    let mut count = 0_f64;

    // `count` will be 1.0 the first time the curly-braced block
    // is evaluated but 2.0 the second time.
    assert_f64!(eq, 0_f64, 100_f64, {
        count += 1_f64;
        count
    });
}
