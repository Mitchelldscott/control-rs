//! Tests for the calculus functions of polynomial. Also includes roots and companion tests, even
//! though those aren't really calculus.

use super::*;

mod largest_nonzero_index {
    use super::utils::largest_nonzero_index;
    #[test]
    fn empty() { assert_eq!(largest_nonzero_index(&[0u8; 0]), None); }
    #[test]
    fn zeros() { assert_eq!(largest_nonzero_index(&[0; 10]), None); }
    #[test]
    fn leading_zeros() { assert_eq!(largest_nonzero_index(&[1, 0, 0, 0]), Some(0)); }
    #[test]
    fn one() { assert_eq!(largest_nonzero_index(&[1, 1]), Some(1)); }
}

mod derivative {
    use super::utils::differentiate;
    #[test]
    fn emtpy() {
        assert_eq!(differentiate([0_i16]), [0_i16; 0]);
    }
    #[test]
    fn zero() {
        assert_eq!(differentiate([0_i16, 0_i16]), [0_i16; 1]);
    }
    #[test]
    fn one() {
        assert_eq!(differentiate([1_i8]), [0_i8; 0]);
    }
    #[test]
    fn lots_of_zeros() {
        assert_eq!(differentiate([0; 15]), [0; 14]);
    }
    #[test]
    fn lots_of_zeros_with_constant() {
        assert_eq!(differentiate([1, 0, 0, 0, 0]), [0; 4]);
    }
    #[test]
    fn line() {
        assert_eq!(differentiate([1, 2]), [2]);
    }
    #[test]
    fn quadratic() {
        assert_eq!(differentiate([1, 1, 1]), [1, 2]);
    }
    // #[test] // The array is too large and violates the trait bounds
    // fn too_large_array() {
    //     assert_eq!(differentiate([0i8; i8::MAX as usize + 1]), [0i8; i8::MAX as usize]);
    // }
}

mod integrals{
    use super::utils::integrate;
    #[test]
    fn empty() { assert_eq!(integrate([], 1i8), [1]); }
    #[test]
    fn zeros() { assert_eq!(integrate([0, 0, 0], 1usize), [1, 0, 0, 0]); }
    #[test]
    fn one() {
        assert_eq!(integrate([1], 0), [0, 1]);
    }
    #[test]
    fn quadratic() {
        assert_eq!(integrate([0, 0, 3], 1), [1, 0, 0, 1]);
    }
    #[test]
    fn cubic() { assert_eq!(integrate([1, 2, 3, 4], 0), [0, 1, 1, 1, 1]); }
    // #[test] // The array is too large and violates the trait bounds
    // fn too_large_array() {
    //     assert_eq!(integrate([0i8; i8::MAX as usize], 0), [0i8; i8::MAX as usize + 1]);
    // }
}

mod companion {
    use super::utils::unchecked_companion;
    #[test]
    fn quadratic() {
        // Polynomial: x^2 + 2x + 3
        // Coefficients: [3, 2, 1] (constant, x^1, x^2) -> N = 3, M = 2
        assert_eq!(unchecked_companion(&[3.0, 2.0, 1.0]), [[-2.0, -3.0], [1.0, 0.0]]);
    }

    #[test]
    fn cubic() {
        // Polynomial: 2x^3 + 4x^2 + 6x + 8
        // Coefficients: [8, 6, 4, 2] -> N = 4, M = 3
        assert_eq!(
            unchecked_companion(&[8.0, 6.0, 4.0, 2.0]),
            [[-2.0, -3.0, -4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn monic_quadratic() {
        // Polynomial: x^2 + 5x + 10
        // Coefficients: [10, 5, 1] -> N = 3, M = 2
        assert_eq!(
            unchecked_companion(&[10_i8, 5_i8, 1_i8]),
            [[-5_i8, -10_i8], [1_i8, 0_i8]]
        );
    }

    #[test]
    fn constant() {
        // If N=1, M=0, the matrix is `[[T; 0]; 0]`, which is an empty array of empty arrays.
        assert_eq!(unchecked_companion(&[5.0]), [[0.0f64; 0]; 0]);
    }

    #[test]
    #[should_panic = "attempt to divide by zero"]
    fn leading_zero_int() {
        // Polynomial: 0x^2 + 2x + 3 (Effectively 2x + 3)
        // Coefficients: [3.0, 2.0, 0.0] -> N = 3, M = 2
        assert_eq!(unchecked_companion(&[3, 2, 0]), [[0, 0], [1, 0]]);
    }
    #[test]
    fn leading_zero_float() {
        // Polynomial: 0x^2 + 2x + 3 (Effectively 2x + 3)
        // Coefficients: [3.0, 2.0, 0.0] -> N = 3, M = 2
        assert_eq!(
            unchecked_companion(&[3.0, 2.0, 0.0]),
            [[-f64::INFINITY, -f64::INFINITY], [1.0, 0.0]]
        );
    }

    #[test]
    fn integer_cubic() {
        // Polynomial: 2x^3 + x^2 - 2x + 1
        // Coefficients: [1, -2, 1, 2] -> N = 4, M = 3
        assert_eq!(unchecked_companion(&[1, -2, 1, 2]), [[0, 1, 0], [1, 0, 0], [0, 1, 0]]);
    }
}

mod roots {
    use super::*;
    use crate::{
        assert_f32_eq, assert_f64_eq,
        polynomial::utils::{
            unchecked_roots,
            RootFindingError,
        },
    };
    #[test]
    fn zero() {
        assert_eq!(unchecked_roots(&[0.0, 0.0]), Err(RootFindingError::DegeneratePolynomial));
    }
    #[test]
    fn one() {
        assert_eq!(unchecked_roots(&[1.0, 0.0]), Err(RootFindingError::NoSolution));
    }
    #[test]
    fn zero_slope() { assert_eq!(
            unchecked_roots(&[0.0, 0.0]),
            Err(RootFindingError::DegeneratePolynomial)
        );
    }
    #[test]
    fn nan_slope() { assert_eq!(
            unchecked_roots(&[f64::NAN, 0.0]),
            Err(RootFindingError::NoSolution)
        );
    }
    #[test]
    fn inf_slope() {
        assert_eq!(
            unchecked_roots(&[f64::INFINITY, 0.0]),
            Err(RootFindingError::NoSolution)
        );
    }
    #[test]
    fn line() {
        assert_eq!(unchecked_roots(&[1.0, 1.0]), Ok([Complex::new(-1.0, 0.0)]));
    }

    #[test]
    fn quadratic_real() {
        assert_eq!(
            unchecked_roots(&[0.0, 0.0, 1.0]),
            Ok([Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)])
        );
    }

    #[test]
    fn quadratic_imaginary() {
        assert_eq!(
            unchecked_roots(&[1.0, 0.0, 1.0]),
            Ok([Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)])
        );
    }

    #[allow(clippy::panic)]
    #[allow(clippy::unwrap_used)]
    #[allow(clippy::expect_used)]
    #[test]
    fn cubic_imaginary() {
        let unchecked_roots = unchecked_roots(&[1.0, 1.0, 1.0, 1.0]).expect("failed to compute unchecked_roots");
        let mut complex1 = None;
        let mut complex2 = None;
        for root in &unchecked_roots {
            if root.im.is_zero() {
                assert_f32_eq!(root.re, -1.0);
            } else if complex1.is_none() {
                complex1 = Some(root);
            } else if complex2.is_none() {
                complex2 = Some(root);
            } else {
                panic!("More than 2 complex unchecked_roots");
            }
        }
        let complex1 = complex1.unwrap();
        let complex2 = complex2.unwrap();
        assert_f32_eq!(complex1.re, complex2.re);
        assert_f32_eq!(complex1.im, complex2.im.neg());
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn quartic_monomial() {
        let unchecked_roots = unchecked_roots(&[0.0, 0.0, 0.0, 0.0, 1.0]);
        for root in unchecked_roots.expect("failed to compute unchecked_roots") {
            assert_f64_eq!(root.re, 0.0);
            assert_f64_eq!(root.im, 0.0);
        }
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn line_with_leading_zeros() {
        let unchecked_roots = unchecked_roots(&[0.0, 1.0, 0.0, 0.0]).expect("failed to compute unchecked_roots");
        assert_f32_eq!(unchecked_roots[0].re, 0.0);
        assert_f32_eq!(unchecked_roots[0].im, 0.0);
        assert!(unchecked_roots[1].re.is_nan());
        assert!(unchecked_roots[1].im.is_nan());
        assert!(unchecked_roots[2].re.is_nan());
        assert!(unchecked_roots[2].im.is_nan());
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn cubic_with_zero_constant() {
        let unchecked_roots = unchecked_roots(&[0.0, 1.0, 1.0, 1.0]).expect("failed to compute unchecked_roots");
        assert_f32_eq!(unchecked_roots[0].re, -0.5);
        assert_f32_eq!(unchecked_roots[0].im, 0.866_025_4);
        assert_f32_eq!(unchecked_roots[1].re, -0.5);
        assert_f32_eq!(unchecked_roots[1].im, -0.866_025_4);
        assert_f32_eq!(unchecked_roots[2].re, 0.0);
        assert_f32_eq!(unchecked_roots[2].im, 0.0);
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn sixth_order_zero_constant() {
        let unchecked_roots = unchecked_roots(&[
            0.0,
            0.000_027_940_1,
            0.000_028_772_7,
            0.000_009_786_5,
            0.000_001_341_6,
            0.000_000_078_2,
            0.000_000_001_6,
            0.0,
        ])
        .expect("failed to compute unchecked_roots");
        assert_f32_eq!(unchecked_roots[0].re, -20.7682, 0.03);
        assert_f32_eq!(unchecked_roots[1].re, -13.0648, 0.05);
        assert_f32_eq!(unchecked_roots[2].re, -9.7396, 0.02);
        assert_f32_eq!(unchecked_roots[3].re, -3.3001, 0.00025);
        assert_f32_eq!(unchecked_roots[4].re, -2.0024, 1e-4);
        assert_f32_eq!(unchecked_roots[5].re, 0.0);
        assert!(unchecked_roots[6].re.is_nan() && unchecked_roots[6].im.is_nan());
        assert_f32_eq!(
            unchecked_roots[0].im
            + unchecked_roots[1].im
            + unchecked_roots[2].im
            + unchecked_roots[3].im
            + unchecked_roots[4].im
            + unchecked_roots[5].im,
            0.0
        );
    }
}

mod add {
    use super::utils::unchecked_polynomial_add;
    #[test]
    fn add_shorter() { assert_eq!(unchecked_polynomial_add([1, 1], [1]), [2, 1]); }
    #[test]
    fn add_longer() { assert_eq!(unchecked_polynomial_add([1], [1, 1]), [2, 1]); }
    #[should_panic = "attempt to add with overflow"]
    #[test]
    fn add_overflow() { assert_eq!(unchecked_polynomial_add([1, 1], [u32::MAX]), [1, 1]); }
    #[should_panic = "attempt to add with overflow"]
    #[test]
    fn add_underflow() { assert_eq!(unchecked_polynomial_add([i8::MIN], [i8::MIN]), [0]); }
}

mod sub {
    use super::utils::unchecked_polynomial_sub;
    #[test]
    fn sub_shorter() { assert_eq!(unchecked_polynomial_sub([1, 1], [1]), [0, 1]); }
    #[test]
    fn sub_longer() { assert_eq!(unchecked_polynomial_sub([1], [1, 1]), [0, -1]); }
    #[should_panic = "attempt to subtract with overflow"]
    #[test]
    fn sub_overflow() { assert_eq!(unchecked_polynomial_sub([i8::MAX], [i8::MIN]), [0]); }
    #[should_panic = "attempt to subtract with overflow"]
    #[test]
    fn sub_underflow() { assert_eq!(unchecked_polynomial_sub([i8::MIN], [i8::MAX]), [0]); }
}

mod convolution {
    use super::utils::convolution;
    #[test]
    fn mul_shorter() { assert_eq!(convolution(&[1, 1], &[0]), [0, 0]); }
    #[test]
    fn mul_longer() { assert_eq!(convolution(&[1, 1], &[1, 1]), [1, 2, 1]); }
    #[should_panic = "attempt to multiply with overflow"]
    #[test]
    fn mul_overflow() { assert_eq!(convolution(&[i8::MAX], &[i8::MAX]), [0]); }
    #[should_panic = "attempt to add with overflow"]
    #[test]
    fn mul_add_overflow() { assert_eq!(convolution(&[1, 1], &[(u8::MAX/2)+1, (u8::MAX/2)+1]), [u8::MAX/2, u8::MAX, u8::MAX/2]); }
}