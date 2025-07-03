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
    // #[test] // should not compile
    // fn too_large_array() { assert_eq!(largest_nonzero_index(&[1u8; usize::MAX+1]), Some(usize::MAX)); }
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
    // fn too_large_array() {assert_eq!(differentiate([0i8; (i8::MAX+1) as usize]), [0i8; i8::MAX]);}
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
    // fn too_large_array() { assert_eq!(integrate([0i8; i8::MAX as usize], 0), [0i8; (i8::MAX+1) as usize]); }
}

mod companion {
    use super::*;
    use utils::companion;
    #[test]
    fn quadratic() {
        // Polynomial: x^2 + 2x + 3
        // Coefficients: [3, 2, 1] (constant, x^1, x^2) -> N = 3, M = 2
        assert_eq!(companion(&[3.0, 2.0, 1.0]), [[-2.0, -3.0], [1.0, 0.0]]);
    }

    #[test]
    fn cubic() {
        // Polynomial: 2x^3 + 4x^2 + 6x + 8
        // Coefficients: [8, 6, 4, 2] -> N = 4, M = 3
        assert_eq!(
            companion(&[8.0, 6.0, 4.0, 2.0]),
            [[-2.0, -3.0, -4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn monic_quadratic() {
        // Polynomial: x^2 + 5x + 10
        // Coefficients: [10, 5, 1] -> N = 3, M = 2
        assert_eq!(
            companion(&[10_i8, 5_i8, 1_i8]),
            [[-5_i8, -10_i8], [1_i8, 0_i8]]
        );
    }

    #[test]
    fn constant() {
        // If N=1, M=0, the matrix is `[[T; 0]; 0]`, which is an empty array of empty arrays.
        assert_eq!(companion(&[5.0]), [[0.0f64; 0]; 0]);
    }

    #[test]
    fn leading_zero() {
        // Polynomial: 0x^2 + 2x + 3 (Effectively 2x + 3)
        // Coefficients: [3.0, 2.0, 0.0] -> N = 3, M = 2
        assert_eq!(companion(&[3.0, 2.0, 0.0]), [[0.0, 0.0], [1.0, 0.0]]);
    }

    #[test]
    fn integer_cubic() {
        // Polynomial: 2x^3 + x^2 - 2x + 1
        // Coefficients: [1, -2, 1, 2] -> N = 4, M = 3
        assert_eq!(companion(&[1, -2, 1, 2]), [[0, 1, 0], [1, 0, 0], [0, 1, 0]]);
    }
}

mod roots {
    use super::*;
    use crate::{
        assert_f32_eq, assert_f64_eq,
        polynomial::utils::{
            roots,
            x_intercept,
            NoRoots
        },
    };
    #[test]
    fn zero() {
        assert_eq!(roots(&[0.0, 0.0]), Err(NoRoots));
    }
    #[test]
    fn one() {
        assert_eq!(roots(&[1.0, 0.0]), Err(NoRoots));
    }
    #[test]
    fn nan_slope() { assert_eq!(x_intercept(f64::NAN, 0.0), Ok(f64::NAN)); }
    #[test]
    fn inf_slope() { assert_eq!(x_intercept(f64::INFINITY, 0.0), Ok(0.0)); }
    #[test]
    fn line() {
        assert_eq!(roots(&[1.0, 1.0]), Ok([Complex::new(-1.0, 0.0)]));
    }

    #[test]
    fn quadratic_real() {
        assert_eq!(
            roots(&[0.0, 0.0, 1.0]),
            Ok([Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)])
        );
    }

    #[test]
    fn quadratic_imaginary() {
        assert_eq!(
            roots(&[1.0, 0.0, 1.0]),
            Ok([Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)])
        );
    }

    #[allow(clippy::panic)]
    #[allow(clippy::unwrap_used)]
    #[allow(clippy::expect_used)]
    #[test]
    fn cubic_imaginary() {
        let roots = roots(&[1.0, 1.0, 1.0, 1.0]).expect("failed to compute roots");
        let mut complex1 = None;
        let mut complex2 = None;
        for root in &roots {
            if root.im.is_zero() {
                assert_f32_eq!(root.re, -1.0);
            } else if complex1.is_none() {
                complex1 = Some(root);
            } else if complex2.is_none() {
                complex2 = Some(root);
            } else {
                panic!("More than 2 complex roots");
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
        let roots = roots(&[0.0, 0.0, 0.0, 0.0, 1.0]);
        for root in roots.expect("failed to compute roots") {
            assert_f64_eq!(root.re, 0.0);
            assert_f64_eq!(root.im, 0.0);
        }
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn line_with_leading_zeros() {
        let roots = roots(&[0.0, 1.0, 0.0, 0.0]).expect("failed to compute roots");
        assert_f32_eq!(roots[0].re, 0.0);
        assert_f32_eq!(roots[0].im, 0.0);
        assert!(roots[1].re.is_nan());
        assert!(roots[1].im.is_nan());
        assert!(roots[2].re.is_nan());
        assert!(roots[2].im.is_nan());
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn cubic_with_zero_constant() {
        let roots = roots(&[0.0, 1.0, 1.0, 1.0]).expect("failed to compute roots");
        assert_f32_eq!(roots[0].re, -0.5);
        assert_f32_eq!(roots[0].im, 0.866_025_4);
        assert_f32_eq!(roots[1].re, -0.5);
        assert_f32_eq!(roots[1].im, -0.866_025_4);
        assert_f32_eq!(roots[2].re, 0.0);
        assert_f32_eq!(roots[2].im, 0.0);
    }
    #[allow(clippy::expect_used)]
    #[test]
    fn sixth_order_zero_constant() {
        let roots = roots(&[
            0.0,
            0.000_027_940_1,
            0.000_028_772_7,
            0.000_009_786_5,
            0.000_001_341_6,
            0.000_000_078_2,
            0.000_000_001_6,
            0.0,
        ])
        .expect("failed to compute roots");
        assert_f32_eq!(roots[0].re, -20.7682, 0.03);
        assert_f32_eq!(roots[1].re, -13.0648, 0.05);
        assert_f32_eq!(roots[2].re, -9.7396, 0.02);
        assert_f32_eq!(roots[3].re, -3.3001, 0.00025);
        assert_f32_eq!(roots[4].re, -2.0024, 1e-4);
        assert_f32_eq!(roots[5].re, 0.0);
        assert!(roots[6].re.is_nan() && roots[6].im.is_nan());
        assert_f32_eq!(
            roots[0].im + roots[1].im + roots[2].im + roots[3].im + roots[4].im + roots[5].im,
            0.0
        );
    }
}
