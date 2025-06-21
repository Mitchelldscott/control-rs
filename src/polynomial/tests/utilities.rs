//! Tests for the calculus functions of polynomial. Also includes roots and companion tests, even
//! though those aren't really calculus.

use super::*;

mod derivative {
    use super::*;
    use utils::differentiate;
    #[test]
    fn emtpy() {
        assert_eq!(differentiate(&[0_i16]), [0_i16; 0]);
    }
    #[test]
    fn zero() {
        let zero: Polynomial<u8, 1> = Polynomial::zero();
        assert_eq!(zero.derivative(), Err(DerivativeEdgeCase::DerivativeOfZero));
    }
    #[test]
    fn one() {
        let p = Line::<i8>::identity();
        assert_eq!(p.derivative(), Err(DerivativeEdgeCase::Zero));
    }
    #[test]
    fn lots_of_zeros() {
        let p = Quartic::<f32>::zero();
        assert_eq!(p.derivative(), Err(DerivativeEdgeCase::DerivativeOfZero));
    }
    #[test]
    fn lots_of_zeros_with_constant() {
        let p: Polynomial<f32, 10> = Polynomial::identity();
        assert_eq!(p.derivative(), Err(DerivativeEdgeCase::Zero));
    }
    #[test]
    fn line() {
        let p = Polynomial::from_data([1, 1]);
        assert_eq!(p.derivative(), Ok(Polynomial::from_data([1])));
    }
    #[test]
    fn quadratic() {
        let p = Quadratic::monomial(1.0);
        assert_eq!(p.derivative(), Ok(Polynomial::from_data([0.0, 2.0])));
    }
    #[test]
    fn cubic() {
        let p = Cubic::monomial(1.0);
        assert_eq!(p.derivative(), Ok(Polynomial::from_data([0.0, 0.0, 3.0])));
    }
}

mod integrals {
    use super::*;
    use utils::integrate;
    #[test]
    fn empty() {
        // passing an emtpy array will return a single element array with the constant
        assert_eq!(integrate(&[], 1i8), [1]);
    }
    #[test]
    fn zero() {
        // integrating an array of zeros will return an array of zeros with the constant at index 0
        assert_eq!(integrate(&[0, 0, 0], 1usize), [1, 0, 0, 0]);
    }
    #[test]
    fn one() {
        assert_eq!(integrate(&[1, 0, 0], 1), [1, 1, 0, 0])
    }
    #[test]
    fn quadratic() {
        assert_eq!(integrate(&[0, 0, 3], 1), [1, 0, 0, 1])
    }
    #[test]
    fn cubic() {
        assert_eq!(integrate(&[1, 2, 3, 4], 0), [0, 1, 1, 1, 1])
    }
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
        assert_eq!(companion(&[3.0, 2.0, 0.0]), [[0.0, 0.0], [0.0, 0.0]]);
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
    use crate::assert_f32_eq;
    use crate::polynomial::utils::NoRoots;
    use utils::roots;

    #[test]
    fn zero() {
        assert_eq!(roots(&[0.0, 0.0]), Err(NoRoots));
    }
    #[test]
    fn one() {
        assert_eq!(roots(&[1.0, 0.0]), Err(NoRoots));
    }

    #[test]
    fn line() {
        assert_eq!(roots(&[0.0, 1.0]), Ok([Complex::new(0.0, 0.0)]));
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
}
