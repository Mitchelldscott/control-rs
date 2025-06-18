//! Tests for the calculus functions of polynomial. Also includes roots and companion tests, even
//! though those aren't really calculus.

use super::*;

mod derivative {
    use super::*;
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
}

mod integral {
    use super::*;
    #[test]
    fn empty() {
        let empty: Polynomial<i16, 0> = Polynomial::zero();
        assert_eq!(empty.integral(1), Constant::identity());
    }
    #[test]
    fn zero() {
        let zero: Polynomial<u8, 1> = Polynomial::zero();
        assert_eq!(zero.integral(1), Line::identity());
    }
    #[test]
    fn one() {
        let one = Cubic::identity();
        assert_eq!(
            one.integral(1.0),
            Quartic::<f64>::identity() + Line::monomial(1.0)
        );
    }

    #[test]
    fn quadratic() {
        let p = Quadratic::monomial(3_i32);
        assert_eq!(p.integral(-1), Cubic::monomial(1) - Constant::identity());
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
        // The leading coefficient is 2.0.
        // First row: [-8/2, -6/2, -4/2] = [-4.0, -3.0, -2.0]
        assert_eq!(
            companion(&[8.0, 6.0, 4.0, 2.0]),
            [[-2.0, -3.0, -4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn monic_quadratic() {
        // Polynomial: x^2 + 5x + 10
        // Coefficients: [10, 5, 1] -> N = 3, M = 2
        assert_eq!(companion(&[10.0, 5.0, 1.0]), [[-5.0, -10.0], [1.0, 0.0]]);
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
        // First row: [1/2, -2/2, 1/1] = [0, 1, 0]
        let result = companion(&[1, -2, 1, 2]);
        assert_eq!(result, [[0, 1, 0], [1, 0, 0], [0, 1, 0]]);
    }
}
