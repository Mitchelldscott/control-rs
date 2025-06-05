//! Tests for constructors and accessors of the basic polynomial type.
//!
//! This file provides basic usage testing, initialization, accessors and setters, for
//! specialized polynomial types. The specializations involve varying the underlying array data
//! type `T` and the capacity of the array `N`. The tests are written in a way that they can be
//! re-used for all specializations. The types should cover numbers and the sizes should not exceed
//! 4096 bytes.
//!
// TODO:
//     * constructors
//         * `new(coefficients; [T; N])`
//         * `compose(f: Polynomial, g: Polynomial) -> Polynomial`
//         * `to_monic(&self) -> Self`
//     * accessors
//         * `coefficient()`
//         * `coefficient_mut()`
//         * `constant()`
//         * `constant_mut()`
//         * `leading_coefficient()`
//         * `leading_coefficient_mut()`
//         * `resize(polynomial: Polynomial)`

use crate::polynomial::{Constant, Line, Polynomial};

mod neg {
    use super::*;
    #[test]
    fn empty_neg() {
        let empty = Polynomial::from_data([0.0f32; 0]);
        assert_eq!(-empty, empty);
    }
    #[test]
    fn constant_neg() {
        let constant = Polynomial::monomial(1.0f64);
        assert_eq!(-constant, Polynomial::from_data([-1.0f64]));
    }
    fn line_neg() {
        let line = Polynomial::from_data([0, 1]);
        assert_eq!(-line, Polynomial::from_data([0, -1]));
    }
    #[test]
    fn quadratic_neg() {
        let quadratic = Polynomial::new([1, 0, 0]);
        assert_eq!(-quadratic, Polynomial::from_data([0, 0, -1]));
    }
    #[test]
    fn large_neg() {
        let polynomial: Polynomial<i8, 16> = Polynomial::from_element(1);
        assert_eq!(-polynomial, Polynomial::from_element(-1));
    }
}

mod coefficient_accessors {
    use super::*;
    #[test]
    fn empty() {
        let p = Polynomial::new([1i32; 0]);
        assert_eq!(p.is_empty(), true, "not empty");
        assert_eq!(p.degree(), None, "degree not none");
        assert_eq!(p.is_monic(), false, "monic");
        assert_eq!(p.coefficient(0), None, "coefficient(0) not none");
        assert_eq!(p.coefficient(1), None, "coefficient(1) not none");
        assert_eq!(p.constant(), None, "constant not none");
        assert_eq!(
            p.leading_coefficient(),
            None,
            "leading_coefficient not none"
        );
    }
    
    #[test]
    fn empty_mut() {
        let mut p_mut = Polynomial::from_data([1i32; 0]);
        assert_eq!(p_mut.is_empty(), true, "mut not empty");
        assert_eq!(p_mut.degree(), None, "degree not none");
        assert_eq!(p_mut.is_monic(), false, "monic");
        assert_eq!(
            p_mut.coefficient_mut(0),
            None,
            "coefficient_mut(0) not none"
        );
        assert_eq!(
            p_mut.coefficient_mut(1),
            None,
            "coefficient_mut(1) not none"
        );
        assert_eq!(p_mut.constant_mut(), None, "constant_mut not none");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            None,
            "leading_coefficient_mut not none"
        );
    }
    #[test]
    fn constant() {
        let p = Constant::from_element(10i32);
        assert_eq!(p.is_empty(), false, "p is empty");
        assert_eq!(p.degree(), Some(0), "degree");
        assert_eq!(p.is_monic(), false, "monic");
        assert_eq!(p.coefficient(0), Some(&1), "coefficient(0)");
        assert_eq!(p.coefficient(1), None, "coefficient(1) not none");
        assert_eq!(p.constant(), Some(&1), "constant");
        assert_eq!(p.leading_coefficient(), Some(&1), "leading_coefficient");
        let mut p_mut = Constant::from_iterator([1i32]);
        assert_eq!(p_mut.is_empty(), false, "mut not empty");
        assert_eq!(p_mut.degree(), Some(0), "degree");
        assert_eq!(p_mut.is_monic(), true, "monic");
        assert_eq!(p_mut.coefficient_mut(0), Some(&mut 1), "coefficient_mut(0)");
        assert_eq!(
            p_mut.coefficient_mut(1),
            None,
            "coefficient_mut(1) not none"
        );
        assert_eq!(p_mut.constant_mut(), Some(&mut 1), "constant_mut");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 1),
            "leading_coefficient_mut"
        );
    }
    #[test]
    fn line() {
        let p = Line::from_fn(|_| 1i32);
        assert_eq!(p.is_empty(), false, "p is empty");
        assert_eq!(p.degree(), Some(1), "degree");
        assert_eq!(p.is_monic(), true, "monic");
        assert_eq!(p.coefficient(0), Some(&1), "coefficient(0)");
        assert_eq!(p.coefficient(1), Some(&1), "coefficient(1)");
        assert_eq!(p.coefficient(2), None, "coefficient(2)");
        assert_eq!(p.constant(), Some(&1), "constant");
        assert_eq!(p.leading_coefficient(), Some(&1), "leading_coefficient");
        let mut p_mut = Line::from_iterator([1i32]); // y = 0*x + 1
        assert_eq!(p_mut.is_empty(), false, "mut not empty");
        assert_eq!(p_mut.degree(), Some(1), "degree");
        assert_eq!(p_mut.is_monic(), false, "monic");
        assert_eq!(p_mut.coefficient_mut(0), Some(&mut 1), "coefficient_mut(0)");
        assert_eq!(p_mut.coefficient_mut(1), Some(&mut 0), "coefficient_mut(1)");
        assert_eq!(
            p_mut.coefficient_mut(2),
            None,
            "coefficient_mut(2) not none"
        );
        assert_eq!(p_mut.constant_mut(), Some(&mut 1), "constant_mut");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 0),
            "leading_coefficient_mut"
        );
    }
    fn large() {
        let p = Polynomial::<f32, 24>::from_fn(|_| 1.0);
        assert_eq!(p.is_empty(), false, "p is empty");
        assert_eq!(p.degree(), Some(23), "degree");
        assert_eq!(p.is_monic(), true, "monic");
        assert_eq!(p.coefficient(0), Some(&1.0), "coefficient(0)");
        assert_eq!(p.coefficient(1), Some(&1.0), "coefficient(1)");
        assert_eq!(p.coefficient(2), Some(&1.0), "coefficient(2)");
        assert_eq!(p.coefficient(23), Some(&1.0), "coefficient(23)");
        assert_eq!(p.coefficient(24), None, "coefficient(24)");
        assert_eq!(p.constant(), Some(&1.0), "constant");
        assert_eq!(p.leading_coefficient(), Some(&1.0), "leading_coefficient");
        let mut p_mut = Polynomial::<f64, 24>::from_iterator([1.0]); // y = 0*x + 1
        assert_eq!(p_mut.is_empty(), false, "mut not empty");
        assert_eq!(p_mut.degree(), Some(23), "degree");
        assert_eq!(p_mut.is_monic(), false, "monic");
        assert_eq!(
            p_mut.coefficient_mut(0),
            Some(&mut 1.0),
            "coefficient_mut(0)"
        );
        assert_eq!(
            p_mut.coefficient_mut(1),
            Some(&mut 0.0),
            "coefficient_mut(1)"
        );
        assert_eq!(
            p_mut.coefficient_mut(2),
            Some(&mut 0.0),
            "coefficient_mut(2)"
        );
        assert_eq!(
            p_mut.coefficient_mut(23),
            Some(&mut 0.0),
            "coefficient_mut(23)"
        );
        assert_eq!(p_mut.coefficient_mut(24), None, "coefficient_mut(24)");
        assert_eq!(p_mut.constant_mut(), Some(&mut 1.0), "constant_mut");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 0.0),
            "leading_coefficient_mut"
        );
    }
}
