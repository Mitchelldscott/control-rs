//! Tests for constructors and accessors of the basic polynomial type.
//!
//! This file provides basic usage testing, initialization, accessors and setters for
//! specialized polynomial types.
//!
//! The tests currently cover u8, i8, u16, i16, u32, i32, f32, f64.(neg tests do not cover
//! unsigned integers). 64-bit and up: u64, i64, u128, i128, and other `big_nums` are not covered.
//!
//! ## Methods covered in this module:
//! * `neg`
//! * `coefficient`/`coefficient_mut`
//! * `constant`/`constant_mut`
//! * `degree`
//! * `is_emtpy`
//! * `is_monic`
//! *  `leading_coefficient`/`leading_coefficient_mut`
//! * `from_data`
//! * `from_iterator`
//! * `from_element`
//! * `from_fn`
//! * `new`

use crate::{
    assert_f32_eq, assert_f64_eq,
    polynomial::{Constant, Cubic, Line, Polynomial, Quadratic},
};

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
    #[test]
    fn line_neg() {
        let line = Polynomial::from_data([0i8, 1i8]);
        assert_eq!(-line, Polynomial::from_data([0, -1]));
    }
    #[test]
    fn quadratic_neg() {
        let quadratic = Polynomial::new([1i16, 0i16, 0i16]);
        assert_eq!(-quadratic, Polynomial::from_data([0, 0, -1]));
    }
    #[test]
    fn large_neg() {
        let polynomial: Polynomial<i32, 16> = Polynomial::from_element(1);
        assert_eq!(-polynomial, Polynomial::from_element(-1));
    }
}

mod coefficient_accessors {
    use super::*;
    #[test]
    fn empty() {
        let p = Polynomial::new([1u8; 0]);
        assert!(p.is_empty(), "not empty");
        assert_eq!(p.degree(), None, "degree not none");
        assert!(!p.is_monic(), "monic");
        assert_eq!(p.coefficient(0), None, "coefficient(0) not none");
        assert_eq!(p.coefficient(1), None, "coefficient(1) not none");
        // assert_eq!(p.constant(), None, "constant not none");
        assert_eq!(
            p.leading_coefficient(),
            None,
            "leading_coefficient not none"
        );
    }

    #[test]
    fn empty_mut() {
        let mut p_mut = Polynomial::from_data([1i8; 0]);
        assert!(p_mut.is_empty(), "mut not empty");
        assert_eq!(p_mut.degree(), None, "degree not none");
        assert!(!p_mut.is_monic(), "monic");
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
        // assert_eq!(p_mut.constant_mut(), None, "constant_mut not none");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            None,
            "leading_coefficient_mut not none"
        );
    }
    #[test]
    fn constant() {
        let p = Constant::from_element(10u16);
        assert!(!p.is_empty(), "p is empty");
        assert_eq!(p.degree(), Some(0), "degree");
        assert!(!p.is_monic(), "monic");
        assert_eq!(p.coefficient(0), Some(&10), "coefficient(0)");
        assert_eq!(p.coefficient(1), None, "coefficient(1) not none");
        assert_eq!(p.constant(), &10, "constant");
        assert_eq!(p.leading_coefficient(), Some(&10), "leading_coefficient");
    }
    #[test]
    fn constant_mut() {
        let mut p_mut = Constant::from_iterator([10i16]);
        assert!(!p_mut.is_empty(), "mut not empty");
        assert_eq!(p_mut.degree(), Some(0), "degree");
        assert!(!p_mut.is_monic(), "monic");
        assert_eq!(
            p_mut.coefficient_mut(0),
            Some(&mut 10),
            "coefficient_mut(0)"
        );
        assert_eq!(
            p_mut.coefficient_mut(10),
            None,
            "coefficient_mut(1) not none"
        );
        assert_eq!(p_mut.constant_mut(), &mut 10, "constant_mut");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 10),
            "leading_coefficient_mut"
        );
    }
    #[test]
    fn line() {
        let p = Line::from_fn(|_| 1u32);
        assert!(!p.is_empty(), "p is empty");
        assert_eq!(p.degree(), Some(1), "degree");
        assert!(p.is_monic(), "monic");
        assert_eq!(p.coefficient(0), Some(&1), "coefficient(0)");
        assert_eq!(p.coefficient(1), Some(&1), "coefficient(1)");
        assert_eq!(p.coefficient(2), None, "coefficient(2)");
        assert_eq!(p.constant(), &1, "constant");
        assert_eq!(p.leading_coefficient(), Some(&1), "leading_coefficient");
    }
    #[test]
    fn line_mut() {
        let mut p_mut = Line::new([2i32, 1i32]);
        assert!(!p_mut.is_empty(), "mut not empty");
        assert_eq!(p_mut.degree(), Some(1), "degree");
        assert!(!p_mut.is_monic(), "monic");
        assert_eq!(p_mut.coefficient_mut(0), Some(&mut 1), "coefficient_mut(0)");
        assert_eq!(p_mut.coefficient_mut(1), Some(&mut 2), "coefficient_mut(1)");
        assert_eq!(
            p_mut.coefficient_mut(2),
            None,
            "coefficient_mut(2) not none"
        );
        assert_eq!(p_mut.constant_mut(), &mut 1, "constant_mut");
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 2),
            "leading_coefficient_mut"
        );
    }
    #[test]
    fn large() {
        let p = Polynomial::<f32, 24>::from_fn(|_| 1.0);
        assert!(!p.is_empty(), "p is empty");
        assert_eq!(p.degree(), Some(23), "degree");
        assert!(p.is_monic(), "monic");
        assert_eq!(p.coefficient(0), Some(&1.0), "coefficient(0)");
        assert_eq!(p.coefficient(1), Some(&1.0), "coefficient(1)");
        assert_eq!(p.coefficient(2), Some(&1.0), "coefficient(2)");
        assert_eq!(p.coefficient(23), Some(&1.0), "coefficient(23)");
        assert_eq!(p.coefficient(24), None, "coefficient(24)");
        assert_f32_eq!(*p.constant(), 1.0);
        assert_eq!(p.leading_coefficient(), Some(&1.0), "leading_coefficient");
    }
    #[test]
    fn large_mut() {
        let mut p_mut = Polynomial::<f64, 16>::from_iterator([1.0]);
        assert!(!p_mut.is_empty(), "mut not empty");
        assert_eq!(p_mut.degree(), Some(0), "degree");
        assert!(p_mut.is_monic(), "monic");
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
            p_mut.coefficient_mut(15),
            Some(&mut 0.0),
            "coefficient_mut(15)"
        );
        assert_eq!(p_mut.coefficient_mut(16), None, "coefficient_mut(16)");
        assert_f64_eq!(*p_mut.constant_mut(), 1.0);
        assert_eq!(
            p_mut.leading_coefficient_mut(),
            Some(&mut 1.0),
            "leading_coefficient_mut"
        );
    }
}

mod constructors {
    use super::*;
    use crate::static_storage::array_from_iterator_with_default;
    #[test]
    fn new_polynomial() {
        let incremental_array: [i32; 10] = array_from_iterator_with_default(0..10, 0);
        let p = Polynomial::new(incremental_array);
        for (i, a_i) in p.into_iter().rev().enumerate() {
            assert_eq!(a_i, incremental_array[i]);
        }
    }
    #[test]
    fn polynomial_from_element() {
        let p: Polynomial<i32, 10> = Polynomial::from_element(1i32);
        for a_i in p {
            assert_eq!(a_i, 1i32);
        }
    }
    #[test]
    fn polynomial_from_fn() {
        let p: Polynomial<usize, 5> = Polynomial::from_fn(|i| i * 2);
        for (i, a_i) in p.into_iter().enumerate() {
            assert_eq!(a_i, i * 2);
        }
    }
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn polynomial_from_iterator_exact() {
        let p: Polynomial<i32, 1000> = Polynomial::from_iterator(0..1000);
        for (i, a_i) in p.into_iter().enumerate() {
            assert_eq!(a_i, i as i32);
        }
    }
    #[test]
    fn polynomial_from_iterator_empty() {
        let p: Polynomial<i32, 10> = Polynomial::from_iterator(0..0);
        for a_i in p {
            assert_eq!(a_i, 0);
        }
    }
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn polynomial_from_iterator_too_long() {
        let p: Polynomial<i32, 10> = Polynomial::from_iterator(0..100);
        for (i, a_i) in p.into_iter().enumerate() {
            assert_eq!(a_i, i as i32);
        }
    }
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn resize_to_larger() {
        let p = Polynomial::from_data([0, 1, 2, 3, 4]);
        let new_p = p.resize::<10>();
        for (i, a_i) in new_p.into_iter().enumerate() {
            if i >= 5 {
                assert_eq!(a_i, 0);
            } else {
                assert_eq!(a_i, i as i32);
            }
        }
    }
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn resize_to_same() {
        let p = Polynomial::from_data([0, 1, 2, 3, 4]);
        let new_p = p.resize::<5>();
        for (i, a_i) in new_p.into_iter().enumerate() {
            assert_eq!(a_i, i as i32);
        }
    }
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn resize_to_shorter() {
        let p = Polynomial::from_data([0, 1, 2, 3, 4]);
        let new_p = p.resize::<1>();
        for (i, a_i) in new_p.into_iter().enumerate() {
            assert_eq!(a_i, i as i32);
            assert!(i < 1);
        }
    }
}

mod evaluation {
    use super::*;
    #[test]
    fn constant() {
        let constant = Constant::from_element(10u16);
        assert_eq!(constant.evaluate(&0), 10);
        assert_eq!(constant.evaluate(&1), 10);
        assert_eq!(constant.evaluate(&2), 10);
    }
    #[test]
    fn line() {
        let line = Line::new([1i8, 0i8]);
        assert_eq!(line.evaluate(&-2), -2);
        assert_eq!(line.evaluate(&-1), -1);
        assert_eq!(line.evaluate(&0), 0);
        assert_eq!(line.evaluate(&1), 1);
        assert_eq!(line.evaluate(&2), 2);
    }
    #[test]
    fn quadratic() {
        let quadratic = Quadratic::new([1i8, 0i8, 0i8]);
        assert_eq!(quadratic.evaluate(&-2), 4);
        assert_eq!(quadratic.evaluate(&-1), 1);
        assert_eq!(quadratic.evaluate(&0), 0);
        assert_eq!(quadratic.evaluate(&1), 1);
        assert_eq!(quadratic.evaluate(&2), 4);
    }
    #[test]
    fn cubic() {
        let cubic = Cubic::new([1i8, 0i8, 0i8, 0i8]);
        assert_eq!(cubic.evaluate(&-2), -8);
        assert_eq!(cubic.evaluate(&-1), -1);
        assert_eq!(cubic.evaluate(&0), 0);
        assert_eq!(cubic.evaluate(&1), 1);
        assert_eq!(cubic.evaluate(&2), 8);
    }
}
