//! Tests for constructors and accessors of the basic polynomial type.
//!
//! This file provides basic usage testing, initialization, accessors and setters, for
//! specialized polynomial types. The specializations involve varying the underlying array data
//! type `T` and the capacity of the array `N`. The tests are written in a way that they can be
//! re-used for all specializations. The types should cover numbers and the sizes should not exceed
//! 4096 bytes.
//!
//! TODO:
//!     * constructors
//!         * new(coefficients; [T; N])
//!         * resize()
//!         * compose(f: Polynomial, g: Polynomial) -> Polynomial
//!         * to_monic(&self) -> Self
//!     * accessors
//!         * coefficient()
//!         * coefficient_mut()
//!         * constant()
//!         * constant_mut()
//!         * leading_coefficient()
//!         * leading_coefficient_mut()
//!         * resize(polynomial: Polynomial)


#[cfg(feature = "std")]
use std::{fmt, any::type_name};

#[cfg(not(feature = "std"))]
use core::{fmt, any::type_name};

use crate::polynomial::Polynomial;

use num_traits::{Zero, One};

fn is_empty_degree_constant_monic_validator<T: Clone + Zero + One + PartialEq + fmt::Debug, const N: usize>(
    _test_name: &str,
    polynomial: Polynomial<T, N>,
    is_empty: bool,
    expected_degree: Option<usize>,
    expected_constant: Option<&T>,
    expected_monic: bool
) {
    let _type_name = type_name::<T>();
    assert_eq!(polynomial.is_empty(), is_empty, "Polynomial::<{_type_name}, {N}>::{_test_name}().is_empty()");
    assert_eq!(polynomial.degree(), expected_degree, "Polynomial::<{_type_name}, {N}>::{_test_name}().degree()");
    assert_eq!(polynomial.constant(), expected_constant, "Polynomial::<{_type_name}, {N}>::{_test_name}().constant()");
    assert_eq!(polynomial.is_monic(), expected_monic, "Polynomial::<{_type_name}, {N}>::{_test_name}().is_monic()");
}

fn from_data_validator<T>()
where
    T: Clone + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::from_data([]);
    is_empty_degree_constant_monic_validator("from_data", empty_polynomial, true, None, None, false);

    let constant = Polynomial::from_data([T::one()]);
    is_empty_degree_constant_monic_validator("from_data", constant, false, Some(0), Some(&T::one()), true);
    // TODO: Line + Quadratic + Cubic + ...
}

fn from_fn_validator<T>()
where
    T: Default + Clone + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::from_fn(|_| T::one());
    is_empty_degree_constant_monic_validator("from_fn", empty_polynomial, true, None, None, false);

    let mut counter = T::zero();
    let constant: Polynomial<T, 1> = Polynomial::from_fn(|_| {counter = counter.clone() + T::one(); counter.clone() });
    is_empty_degree_constant_monic_validator("from_fn", constant, false, Some(0), Some(&T::one()), true);
    // TODO: Line + Quadratic + Cubic + ...
}

fn from_iterator_validator<T>()
where
    T: Clone + Copy + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::from_iterator([]);
    is_empty_degree_constant_monic_validator("from_iterator", empty_polynomial, true, None, None, false);

    let constant: Polynomial<T, 1> = Polynomial::from_iterator([T::zero()]);
    is_empty_degree_constant_monic_validator("from_iterator", constant, false, None, Some(&T::zero()), false);
    // TODO: Line + Quadratic + Cubic + ...
}

fn default_validator<T>()
where
    T: Default + Clone + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::default();
    is_empty_degree_constant_monic_validator("default", empty_polynomial, true, None, None, false);

    let constant: Polynomial<T, 1> = Polynomial::default();
    let degree = match T::default() == T::zero() {
        true => None,
        _ => Some(0),
    };
    is_empty_degree_constant_monic_validator("default", constant, false, degree, Some(&T::default()), T::default() == T::one());
    // TODO: Line + Quadratic + Cubic + ...
}

fn monomial_validator<T>()
where
    T: Clone + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::monomial(T::zero());
    is_empty_degree_constant_monic_validator("monomial", empty_polynomial, true, None, None, false);

    let constant: Polynomial<T, 1> = Polynomial::monomial(T::one());
    is_empty_degree_constant_monic_validator("monomial", constant, false, Some(0), Some(&T::one()), true);
    // TODO: Line + Quadratic + Cubic + ...
}

fn from_element_validator<T>()
where
    T: Clone + Copy + Zero + One + PartialEq + fmt::Debug,
{
    let empty_polynomial: Polynomial<T, 0> = Polynomial::from_element(T::zero());
    is_empty_degree_constant_monic_validator("from_element", empty_polynomial, true, None, None, false);

    let constant: Polynomial<T, 1> = Polynomial::from_element(T::one());
    is_empty_degree_constant_monic_validator("from_element", constant, false, Some(0), Some(&T::one()), true);
    // TODO: Line + Quadratic + Cubic + ...
}

#[test]
fn i8_constructors() {
    from_data_validator::<i8>();
    from_fn_validator::<i8>();
    from_iterator_validator::<i8>();
    default_validator::<i8>();
    monomial_validator::<i8>();
    from_element_validator::<i8>();
}

#[test]
fn u8_constructors() {
    from_data_validator::<u8>();
    from_fn_validator::<u8>();
    from_iterator_validator::<u8>();
    default_validator::<u8>();
    monomial_validator::<u8>();
    from_element_validator::<u8>();
}