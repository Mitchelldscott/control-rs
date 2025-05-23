//! Tests for constructors and accessors of the basic polynomial type.
//!
//! This file provides basic usage testing, initialization, accessors and setters, for
//! specialized polynomial types. The specializations involve varying the underlying array data
//! type `T` and the capacity of the array `N`. The tests are written in a way that they can be
//! re-used for all specializations. The types should only cover numerical primitives (maybe custom
//! numbers) and the sizes should not exceed 4096 bytes.
//!
//!
//! TODO:
//!     * constructors
//!         * from_fn<F>(cb: F): needs example and tests
//!         * from_element(element: T): needs example
//!         * new(coefficients; [T; N]): needs example
//!         * from_constant(constant: T): needs example
//!         * monomial(): needs example
//!         * resize(): needs example and tests
//!         * compose(f: Polynomial, g: Polynomial) -> Polynomial: Needs investigation
//!         * to_monic(&self) -> Self
//!         * zero(): needs tests and example
//!         * one(): needs tests and example
//!     * accessors
//!         * degree(): needs tests and example
//!         * is_monic(): needs tests and example
//!         * is_zero(): needs tests and example
//!         * is_one(): needs tests and example
//!         * coefficient(): needs tests and example
//!         * coefficient_mut(): needs tests and example
//!         * constant(): needs tests and example
//!         * constant_mut(): needs tests and example
//!         * leading_coefficient(): needs tests and example
//!         * leading_coefficient_mut(): needs tests and example


#[cfg(feature = "std")]
use std::{fmt, any::type_name};

#[cfg(not(feature = "std"))]
use core::{fmt, any::type_name};

use crate::polynomial::Polynomial;

use num_traits::Num;

fn default_constructor_validator<T, const N: usize>(expected_coefficients: [T; N])
where
    T: Default + Copy + Clone + PartialEq + fmt::Debug,
{
    let type_name = type_name::<T>();
    let polynomial = Polynomial::<T, N>::default();
    assert_eq!(polynomial.coefficients, expected_coefficients, "Polynomial::<{type_name}, {N}>::default()");
}

fn monomial_constructor_validator<T, const N: usize>()
where
    T: Clone + Num + PartialEq + fmt::Debug,
{
    let type_name = type_name::<T>();
    let polynomial = Polynomial::<T, N>::monomial();
    assert_eq!(polynomial.coefficients[N-1], T::one(), "Polynomial::<{type_name}, {N}>::monomial() leading term");
    for i in 0..N-1 {
        assert_eq!(polynomial.coefficients[i], T::zero(), "Polynomial::<{type_name}, {N}>::monomial() term {i}");
    }
}

// impl<T, const N: usize> Polynomial<T, N> {
//     /// Creates a new polynomial from an iterator
//     ///
//     /// This is using unsafe code and should be ditched. Or find a way to make it safe
//     ///
//     /// # Arguments
//     /// * `iterator` - The iterator to be copied into the coefficient array
//     ///
//     /// # Returns
//     /// * `polynomial` - polynomial with all coefficients set to `iterator`
//     pub fn from_iterator<U: IntoIterator<Item=T>>(iterator: U) -> Result<Self, PolynomialInitError> {
//         // SAFETY: An uninitialized `[MaybeUninit<_>; _]` is valid.
//         let mut uninit_buffer: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
//         let mut count =  0;
//
//         for (buffer, value) in uninit_buffer.iter_mut().zip(iterator.into_iter()) {
//             *buffer = MaybeUninit::new(value);
//             count += 1;
//         }
//
//         if count == N {
//             return Ok(
//                 unsafe {
//                     Self::new((&uninit_buffer as *const _ as *const [_; N]).read())
//                 }
//             );
//         }
//
//         Err(PolynomialInitError::LengthMismatch{ expected: N, actual: count })
//     }
// }

// fancy constructors introduce unsafe code and depend on [core::mem::MaybeUninit]
// for now restrict constructors to types implementing Copy
// fn from_iter_constructor_validator<T: Default, const N: usize, U: IntoIterator<Item=T> + Clone>(type_name: &str, coefficients_iter: U) {
//     let from_iter_poly = Polynomial::<T, N>::from_iterator(coefficients_iter.clone()).expect("Polynomial::<{type_name}, {N}>::from_iterator()");
//     assert_eq!(from_iter_poly.coefficients, coefficients_iter.into_iter().collect().try_into(), "Polynomial::<{type_name}, {N}>::from_iterator()");
// }

fn constant_constructor_validator<T, const N: usize>(constant: T)
where
    T: Clone + Num + PartialEq + fmt::Debug,
{
    let type_name = std::any::type_name::<T>();
    let polynomial = Polynomial::<T, N>::from_constant(constant.clone());
    assert_eq!(polynomial.coefficients[0], constant, "Polynomial::<{type_name}, {N}>::from_constant({constant:?}) constant");
    for i in 1..N {
        assert_eq!(polynomial.coefficients[i], T::zero(), "Polynomial::<{type_name}, {N}>::from_constant({constant:?}) term {i}");
    }
}

fn from_element_constructor_validator<T, const N: usize>(element: T)
where
    T: Copy + Clone + Num + PartialEq + fmt::Debug,
{
    let type_name = std::any::type_name::<T>();
    let polynomial = Polynomial::<T, N>::from_element(element);
    for i in 0..N {
        assert_eq!(polynomial.coefficients[i], element, "Polynomial::<{type_name}, {N}>::from_element({element:?}) term {i}");
    }
}

#[test]
fn i8_constructors() {
    default_constructor_validator::<i8, 1>([0i8]);
    monomial_constructor_validator::<i8, 3>();
    // from_iter_constructor_validator::<i8, 4, Range<i8>>(, (1..4));
    constant_constructor_validator::<i8, 10>(127i8);
    from_element_constructor_validator::<i8, 2>(1i8);
}

#[test]
fn u8_constructors() {
    default_constructor_validator::<u8, 10>([0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8]);
    monomial_constructor_validator::<u8, 3>();
    // from_iter_constructor_validator::<i8, 4, Range<i8>>(, (1..4));
    constant_constructor_validator::<u8, 10>(127u8);
    from_element_constructor_validator::<u8, 0>(1u8);
}