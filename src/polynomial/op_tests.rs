use super::*;
use nalgebra::{RawStorage, Scalar};
use num_traits::Num;

fn validate_scalar_post_add<T, U, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    value: U,
    expected: Polynomial<T, D, S>,
) where
    T: Copy + Num + Scalar + Neg<Output = T> + PartialOrd + fmt::Display,
    Polynomial<T, D, S>: Add<U, Output = Polynomial<T, D, S>> + Copy,
    U: fmt::Display + Copy,
    D: Dim,
    S: RawStorage<T, D> + PartialEq + fmt::Debug,
{
    // Compute the result of adding the value to the polynomial
    let result = *polynomial + value;
    
    // Check if the resulting polynomial matches the expected one
    assert_eq!(result.coefficients, expected.coefficients, "{name}: ({polynomial}) + {value} = {result}");
}

fn validate_scalar_post_sub<T, U, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    value: U,
    expected: Polynomial<T, D, S>,
) where
    T: Copy + Num + Scalar + Neg<Output = T> + PartialOrd + fmt::Display,
    Polynomial<T, D, S>: Sub<U, Output = Polynomial<T, D, S>> + Copy,
    U: fmt::Display + Copy,
    D: Dim,
    S: RawStorage<T, D> + PartialEq + fmt::Debug,
{
    // Compute the result of adding the value to the polynomial
    let result = *polynomial - value;
    
    // Check if the resulting polynomial matches the expected one
    assert_eq!(result.coefficients, expected.coefficients, "{name}: ({polynomial}) - {value} = {result}");
}

fn validate_scalar_post_mul<T, U, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    value: U,
    expected: Polynomial<T, D, S>,
) where
    T: Copy + Num + Scalar + Neg<Output = T> + PartialOrd + fmt::Display,
    Polynomial<T, D, S>: Mul<U, Output = Polynomial<T, D, S>> + Copy,
    U: fmt::Display + Copy,
    D: Dim,
    S: RawStorage<T, D> + PartialEq + fmt::Debug,
{
    // Compute the result of adding the value to the polynomial
    let result = *polynomial * value;
    
    // Check if the resulting polynomial matches the expected one
    assert_eq!(result.coefficients, expected.coefficients, "{name}: ({polynomial}) * {value} = {result}");
}

fn validate_scalar_post_div<T, U, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    value: U,
    expected: Polynomial<T, D, S>,
) where
    T: Copy + Num + Scalar + Neg<Output = T> + PartialOrd + fmt::Display,
    Polynomial<T, D, S>: Div<U, Output = Polynomial<T, D, S>> + Copy,
    U: fmt::Display + Copy,
    D: Dim,
    S: RawStorage<T, D> + PartialEq + fmt::Debug,
{
    // Compute the result of adding the value to the polynomial
    let result = *polynomial / value;
    
    // Check if the resulting polynomial matches the expected one
    assert_eq!(result.coefficients, expected.coefficients, "{name}: ({polynomial}) / {value} = {result}");
}


#[test]
fn linear_f32() {
    let polynomial = Polynomial::new("x", [1.0, 0.0]);
    validate_scalar_post_add("linear_f32", &polynomial, 1.0, Polynomial::new("x", [1.0, 1.0]));
    validate_scalar_post_sub("linear_f32", &polynomial, 1.0, Polynomial::new("x", [1.0, -1.0]));
    validate_scalar_post_mul("linear_f32", &polynomial, 10.0, Polynomial::new("x", [10.0, 0.0]));
    validate_scalar_post_div("linear_f32", &polynomial, 10.0, Polynomial::new("x", [0.1, 0.0]));
}

#[test]
fn cubic_i32() {
    let polynomial = Polynomial::new("x", [1, 0, 0, 0]);
    validate_scalar_post_add("linear_f32", &polynomial, 1, Polynomial::new("x", [1, 0, 0, 1]));
    validate_scalar_post_sub("linear_f32", &polynomial, 1, Polynomial::new("x", [1, 0, 0, -1]));
    validate_scalar_post_mul("linear_f32", &polynomial, 10, Polynomial::new("x", [10, 0, 0, 0]));
    validate_scalar_post_div("linear_f32", &polynomial, 10, Polynomial::new("x", [0, 0, 0, 0]));
}

#[test]
fn empty() {
    let polynomial: Polynomial<i32, Const<0>, ArrayStorage<i32, 0, 1>> = Polynomial::new("x", []);
    // cannot add or sub polynomial with no coeff, is that ok?
    // validate_scalar_post_add("linear_f32", &polynomial, 1, Polynomial::new("x", []));
    // validate_scalar_post_sub("linear_f32", &polynomial, 1, Polynomial::new("x", []));
    validate_scalar_post_mul("linear_f32", &polynomial, 10, Polynomial::new("x", []));
    validate_scalar_post_div("linear_f32", &polynomial, 10, Polynomial::new("x", []));
}

#[test]
fn degenerate() {
    let polynomial = Polynomial::new("x", [0]);
    validate_scalar_post_add("linear_f32", &polynomial, 1, Polynomial::new("x", [1]));
    validate_scalar_post_sub("linear_f32", &polynomial, 1, Polynomial::new("x", [-1]));
    validate_scalar_post_mul("linear_f32", &polynomial, 10, Polynomial::new("x", [0]));
    validate_scalar_post_div("linear_f32", &polynomial, 10, Polynomial::new("x", [0]));
}