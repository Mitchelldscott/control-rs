use super::*;
use nalgebra::{Const, Matrix, RawStorage, SMatrix, Scalar};
use num_traits::Num;

fn validate_evaluate<T, U, const D: usize>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    input: &[U],
    expected: &[U],
) where
    T: Copy + Num + Scalar,
    U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U> + Scalar,
    Complex<U>: Add<T, Output = Complex<U>> + Mul<U, Output = Complex<U>>,
{
    for (input, expected) in input.iter().zip(expected) {
        let result = polynomial.evaluate(*input);
        assert_eq!(
            result, *expected,
            "{name}.evaluate({input:?})"
        );
        let result = polynomial.evaluate(Complex::new(*input, U::zero()));
        assert_eq!(
            result.re, *expected,
            "{name}.evaluate(Complex<{input:?}>)"
        );
    }
}

fn validate_derivative<T, const D: usize, const D1: usize>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    expected: Polynomial<T, D1>,
) where
    T: Copy + Num + Scalar,
{
    let result = polynomial.derivative("x'");
    assert_eq!(
        result.coefficients,
        expected.coefficients,
        "{name}.derivative() coefficients",
    );
    assert_eq!(
        result.variable,
        expected.variable,
        "{name}.derivative() variable",
    );
}

fn validate_companion<T, const D: usize, S: RawStorage<T, Const<D>, Const<D>> + fmt::Debug>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    expected: Matrix<T, Const<D>, Const<D>, S>,
) where
    T: Copy + Num + Scalar + Neg<Output = T> + fmt::Display,
{
    let result = polynomial.companion();
    assert_eq!(
        result, expected,
        "{name}.companion()"
    );
}


fn validate<T, U, const D: usize, const D1: usize, const N: usize, S>(
    name: &'static str,
    polynomial: Polynomial<T, D>,
    evaluation_input: [U; N],
    expected_evaluation: [U; N],
    expected_derivative: Polynomial<T, D1>,
    expected_companion: Matrix<T, Const<D>, Const<D>, S>
) 
where
    T: Copy + Num + Scalar + Neg<Output = T> + fmt::Display,
    U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U> + Scalar,
    Complex<U>: Add<T, Output = Complex<U>> + Mul<U, Output = Complex<U>>,
    S: RawStorage<T, Const<D>, Const<D>> + fmt::Debug
{
    validate_evaluate(name, &polynomial, &evaluation_input, &expected_evaluation);
    validate_derivative(name, &polynomial, expected_derivative);
    validate_companion(name, &polynomial, expected_companion);
}

#[test]
fn constant_zero_f64() {
    let polynomial = Polynomial::new("x", [0.0f64]);
    let evaluation_input = [-2.0, -2.0, 0.0, 1.0, 2.0];
    let expected_evaluation = [0.0, 0.0, 0.0, 0.0, 0.0];
    let expected_derivative = Polynomial::new("x'", []);
    let expected_companion = SMatrix::<f64, 1, 1>::new(0.0);
    validate("constant_zero", polynomial, evaluation_input, expected_evaluation, expected_derivative, expected_companion);
}

#[test]
fn constant_one_i32() {
    let polynomial = Polynomial::new("x", [1]);
    let evaluation_input = [-2, -1, 0, 1, 2];
    let expected_evaluation = [1, 1, 1, 1, 1];
    let expected_derivative = Polynomial::new("x'", []);
    validate_evaluate("constant_one", &polynomial, &evaluation_input, &expected_evaluation);
    validate_derivative("constant_one", &polynomial, expected_derivative);
}

#[test]
fn linear_u32() {
    let polynomial = Polynomial::new("x", [1u32, 0u32]);
    let evaluation_input= [0, 1, 2];
    let expected_evaluation= [0, 1, 2];
    let expected_derivative = Polynomial::new("x'", []);
    validate_evaluate("linear", &polynomial, &evaluation_input, &expected_evaluation);
    validate_derivative("linear", &polynomial, expected_derivative);
}

#[test]
fn unit_quadratic_i16() {
    let polynomial = Polynomial::new("x", [1, 0, 0]);
    let evaluation_input = [-2, -1, 0, 1, 2];
    let expected_evaluation = [4, 1, 0, 1, 4];
    let expected_derivative = Polynomial::new("x'", []);
    let expected_companion = SMatrix::<i16, 3, 3>::new(-1, 0, 0, 1, 0, 0, 0, 1, 0);
    validate("unit_quadratic", polynomial, evaluation_input, expected_evaluation, expected_derivative, expected_companion);
}