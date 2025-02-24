use num_traits::Num;

use super::*;

fn validate_polynomial<T, D, S>(
    name: &str,
    polynomial: Polynomial<T, D, S>,
    input: &[T],
    expected: &[T],
) where
    T: Copy + Num + fmt::Debug,
    D: Dim,
    S: RawStorage<T, D>,
{
    for (input, expected) in input.iter().zip(expected) {
        let result = polynomial.evaluate(*input);
        assert_eq!(
            result, *expected,
            "{name}.evaluate({input:?}) expected {expected:?} (found {result:?})"
        );
    }
}

#[test]
fn constant_one() {
    validate_polynomial(
        "1f32 ArrayStorage",
        Polynomial::new([1.0]),
        &[-1.0, 0.0, 1.0],
        &[1.0, 1.0, 1.0],
    );
    validate_polynomial(
        "1f32 VecStorage",
        Polynomial::from_vec(vec![1.0]),
        &[-1.0, 0.0, 1.0],
        &[1.0, 1.0, 1.0],
    );
}

#[test]
fn zero_polynomial() {
    validate_polynomial(
        "0f64 ArrayStorage",
        Polynomial::new([0.0f64]),
        &[-1.0, 0.0, 1.0],
        &[0.0, 0.0, 0.0],
    );
    validate_polynomial(
        "0f64 VecStorage",
        Polynomial::from_vec(vec![0.0f64]),
        &[-1.0, 0.0, 1.0],
        &[0.0, 0.0, 0.0],
    );
}

#[test]
fn linear_polynomial_x() {
    validate_polynomial(
        "linear i32 ArrayStorage",
        Polynomial::new([1i32, 0i32]),
        &[-1, 0, 1],
        &[-1, 0, 1],
    );
    validate_polynomial(
        "linear i32 VecStorage",
        Polynomial::from_vec(vec![1i32, 0i32]),
        &[-1, 0, 1],
        &[-1, 0, 1],
    );
}

#[test]
fn quadratic_polynomial_x2() {
    validate_polynomial(
        "quadratic u8 ArrayStorage",
        Polynomial::new([1u8, 0u8, 0u8]),
        &[0, 1, 2],
        &[0, 1, 4],
    );
    validate_polynomial(
        "quadrtic u8 VecStorage",
        Polynomial::from_vec(vec![1u8, 0u8, 0u8]),
        &[0, 1, 2],
        &[0, 1, 4],
    );
}
