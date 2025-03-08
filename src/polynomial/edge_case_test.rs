use super::*;
use nalgebra::{ArrayStorage, Matrix, RawStorage, SMatrix, Scalar, U1, U2};
use num_traits::{Float, Num, Zero};

fn validate_evaluate<T, U, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    input: &[U],
    expected: &[U],
) where
    T: Copy + Num + Scalar,
    U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U> + Scalar,
    Complex<U>: Add<T, Output = Complex<U>> + Mul<U, Output = Complex<U>>,
    D: Dim,
    S: RawStorage<T, D>,
{
    for (input, expected) in input.iter().zip(expected) {
        let result = polynomial.evaluate(*input);
        assert_eq!(result, *expected, "{name}.evaluate({input:?})");
        let complex_result = polynomial.evaluate(Complex::new(*input, U::zero()));
        assert_eq!(
            complex_result.re, *expected,
            "{name}.evaluate(Complex<{input:?},0>)"
        );
        assert_eq!(
            complex_result.im,
            U::zero(),
            "{name}.evaluate(Complex<{input:?},0>)"
        );
    }
    if polynomial.num_coefficients() > 0 {
        let constant_result = polynomial.evaluate(U::zero());
        assert_eq!(
            constant_result,
            U::zero() + polynomial[polynomial.num_coefficients() - 1],
            "{name}.evaluate(0) == {:?}",
            polynomial[polynomial.num_coefficients() - 1]
        );
    }
    if polynomial.num_coefficients() > 1 {
        let imaginary_result = polynomial.evaluate(Complex::new(U::zero(), U::one()));
        if polynomial.num_coefficients() % 2 == 0 {
            assert!(
                !imaginary_result.im.is_zero(),
                "{name}.evaluate(Complex<0,1>).im == {imaginary_result:?}"
            );
        } else {
            assert!(
                imaginary_result.im.is_zero(),
                "{name}.evaluate(Complex<0,1>).im == {imaginary_result:?}"
            );
        }
    }
}

// not a safe way to constrain D and D1, don't do this
fn validate_derivative<T, D, S, S1>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    expected: Polynomial<T, DimDiff<D, U1>, S1>,
) where
    T: Scalar + fmt::Display + Num + Copy + Default,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName,
    S: RawStorage<T, D>,
    S1: RawStorageMut<T, DimDiff<D, U1>> + Default,
{
    let result: Polynomial<T, DimDiff<D, U1>, S1> = polynomial.derivative("x'");
    for i in 0..result.num_coefficients() {
        assert_eq!(
            result[i], expected[i],
            "{name}.derivative() coefficients[{i}]",
        );
    }
    assert_eq!(
        result.variable, expected.variable,
        "{name}.derivative() variable",
    );
}

fn validate_companion<T, D, S, S1>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    expected: Matrix<T, DimDiff<D, U1>, DimDiff<D, U1>, S1>,
) where
    T: 'static + Copy + Num + Neg<Output = T> + fmt::Debug,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName + DimSub<U1>,
    S: RawStorage<T, D>,
    S1: RawStorage<T, DimDiff<D, U1>, DimDiff<D, U1>> + fmt::Debug,
    DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>>,
{
    let result = polynomial.companion();
    assert_eq!(result, expected, "{name}.companion()");
}

fn validate_roots<T, D, S>(
    name: &str,
    polynomial: &Polynomial<T, D, S>,
    expected: OMatrix<Complex<T>, DimDiff<D, U1>, U1>,
) where
    T: Scalar + fmt::Display + RealField + Float,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName + DimSub<U1>,
    S: RawStorage<T, D>,
    DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>>
        + Allocator<DimDiff<D, U1>, DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<D, U1>>,
{
    assert_eq!(polynomial.roots(), expected, "{name}.roots()");
}

fn validate<T, D, S, const N: usize, U, S1, S2>(
    name: &'static str,
    polynomial: Polynomial<T, D, S>,
    evaluation_input: [U; N],
    expected_evaluation: [U; N],
    expected_derivative: Polynomial<T, DimDiff<D, U1>, S1>,
    expected_companion: Matrix<T, DimDiff<D, U1>, DimDiff<D, U1>, S2>,
    expected_roots: OMatrix<Complex<T>, DimDiff<D, U1>, U1>,
) where
    T: Scalar + fmt::Display + RealField + Float + Default,
    U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U> + Scalar,
    Complex<U>: Add<T, Output = Complex<U>> + Mul<U, Output = Complex<U>>,
    S: RawStorage<T, D>,
    S1: RawStorageMut<T, DimDiff<D, U1>> + Default,
    S2: RawStorage<T, DimDiff<D, U1>, DimDiff<D, U1>> + fmt::Debug,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>>
        + Allocator<DimDiff<D, U1>, DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<D, U1>>,
{
    validate_evaluate(name, &polynomial, &evaluation_input, &expected_evaluation);
    validate_derivative(name, &polynomial, expected_derivative);
    validate_companion(name, &polynomial, expected_companion);
    validate_roots(name, &polynomial, expected_roots);
}

#[test]
fn constant_zero_f64() {
    let polynomial = Polynomial::new("x", [0.0f64]);
    validate_evaluate(
        "constant_zero",
        &polynomial,
        &[-2.0, -2.0, 0.0, 1.0, 2.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0],
    );
    validate_derivative("constant_zero", &polynomial, Polynomial::new("x'", []));
    // cannot call companion on polynomial degree < 1
    // validate_companion(
    //     "constant_zero",
    //     &polynomial,
    //     OMatrix::<f64, U0, U0>::from_data(ArrayStorage([[0.0; 0]; 0]))
    // );

    // cannot call roots on polynomial degree < 1
    // validate_roots("constant_zero", &polynomial, None);
}

#[test]
fn constant_one_i32() {
    let polynomial = Polynomial::new("x", [1]);
    validate_evaluate(
        "constant_one",
        &polynomial,
        &[-2, -1, 0, 1, 2],
        &[1, 1, 1, 1, 1],
    );
    validate_derivative("constant_one", &polynomial, Polynomial::new("x'", []));
    // cannot call companion on polynomial where D: !DimSub<U1>
    // validate_companion(
    //     "constant_one",
    //     &polynomial,
    //     OMatrix::<i32, U0, U0>::from_data(ArrayStorage([[0; 0]; 0]))
    // );

    // cannot call roots on polynomial where T: !RealField
    // validate_roots(
    //     "constant_one",
    //     &polynomial,
    //     None
    // );
}

#[test]
fn linear_i32() {
    let polynomial = Polynomial::new("x", [1, 0]);
    validate_evaluate("linear_i32", &polynomial, &[0, 1, 2], &[0, 1, 2]);
    validate_derivative("linear_i32", &polynomial, Polynomial::new("x'", [1]));
    validate_companion(
        "linear_i32",
        &polynomial,
        OMatrix::<i32, U1, U1>::from_data(ArrayStorage([[0; 1]; 1])),
    );

    // cannot call roots on polynomial where T: !RealField
    // validate_roots(
    //     "linear",
    //     &polynomial,
    //     Some(OMatrix::<Complex<i32>, U1, U1>::new(Complex::new(0, 0)))
    // );
}

#[test]
fn linear_f32() {
    let polynomial = SPolynomial::new("x", [1.0, 0.0]);
    validate_evaluate(
        "linear_f32",
        &polynomial,
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
    );
    validate_derivative("linear_f32", &polynomial, Polynomial::new("x'", [1.0]));
    validate_companion(
        "linear_f32",
        &polynomial,
        OMatrix::<f32, U1, U1>::from_data(ArrayStorage([[0.0; 1]; 1])),
    );

    // cannot call roots on polynomial where T: !RealField
    // validate_roots(
    //     "linear",
    //     &polynomial,
    //     Some(OMatrix::<Complex<i32>, U1, U1>::new(Complex::new(0, 0)))
    // );
}

#[test]
fn unit_quadratic_f32() {
    validate(
        "unit_quadratic",
        Polynomial::new("x", [1.0, 0.0, 0.0]),
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [4.0, 1.0, 0.0, 1.0, 4.0],
        Polynomial::new("x'", [2.0, 0.0]),
        SMatrix::<f32, 2, 2>::new(0.0, 0.0, 1.0, 0.0),
        OMatrix::<Complex<f32>, U2, U1>::new(Complex::zero(), Complex::zero()),
    );
}

#[test]
fn offset_quadratic_f64() {
    validate(
        "offset_quadratic",
        Polynomial::new("x", [1.0, 0.0, -2.0]),
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [2.0, -1.0, -2.0, -1.0, 2.0],
        Polynomial::new("x'", [2.0, 0.0]),
        SMatrix::<f64, 2, 2>::new(0.0, 2.0, 1.0, 0.0),
        OMatrix::<Complex<f64>, U2, U1>::new(
            Complex {
                re: -1.414213562373095,
                im: 0.0,
            },
            Complex {
                re: 1.414213562373095,
                im: 0.0,
            },
        ),
    );
}

#[test]
fn unit_cubic_f64() {
    let polynomial = Polynomial::new("x", [1.0, 0.0, 0.0, 0.0]);
    validate_evaluate(
        "unit_cubic_f64",
        &polynomial,
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
        &[-8.0, -1.0, 0.0, 1.0, 8.0],
    );
    validate_derivative(
        "unit_cubic_f64",
        &polynomial,
        Polynomial::new("x'", [3.0, 0.0, 0.0]),
    );
    validate_companion(
        "unit_cubic_f64",
        &polynomial,
        SMatrix::<f64, 3, 3>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    );

    // cannot call roots on polynomial where T: !RealField
    // validate_roots(
    //     "linear",
    //     &polynomial,
    //     OMatrix::<Complex<f64>, U3, U1>::new(Complex::zero(), Complex::zero(), Complex::zero())
    // );

    // validate(
    //     "unit_cubic_f64",
    //     Polynomial::new("x", [1.0, 0.0, 0.0, 0.0]),
    //     [-2.0, -1.0, 0.0, 1.0, 2.0],
    //     [-8.0, -1.0, 0.0, 1.0, 8.0],
    //     Polynomial::new("x'", [3.0, 0.0, 0.0]),
    //     SMatrix::<f64, 3, 3>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    //     OMatrix::<Complex<f64>, U3, U1>::new(Complex::zero(), Complex::zero(), Complex::zero()),
    // );
}
