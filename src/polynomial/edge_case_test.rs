use super::*;
use nalgebra::{ArrayStorage, Const, Matrix, RawStorage, SMatrix, Scalar, U1, U2};
use num_traits::{Float, Num, Zero};

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
        assert_eq!(result, *expected, "{name}.evaluate({input:?})");
        let result = polynomial.evaluate(Complex::new(*input, U::zero()));
        assert_eq!(result.re, *expected, "{name}.evaluate(Complex<{input:?}>)");
    }
}

fn validate_derivative<T, const D: usize, const D1: usize>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    expected: Polynomial<T, D1>,
) where
    T: Scalar + fmt::Display + Num + Copy,
{
    let result = polynomial.derivative("x'");
    assert_eq!(
        result.coefficients, expected.coefficients,
        "{name}.derivative() coefficients",
    );
    assert_eq!(
        result.variable, expected.variable,
        "{name}.derivative() variable",
    );
}

fn validate_companion<T, const D: usize, S: RawStorage<T, DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>> + fmt::Debug>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    expected: Matrix<T, DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>, S>,
) where
    T: 'static + Copy + Num + Neg<Output = T> + fmt::Debug,
    Const<D>: DimSub<U1>,
    DimDiff<Const<D>, U1>: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>>,
{
    let result = polynomial.companion();
    assert_eq!(result, expected, "{name}.companion()");
}

fn validate_roots<T, const D: usize>(
    name: &str,
    polynomial: &Polynomial<T, D>,
    expected: Option<OMatrix<Complex<T>, DimDiff<Const<D>, U1>, U1>>,
) where
    T: Scalar + fmt::Display + RealField + Float,
    Const<D>: DimSub<U1>,
    DimDiff<Const<D>, U1>: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>> 
        + Allocator<DimDiff<Const<D>, U1>, DimDiff<DimDiff<Const<D>, U1>, U1>> 
        + Allocator<DimDiff<DimDiff<Const<D>, U1>, U1>> 
        + Allocator<DimDiff<Const<D>, U1>>,
{
    let result = polynomial.roots();
    assert_eq!(result, expected, "{name}.roots()");
}

fn validate<T, U, const D: usize, const D1: usize, const N: usize, S>(
    name: &'static str,
    polynomial: Polynomial<T, D>,
    evaluation_input: [U; N],
    expected_evaluation: [U; N],
    expected_derivative: Polynomial<T, D1>,
    expected_companion: Matrix<T, DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>, S>,
    expected_roots: Option<OMatrix<Complex<T>, DimDiff<Const<D>, U1>, U1>>,
) where
    T: Scalar + fmt::Display + RealField + Float,
    U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U> + Scalar,
    Complex<U>: Add<T, Output = Complex<U>> + Mul<U, Output = Complex<U>>,
    S: RawStorage<T, DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>> + fmt::Debug,
    Const<D>: DimSub<U1>,
    DimDiff<Const<D>, U1>: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<D>, U1>, DimDiff<Const<D>, U1>> 
        + Allocator<DimDiff<Const<D>, U1>, DimDiff<DimDiff<Const<D>, U1>, U1>> 
        + Allocator<DimDiff<DimDiff<Const<D>, U1>, U1>> 
        + Allocator<DimDiff<Const<D>, U1>>,
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
        &[0.0, 0.0, 0.0, 0.0, 0.0]
    );
    validate_derivative(
        "constant_zero", 
        &polynomial, 
        Polynomial::new("x'", [])
    );
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
fn linear_u32() {
    let polynomial = Polynomial::new("x", [1, 0]);
    validate_evaluate(
        "linear",
        &polynomial,
        &[0, 1, 2],
        &[0, 1, 2],
    );
    validate_derivative("linear", &polynomial, Polynomial::new("x'", []));
    validate_companion(
        "linear", 
        &polynomial, 
        OMatrix::<i32, U1, U1>::from_data(ArrayStorage([[0; 1]; 1]))
    );
    
    // cannot call roots on polynomial where T: !RealField
    // validate_roots(
    //     "linear", 
    //     &polynomial, 
    //     Some(OMatrix::<Complex<i32>, U1, U1>::new(Complex::new(0, 0)))
    // );
}

#[test]
fn unit_quadratic_i16() {
    validate(
        "unit_quadratic",
        Polynomial::new("x", [1.0, 0.0, 0.0]),
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [4.0, 1.0, 0.0, 1.0, 4.0],
        Polynomial::new("x'", []),
        SMatrix::<f32, 2, 2>::new(0.0, 0.0, 1.0, 0.0),
        Some(OMatrix::<Complex<f32>, U2, U1>::new(
            Complex::zero(),
            Complex::zero(),
        )),
    );
}
