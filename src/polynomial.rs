//!
//! # Polynomial trait and default implementations
//!
//!
use nalgebra::{allocator::Allocator, Complex, Const, DMatrix, DefaultAllocator, DimDiff, DimSub, OMatrix, RealField, U1};
use num_traits::{Float, Num};

#[cfg(feature = "std")]
use std::{fmt, ops::{Add, Mul, Neg}};

#[cfg(not(feature = "std"))]
use core::{fmt, ops::{Add, Mul, Neg}};

/// static array of coefficients for a polynomial of degree `D - 1` and variable `V`.
pub struct Polynomial<T, const D: usize, const V: u8> {
    // coefficients of the polynomial stored in descending degree order
    coefficients: [T; D]
}

impl<T, const D: usize, const V: u8> Polynomial<T, D, V> {
    /// Create a new polynomial with the given coefficients.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - An array of coefficients of the polynomial in descending degree order.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D, V>` - A new polynomial with the given coefficients.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3, 1>::new([1.0, 2.0, 3.0]);
    /// ```
    pub fn new(coefficients: [T; D]) -> Self {
        Polynomial { coefficients }
    }
}

impl<T: Copy + Num, const D: usize, const V: u8> Polynomial<T, D, V> {
    /// Evaluate the polynomial at the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value at which to evaluate the polynomial.
    ///
    /// # Returns
    ///
    /// * `T` - The value of the polynomial at the given value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3, 1>::new([1.0, 2.0, 3.0]);
    /// let value = p.evaluate(2.0);
    /// ```
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U>,
    {
        self.coefficients.iter().fold(U::zero(), |acc, &c| acc * value + c)
    }

    /// Compute the derivative of the polynomial.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D - 1, V>` - The derivative of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3, 1>::new([1.0, 2.0, 3.0]);
    /// let derivative: Polynomial::<f64, 2, 1> = p.derivative();
    /// ```
    pub fn derivative<const D1: usize>(&self) -> Polynomial<T, D1, V>
    where
        T: From<usize>,
    {
        let mut derivative_coefficients = [T::zero(); D1];
        for i in 0..D1 {
            derivative_coefficients[i] = self.coefficients[i] * T::from(D1 - i);
        }
        Polynomial::new(derivative_coefficients)
    }

    /// Constructs the companion matrix of the polynomial.
    ///
    /// The companion matrix is useful for finding polynomial roots using eigenvalue decomposition.
    ///
    /// # Returns
    ///
    /// * `OMatrix<T, Const<D>, Const<D>>` - The companion matrix representation of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3, 1>::new([1.0, -6.0, 11.0, -6.0]);
    /// let companion_matrix = p.companion();
    /// ```
    pub fn companion(&self) -> OMatrix<T, Const<D>, Const<D>> 
    where 
        T: 'static + Neg<Output = T> + fmt::Debug,
    {
        OMatrix::<T, Const<D>, Const<D>>::from_fn(|i, j| {
            if i == 0 {
                -self.coefficients[j]
            } else {
                if i + 1 == j {
                    T::one()
                } else {
                    T::zero()
                }
            }
        })
    }

    /// Computes the roots of the polynomial.
    ///
    /// Edge cases:
    /// - all coefficients are zero: all roots are infinite
    /// - if there are two coefficients and the lead is non-zero, the root is -coeff[1]/coeff[0]
    /// 
    /// For very high order polynomials this may be inefficient, especially for degenerate cases.
    /// User should consider cases where all/many coeff = 0 and avoid calling this. Would be nice if
    /// nalgebra handled large/sparse matrix eigenvalues.
    ///
    /// # Returns
    ///
    /// * `OMatrix<Complex<T>, Const<D>, U1>` - A column vector containing the computed roots.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3, 1>::new([1.0, -6.0, 11.0, -6.0]);
    /// let roots = p.roots();
    /// ```
    pub fn roots(&self) -> OMatrix<Complex<T>, Const<D>, U1>
    where 
        T: RealField + Float,
        Const<D>: DimSub<U1>,
        DefaultAllocator: Allocator<Const<D>, DimDiff<Const<D>, U1>> + Allocator<DimDiff<Const<D>, U1>> + Allocator<Const<D>, Const<D>> + Allocator<Const<D>>,
    {
        let num_zeros = self.coefficients.iter().fold(0, |acc, &c| match c == T::zero() {
            true => acc + 1,
            false => acc,
        });
        
        if num_zeros == D {
            return OMatrix::<Complex<T>, Const<D>, U1>::from_element(Complex::new(T::infinity(), T::infinity()));
        }

        if D <= 2 {
            if self.coefficients[0].is_zero() || D == 1 {
                return OMatrix::<Complex<T>, Const<D>, U1>::from_element(Complex::new(T::nan(), T::nan()));
            } else {
                let mut roots = OMatrix::<Complex<T>, Const<D>, U1>::from_element(Complex::new(T::nan(), T::nan()));
                roots[0] = Complex::new(-self.coefficients[1] / self.coefficients[0], T::zero());
                roots
            }
        }
        else {
            self.companion().complex_eigenvalues()
        }
    }
}

/// Computes the roots of a univariate polynomial of degree `N` by calculating
/// the eigenvalues of its companion matrix. Also provides some edge case checks
/// on the validity of the solution.
///
/// # Arguments
/// * `coeff` - A slice of polynomial coefficients starting from the highest degree term.
///
/// # Returns
///
/// * `[T; N]` - a fixed-size array containing the roots of the polynomial.
///   If roots cannot be computed, it returns an array filled with NaNs (Not a Number).
///
/// ## Edge cases
///
/// - If `len(coeff) < 1`, the function will return a vector of NaNs.
/// - If the leading coefficient `coeff[0] == 0`, the polynomial is degenerate, and the function will attempt to handle the case by reducing the degree.
/// - If all coefficients are zero except the first one, the function assumes the polynomial is a constant and returns NaNs for all roots.
///
///
/// # Example
/// ```rust
/// use nalgebra::Complex;
/// use control_rs::polynomial::roots;
/// fn main() {
///     let coeff = [1.0, 2.0, 3.0, 4.0];
///     let mut rbuffer = [Complex::new(0.0, 0.0); 3];
///     let roots = roots::<f64>(&coeff, &mut rbuffer);
///     println!("{roots:?}");
/// }
/// ```
///
/// ## Edge Case Examples
/// - see basic_roots_tests
pub fn roots<T: Copy + Float + RealField>(coeff: &[T], root_buffer: &mut [Complex<T>]) {
    // less than 1 coeff has no roots unless len = 1 and coeff[0] = 0 then root[0] = inf
    match coeff.len() {
        0 => {
            root_buffer.fill(Complex::new(T::nan(), T::nan()));
            return;
        }
        1 => {
            if coeff[0].is_zero() {
                root_buffer[0] = Complex::new(T::infinity(), T::zero());
            } else {
                root_buffer.fill(Complex::new(T::nan(), T::nan()));
            }
            return;
        }
        _ => {}
    }

    // count coefficients equal to zero
    let num_zero_coeff = coeff.iter().fold(0, |acc, &c| match c == T::zero() {
        true => acc + 1,
        false => acc,
    });

    // all coeff are zero
    if coeff.len() == num_zero_coeff {
        // constant 0, everything is a root
        root_buffer[0] = Complex::new(T::infinity(), T::zero());
        return;
    }

    // first coeff is zero, recurse
    if coeff[0] == T::zero() {
        return roots::<T>(coeff[1..coeff.len()].try_into().unwrap(), root_buffer);
    }

    // Build the companion matrix
    let companion = DMatrix::from_fn(coeff.len() - 1, coeff.len() - 1, |i, j| {
        if i == 0 {
            Complex::new(-coeff[j + 1] / coeff[0], T::zero())
        } else {
            if i - 1 == j {
                Complex::new(T::one(), T::zero())
            } else {
                Complex::new(T::zero(), T::zero())
            }
        }
    });

    // Compute the eigenvalues of the companion matrix
    match companion.eigenvalues() {
        Some(eigenvalues) => eigenvalues
            .iter()
            .enumerate()
            .for_each(|(i, eigenvalue)| root_buffer[i] = *eigenvalue),
        None => root_buffer.fill(Complex::new(T::nan(), T::zero())),
    }
}

/// tests the trait for computing the roots of a polynomial.
#[cfg(test)]
pub mod basic_roots_tests {
    use super::*;

    #[test]
    fn test_simple_polynomial() {
        let coeff: [f64; 3] = [1.0, 1.0, 1.0]; // x^2 + x + 1
        let mut rbuffer = [Complex::new(0.0, 0.0); 2];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, -0.49999999999999994);
        assert_eq!(rbuffer[1].re, -0.5)
    }

    #[test]
    fn test_polynomial() {
        let coeff: [f64; 4] = [1.0, -6.0, 11.0, -6.0]; // x^3 - 6x^2 + 11x - 6 = 0
        let mut rbuffer = [Complex::new(0.0, 0.0); 3]; // x^3 - 6x^2 + 11x - 6 = 0
        roots(&coeff, &mut rbuffer);

        assert_eq!(rbuffer[0].re, 3.000000000000014);
        assert_eq!(rbuffer[1].re, 1.9999999999999991);
        assert_eq!(rbuffer[2].re, 0.9999999999999999);
    }

    #[test]
    fn test_linear_polynomial() {
        let coeff = [1.0, 0.0]; // y = x
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, 0.0);
    }

    #[test]
    fn test_constant_polynomial() {
        let coeff = [1.0]; // y = 1
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_constant_zero_polynomial() {
        let coeff = [0.0]; // y = 0
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_infinite());
    }

    #[test]
    fn test_leading_zero_constant_polynomial() {
        let coeff = [0.0, 1.0]; // y = 1
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_multiple_zero_coefficients() {
        let coeff = [0.0, 0.0]; // y = 0
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_infinite());
    }
}
