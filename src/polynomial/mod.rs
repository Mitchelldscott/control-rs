//!
//! # Polynomial trait and default implementations
//!
//!
use nalgebra::{
    allocator::Allocator, Complex, Const, DefaultAllocator, DimDiff, DimSub, OMatrix, RealField, U1,
};
use num_traits::Num;

#[cfg(feature = "std")]
use std::{
    fmt,
    ops::{Add, Sub, Mul, Div, Neg, Index},
};

#[cfg(not(feature = "std"))]
use core::{
    fmt,
    ops::{Add, Sub, Mul, Div, Neg, Index},
};


#[cfg(test)]
mod edge_case_test;

#[cfg(test)]
mod fmt_tests;

/// static array of coefficients for a polynomial of degree `D - 1`.
/// 
/// This struct stores the coefficients of a polynomial a(s): 
/// <pre>
/// a(s) = a_0 * s^(D-1) + a_1 * s^(D-2) + ... + a_(D-2)] * s + a_(D-1)]
/// </pre>
/// 
/// The coefficients are stored in descending degree order (i.e. `[a_0, a_1... a_(D-1)]`).
/// 
/// # Generic Arguments
/// 
/// * `T` - Type of the coefficients
/// * `D` - Length of the coefficient array, NOT the degree!
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Polynomial<T, const D: usize> {
    /// string indicating the variable the polynomial represents
    pub variable: &'static str,
    /// coefficients of the polynomial stored in descending degree order
    pub coefficients: [T; D],
}

impl<T, const D: usize> Polynomial<T, D> {
    /// Create a new polynomial with the given coefficients.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - An array of coefficients of the polynomial in descending degree order.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D>` - A new polynomial with the given coefficients.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<f64, 3>::new(b'x', [1.0, 2.0, 3.0]);
    /// ```
    pub const fn new(variable: &'static str, coefficients: [T; D]) -> Self {
        Polynomial {
            variable,
            coefficients,
        }
    }
}

impl<T: Copy, const D: usize> Polynomial<T, D> {
    /// Get the constant term of the polynomial.
    pub fn constant(&self) -> T {
        assert!(D > 0, "Polynomial length 0 does not have a constant term");
        self.coefficients[D - 1]
    }
}

impl<T: Copy + Default, const D: usize> Default for Polynomial<T, D> {
    /// Create a new polynomial with the given coefficients.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - An array of coefficients of the polynomial in descending degree order.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D>` - A new polynomial with the given coefficients.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::<i32, 6>::default();
    /// ```
    fn default() -> Self {
        Polynomial {
            variable: "x",
            coefficients: [T::default(); D],
        }
    }
}

impl<T: Copy + Num, const D: usize> Polynomial<T, D> {
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
    /// let p = Polynomial::new("x", [1.0, 2.0, 3.0]);
    /// let value = p.evaluate(2.0);
    /// ```
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Copy + Num + Add<T, Output = U> + Mul<U, Output = U>,
    {
        self.coefficients
            .iter()
            .fold(U::zero(), |acc, &c| acc * value + c)
    }

    /// Compute the derivative of the polynomial.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D - 1>` - The derivative of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new(b'x', [1.0, 2.0, 3.0]);
    /// let derivative: Polynomial::<f64, 2, 1> = p.derivative();
    /// ```
    pub fn derivative<const D1: usize>(&self, variable: &'static str) -> Polynomial<T, D1> {
        let mut coefficients = [T::zero(); D1];
        for i in 0..D1 {
            coefficients[i] = (0..D1 - i).fold(T::zero(), |acc, j| acc + self.coefficients[j]);
        }

        Polynomial {
            variable,
            coefficients,
        }
    }
}

impl<T, const D:usize> Polynomial<T, D> 
where 
    T: 'static + Copy + Num + Neg<Output = T> + fmt::Debug
{
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
    pub fn companion(&self) -> OMatrix<T, Const<D>, Const<D>> {
        OMatrix::<T, Const<D>, Const<D>>::from_fn(|i, j| {
            if i == 0 {
                -self.coefficients[j]
            } else {
                if i - 1 == j {
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
    pub fn roots(&self) -> Option<OMatrix<Complex<T>, Const<D>, U1>>
    where
        T: RealField,
        Const<D>: DimSub<U1>,
        DefaultAllocator: Allocator<Const<D>, DimDiff<Const<D>, U1>>
            + Allocator<DimDiff<Const<D>, U1>>
            + Allocator<Const<D>, Const<D>>
            + Allocator<Const<D>>,
    {
        if D <= 1 || self
            .coefficients
            .iter()
            .fold(true, |acc, &c| match c == T::zero() {
                true => acc,
                false => false,
        }) {
            return None;
        }

        if D == 2 {
            if self.coefficients[0].is_zero() {
                None
            } else {
                let mut roots = OMatrix::<Complex<T>, Const<D>, U1>::from_element(Complex::new(
                    T::zero(),
                    T::zero(),
                ));
                roots[0] = Complex::new(-self.coefficients[1] / self.coefficients[0], T::zero());
                Some(roots)
            }
        } else {
            Some(self.companion().complex_eigenvalues())
        }
    }
}

// ===============================================================================================
//      Polynomial Display Implementation
// ===============================================================================================

impl<T, const D: usize> fmt::Display for Polynomial<T, D>
where
    T: Copy + Num + PartialOrd + Neg<Output = T> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            if coeff == T::zero() {
                continue;
            }

            if i > 0 {
                write!(f, " {} ", if coeff >= T::zero() { "+" } else { "-" })?;
            } else if coeff < T::zero() {
                write!(f, "-")?;
            }

            let abs_coeff = if coeff < T::zero() { -coeff } else { coeff };
            let exp = D - 1 - i;

            if abs_coeff != T::one() || exp == 0 {
                write!(f, "{}", abs_coeff)?;
            }

            if exp > 0 {
                write!(f, "{}", self.variable)?;
                if exp > 1 {
                    write!(f, "^{}", exp)?;
                }
            }
        }

        Ok(())
    }
}

// ===============================================================================================
//      Polynomial Operator Implementations
// ===============================================================================================

//
//  Index
//

impl<T, const D: usize> Index<usize> for Polynomial<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}

//
//  Scalar Math
//

impl<T, U, const D: usize> Add<U> for Polynomial<T, D>
where
    T: Copy + Add<U, Output = T>,
{
    type Output = Self;

    fn add(self, rhs: U) -> Self::Output {
        let mut new_poly = self;
        if D > 0 {
            new_poly.coefficients[D-1] = self.coefficients[D-1] + rhs;
        }
        new_poly
    }
}

impl<T, U, const D: usize> Sub<U> for Polynomial<T, D>
where
    T: Copy + Sub<U, Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: U) -> Self::Output {
        let mut new_poly = self;
        if D > 0 {
            new_poly.coefficients[D-1] = self.coefficients[D-1] - rhs;
        }
        new_poly
    }
}

impl<T, U, const D: usize> Mul<U> for Polynomial<T, D>
where
    U: Copy,
    T: Copy + Mul<U, Output = T>,
{
    type Output = Polynomial<T, D>;

    fn mul(self, rhs: U) -> Self::Output {
        let mut new_poly = self;
        for i in 0..D {
            new_poly.coefficients[i] = self.coefficients[i] * rhs;
        }
        new_poly
    }
}

impl<T, U, const D: usize> Div<U> for Polynomial<T, D>
where
    U: Copy,
    T: Copy + Div<U, Output = T>,
{
    type Output = Self;

    fn div(self, rhs: U) -> Self::Output {
        let mut new_poly = self;
        for i in 0..D {
            new_poly.coefficients[i] = self.coefficients[i] / rhs;
        }
        new_poly
    }
}