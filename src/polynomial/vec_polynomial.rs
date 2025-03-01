//! Dynamically sized Polynomial

use super::*;
use nalgebra::{Dyn, VecStorage};

/// A dynamic polynomial with D coefficients. *Similar to [nalgebra::DMatrix], this is a type alias
/// so not all of its methods are listed here. See [Polynomial] too*
pub type DPolynomial<T> = Polynomial<T, Dyn, VecStorage<T, Dyn, U1>>;

impl<T> DPolynomial<T> {
    /// Create a [VecStorage] based [Polynomial]
    ///
    /// This function uses a [Vec] to initialize the coefficients of a polynomial. It is
    /// assumed the coefficients are sorted from highest to lowest degree, so the largest index
    /// refers to the constant term.
    ///
    /// # Arguments
    ///
    /// * `variable` - the variable of the polynomial
    /// * `coefficients` - a vec of the coefficients, length = degree + 1
    ///
    /// # Returns
    ///
    /// * `Polynomial` - dynamic polynomial
    pub fn from_vec(variable: &'static str, coefficients: Vec<T>) -> Self {
        Self {
            variable,
            coefficients: VecStorage::new(Dyn(coefficients.len()), U1, coefficients),
            _phantom: PhantomData,
        }
    }

    /// wrapper for [VecStorage::as_slice()]
    ///
    /// # Returns
    ///
    /// * `coeffs` - slice of coefficients sorted from high degree -> low degree
    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}
