//! Statically sized Polynomial

use super::*;

/// A static polynomial with D coefficients. *Similar to [nalgebra::SMatrix], this is a type alias
/// so not all of its methods are listed here. See [Polynomial] too* 
pub type SPolynomial<T, const D: usize> = Polynomial<T, Const<D>, ArrayStorage<T, D, 1>>;

impl<T, const D: usize> SPolynomial<T, D> {
    /// Create an [ArrayStorage] based [Polynomial]
    /// 
    /// This function uses a static array to initialize the coefficients of a polynomial. It is
    /// assumed the coefficients are sorted from highest to lowest degree, so the largest index
    /// refers to the constant term.
    /// 
    /// # Arguments
    /// 
    /// * `variable` - the variable of the polynomial
    /// * `coefficients` - an array of the coefficients, length = degree + 1
    /// 
    /// # Returns
    /// 
    /// * `Polynomial` - static polynomial
    pub const fn new(variable: &'static str, coefficients: [T; D]) -> Self {
        Self {
            variable,
            coefficients: ArrayStorage([coefficients]),
            _phantom: PhantomData,
        }
    }

    /// wrapper for [ArrayStorage::as_slice()]
    /// 
    /// # Returns
    /// 
    /// * `coeffs` - slice of coefficients sorted from high degree -> low degree
    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}