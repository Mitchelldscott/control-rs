//! This module is meant to provide a type export similar to [nalgebra::DMatrix]. DPolynomial will
//! implement Polynomial functions using a VecStorage.
//! 

use super::*;
use nalgebra::{Dyn, VecStorage};

/// A dynamic polynomial with D coefficients. *Similar to [nalgebra::DMatrix], this is a type alias
/// so not all of its methods are listed here. See [Polynomial] too* 
pub type DPolynomial<T> = Polynomial<T, Dyn, VecStorage<T, Dyn, U1>>;


impl<T> DPolynomial<T> {
    /// Create a new Vec based [Polynomial]
    /// 
    /// This function uses a [Vec] to initialize the coefficients
    /// of a polynomial.
    /// 
    /// # Arguments
    /// 
    /// * `variable` - the variable of the polynomial
    /// * `coefficients` - a vec of the coefficients, length = degree + 1
    /// 
    /// # Returns
    /// 
    /// * `Polynomial` - A polynomial using [nalgebra::VecStorage]
    pub fn from_vec(variable: &'static str, coefficients: Vec<T>) -> Self {
        Self {
            variable,
            coefficients: VecStorage::new(Dyn(coefficients.len()), U1, coefficients),
            _phantom: PhantomData,
        }
    }

    /// returns the coefficients as a slice.
    /// 
    /// wrapper for [nalgebra::VecStorage::as_slice()]
    /// 
    /// # Returns
    /// 
    /// * `coeffs` - slice of coefficients sorted from high degree -> low degree
    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}