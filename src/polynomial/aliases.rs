//! Type aliases of [Polynomial].

use crate::Polynomial;
use core::ops::{DivAssign, MulAssign};
use nalgebra::{Const, DimSub, U1};
use num_traits::Zero;

/// Specialization of `Polynomial` that has only the constant term.
pub type Constant<T> = Polynomial<T, 1>;
/// Specialization of `Polynomial` that has a constant and linear term.
pub type Line<T> = Polynomial<T, 2>;
/// Specialization of `Polynomial` for quadratic equations.
pub type Quadratic<T> = Polynomial<T, 3>;
/// Specialization of `Polynomial` for cubic equations.
pub type Cubic<T> = Polynomial<T, 4>;
/// Specialization of `Polynomial` quartic equations.
pub type Quartic<T> = Polynomial<T, 5>;
/// Specialization of `Polynomial` for quintic equations.
pub type Quintic<T> = Polynomial<T, 6>;

/// # Polynomial<T, N> *= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let mut p1 = Polynomial::new([1, 0, 1]);
/// p1 *= Polynomial::new([2]);
/// assert_eq!(p1, Polynomial::new([2, 0, 2]));
/// ```
impl<T, const N: usize> MulAssign<Constant<T>> for Polynomial<T, N>
where
    T: Clone + MulAssign + Zero,
    Const<N>: DimSub<U1>,
{
    /// Multiplies the coefficients of a polynomial by a constant
    #[inline]
    fn mul_assign(&mut self, rhs: Constant<T>) {
        // SAFETY: rhs is a Constant 'N == 1`
        unsafe {
            *self *= rhs.get_unchecked(0).clone();
        }
    }
}

/// # Polynomial<T, N> /= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::Polynomial;
/// let mut p1 = Polynomial::new([2, 0, 2]);
/// p1 /= Polynomial::new([2]);
/// assert_eq!(p1, Polynomial::new([1, 0, 1]));
/// ```
impl<T, const N: usize> DivAssign<Constant<T>> for Polynomial<T, N>
where
    T: Clone + DivAssign + Zero,
    Const<N>: DimSub<U1>,
{
    /// Divides the coefficients of a polynomial by a constant
    #[inline]
    fn div_assign(&mut self, rhs: Constant<T>) {
        // SAFETY: rhs is a Constant 'N == 1`
        unsafe {
            *self /= rhs.get_unchecked(0).clone();
        }
    }
}
