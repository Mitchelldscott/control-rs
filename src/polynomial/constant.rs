//! Type alias of polynomial that implements a constant.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use crate::Polynomial;

/// Specialization of `Polynomial` that has only the constant term.
pub type Constant<T> = Polynomial<T, 1>;


// ===============================================================================================
//      Polynomial-Scalar Arithmatic
// ===============================================================================================

/// Implementation of [Add] for Polynomial<T, 1> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Add<Output = T>> Add<T> for Constant<T> {
    type Output = Constant<T>;

    fn add(self, rhs: T) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone() + rhs
            ])
        }
    }
}

/// Implementation of [AddAssign] for Polynomial<T, 1> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 += 1;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: AddAssign> AddAssign<T> for Constant<T> {
    fn add_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe { *self.get_unchecked_mut(0) += rhs; }
    }
}

/// Implementation of [Sub] for Polynomial<T, 1> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Sub<Output = T>> Sub<T> for Constant<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone() - rhs
            ])
        }
    }
}

/// Implementation of [SubAssign] for Polynomial<T, 1> and T
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 -= 1;
/// assert_eq!(*p1.constant().unwrap(), -1);
/// ```
/// TODO: Unit Test
impl<T: SubAssign> SubAssign<T> for Constant<T> {
    fn sub_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) -= rhs;
        }
    }
}

// ===============================================================================================
//      Polynomial-Polynomial Arithmatic
// ===============================================================================================

/// Implementation of [Add] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = Polynomial::new([1]);
/// let p3 = p1 + p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Add<Output = T>> Add for Constant<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone()
                    + rhs.get_unchecked(0).clone()
            ])
        }
    }
}

/// Implementation of [AddAssign] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// let p2 = Polynomial::new([1]);
/// p1 += p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + AddAssign> AddAssign for Constant<T> {
    fn add_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) += rhs.get_unchecked(0).clone();
        }
    }
}

/// Implementation of [Sub] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([1]);
/// let p3 = p1 - p2;
/// assert_eq!(*p3.constant().unwrap(), 0);
/// ```
/// TODO: Unit Test
impl<T: Clone + Sub<Output = T>> Sub for Constant<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone()
                    - rhs.get_unchecked(0).clone()
            ])
        }
    }
}

/// Implementation of [SubAssign] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([1]);
/// p1 -= p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// ```
/// TODO: Unit Test
impl<T: Clone + SubAssign> SubAssign for Constant<T> {
    fn sub_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) -= rhs.get_unchecked(0).clone();
        }
    }
}

/// Implementation of [Mul] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// let p3 = p1 * p2;
/// assert_eq!(*p3.constant().unwrap(), 4);
/// ```
/// TODO: Unit Test
impl<T: Clone + Mul<Output = T>> Mul for Constant<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone() * rhs.get_unchecked(0).clone()
            ])
        }
    }
}

/// Implementation of [MulAssign] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 *= p2;
/// assert_eq!(*p1.constant().unwrap(), 4);
/// ```
/// TODO: Unit Test
impl<T: Clone + MulAssign> MulAssign for Constant<T> {
    fn mul_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.coefficients.get_unchecked_mut(0) *= rhs.coefficients.get_unchecked(0).clone();
        }
    }
}

/// Implementation of [Div] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// let p3 = p1 / p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Div<Output = T>> Div for Constant<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone() / rhs.get_unchecked(0).clone()
            ])
        }
    }
}

/// Implementation of [DivAssign] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 /= p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + DivAssign> DivAssign for Constant<T> {
    fn div_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) /= rhs.get_unchecked(0).clone();
        }
    }
}

/// Implementation of [Rem] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// let p3 = p1 % p2;
/// assert_eq!(*p3.constant().unwrap(), 0);
/// ```
/// TODO: Unit Test
impl<T: Clone + Rem<Output = T>> Rem for Constant<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            Self::from_data([
                self.get_unchecked(0).clone() % rhs.get_unchecked(0).clone()
            ])
        }
    }
}

/// Implementation of [RemAssign] for Polynomial<T, 1> and Polynomial<T, 1>
///
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 %= p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// ```
/// TODO: Unit Test
impl<T: Clone + RemAssign> RemAssign for Constant<T> {
    fn rem_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) %= rhs.get_unchecked(0).clone();
        }
    }
}
