//! Type alias of polynomial that implements a constant.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use crate::Polynomial;

/// Specialization of `Polynomial` that has only the constant term.
pub type Constant<T> = Polynomial<T, 1>;

// ===============================================================================================
//      Constant-Scalar Arithmatic
// ===============================================================================================

/// # Polynomial<T, 1> + T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// ```
impl<T: Clone + Add<Output = T>> Add<T> for Constant<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 1, so the index is always valid
            unsafe { self.get_unchecked(0).clone() + rhs },
        ])
    }
}

/// # Polynomial<T, 1> += T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 += 1;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
impl<T: AddAssign> AddAssign<T> for Constant<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) += rhs;
        }
    }
}

/// # Polynomial<T, 1> - T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = p1 - 1;
/// assert_eq!(*p2.constant().unwrap(), -1);
/// ```
impl<T: Clone + Sub<Output = T>> Sub<T> for Constant<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 1, so the index is always valid
            unsafe { self.get_unchecked(0).clone() - rhs },
        ])
    }
}

/// # Polynomial<T, 1> -= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// p1 -= 1;
/// assert_eq!(*p1.constant().unwrap(), -1);
/// ```
impl<T: SubAssign> SubAssign<T> for Constant<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) -= rhs;
        }
    }
}

macro_rules! impl_constant_left_scalar_ops {
    ($($scalar:ty),*) => {
        $(
            impl Add<Constant<$scalar>> for $scalar {
                type Output = Constant<$scalar>;
                #[inline(always)]
                fn add(self, rhs: Constant<$scalar>) -> Self::Output {
                    Self::Output::from_data([
                        // SAFETY: `N` is 1, so the index is always valid
                        unsafe { self.clone() + rhs.get_unchecked(0) },
                    ])
                }
            }
            impl AddAssign<Constant<$scalar>> for $scalar {
                #[inline(always)]
                fn add_assign(&mut self, rhs: Constant<$scalar>) {
                    // SAFETY: `N` is 1, so the index is always valid
                    unsafe {
                        *self += rhs.get_unchecked(0);
                    }
                }
            }
            impl SubAssign<Constant<$scalar>> for $scalar {
                #[inline(always)]
                fn sub_assign(&mut self, rhs: Constant<$scalar>) {
                    // SAFETY: `N` is 1, so the index is always valid
                    unsafe {
                        *self -= rhs.get_unchecked(0);
                    }
                }
            }
            impl Div<Constant<$scalar>> for $scalar {
                type Output = Constant<$scalar>;
                #[inline(always)]
                fn div(self, rhs: Constant<$scalar>) -> Self::Output {
                    Self::Output::from_data([
                        // SAFETY: `N` is 1, so the index is always valid
                        unsafe { self.clone() / rhs.get_unchecked(0) },
                    ])
                }
            }
        )*
    };
}

macro_rules! impl_constant_left_scalar_sub {
    ($($scalar:ty),*) => {
        $(
            impl Sub<Constant<$scalar>> for $scalar {
                type Output = Constant<$scalar>;
                #[inline(always)]
                fn sub(self, rhs: Constant<$scalar>) -> Self::Output {
                    Polynomial::from_data([
                        // SAFETY: `N` is 1, so the index is always valid
                        unsafe { self.clone() - rhs.get_unchecked(0) }
                    ])
                }
            }

        )*
    };
}

impl_constant_left_scalar_ops!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);
impl_constant_left_scalar_sub!(i8, i16, i32, isize, f32, f64);

// ===============================================================================================
//      Constant-Generic Arithmatic
// ===============================================================================================

/// # Polynomial<T, 1> + Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0]);
/// let p2 = Polynomial::new([1, 1]);
/// let p3 = p1 + p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// ```
impl<T, const N: usize> Add<Polynomial<T, N>> for Constant<T>
where
    T: Clone + Add<Polynomial<T, N>, Output = Polynomial<T, N>>,
{
    type Output = Polynomial<T, N>;
    #[inline]
    fn add(self, rhs: Polynomial<T, N>) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe { self.get_unchecked(0).clone() + rhs }
    }
}

/// # Polynomial<T, 1> - Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([1, 1]);
/// let p3 = p1 - p2;
/// assert_eq!(*p3.constant().unwrap(), 0);
/// ```
impl<T, const N: usize> Sub<Polynomial<T, N>> for Constant<T>
where
    T: Clone + Sub<Polynomial<T, N>, Output = Polynomial<T, N>>,
{
    type Output = Polynomial<T, N>;
    #[inline]
    fn sub(self, rhs: Polynomial<T, N>) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe { self.get_unchecked(0).clone() - rhs }
    }
}

/// # Polynomial<T, 1> * Polynomial<T, N>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// let p3 = p1 * p2;
/// assert_eq!(*p3.constant().unwrap(), 4);
/// ```
impl<T, const N: usize> Mul<Polynomial<T, N>> for Constant<T>
where
    T: Clone + Mul<Polynomial<T, N>, Output = Polynomial<T, N>>,
{
    type Output = Polynomial<T, N>;
    #[inline]
    fn mul(self, rhs: Polynomial<T, N>) -> Self::Output {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe { self.get_unchecked(0).clone() * rhs }
    }
}

// ===============================================================================================
//      Constant-Empty Polynomial Arithmatic
// ===============================================================================================

/// # Polynomial<T, 1> += Polynomial<T, 0>
///
/// This function has no effect, it is only implemented for completeness.
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([]);
/// p1 += p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
impl<T> AddAssign<Polynomial<T, 0>> for Constant<T> {
    fn add_assign(&mut self, _rhs: Polynomial<T, 0>) {}
}

/// # Polynomial<T, 1> -= Polynomial<T, 0>
///
/// This function has no effect, it is only implemented for completeness.
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([]);
/// p1 -= p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
impl<T> SubAssign<Polynomial<T, 0>> for Constant<T> {
    fn sub_assign(&mut self, _rhs: Polynomial<T, 0>) {}
}

// ===============================================================================================
//      Constant-Constant Arithmatic
// ===============================================================================================

/// # Polynomial<T, 1> += Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0]);
/// let p2 = Polynomial::new([1]);
/// p1 += p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
impl<T: Clone + AddAssign> AddAssign for Constant<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) += rhs.get_unchecked(0).clone();
        }
    }
}

/// # Polynomial<T, 1> -= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1]);
/// let p2 = Polynomial::new([1]);
/// p1 -= p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// ```
impl<T: Clone + SubAssign> SubAssign for Constant<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) -= rhs.get_unchecked(0).clone();
        }
    }
}

/// # Polynomial<T, 1> *= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 *= p2;
/// assert_eq!(*p1.constant().unwrap(), 4);
/// ```
impl<T: Clone + MulAssign> MulAssign for Constant<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) *= rhs.get_unchecked(0).clone();
        }
    }
}

/// # Polynomial<T, 1> / Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// let p3 = p1 / p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// ```
impl<T: Clone + Div<Output = T>> Div for Constant<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 1, so the index is always valid
            unsafe { self.get_unchecked(0).clone() / rhs.get_unchecked(0).clone() },
        ])
    }
}

/// # Polynomial<T, 1> /= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 /= p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// ```
impl<T: Clone + DivAssign> DivAssign for Constant<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) /= rhs.get_unchecked(0).clone();
        }
    }
}

/// # Polynomial<T, 1> % Polynomial<T, 1>
///
/// # Example
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
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 1, so the index is always valid
            unsafe { self.get_unchecked(0).clone() % rhs.get_unchecked(0).clone() },
        ])
    }
}

/// # Polynomial<T, 1> %= Polynomial<T, 1>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([2]);
/// let p2 = Polynomial::new([2]);
/// p1 %= p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// ```
/// TODO: Unit Test
impl<T: Clone + RemAssign> RemAssign for Constant<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        // SAFETY: `N` is 1, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) %= rhs.get_unchecked(0).clone();
        }
    }
}
