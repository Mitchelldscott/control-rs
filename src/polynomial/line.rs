//! Type alias of polynomial that implements a constant.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::Polynomial;

/// Specialization of `Polynomial` that has a constant and linear term.
pub type Line<T> = Polynomial<T, 2>;

// ===============================================================================================
//      Line-Scalar Arithmatic
// ===============================================================================================

/// # Polynomial<T, 2> + T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1, 0]);
/// let p2 = p1 + 1;
/// assert_eq!(*p2.constant().unwrap(), 1);
/// assert_eq!(*p2.leading_coefficient().unwrap(), 1);
/// ```
impl<T: Clone + Add<Output = T>> Add<T> for Line<T> {
    type Output = Line<T>;

    fn add(self, rhs: T) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 2, so the indices are always valid
            unsafe { self.get_unchecked(0).clone() + rhs },
            unsafe { self.get_unchecked(1).clone() },
        ])
    }
}

/// # Polynomial<T, 2> += T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1, 0]);
/// p1 += 1;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// assert_eq!(*p1.leading_coefficient().unwrap(), 1);
/// ```
impl<T: AddAssign> AddAssign<T> for Line<T> {
    fn add_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 2, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) += rhs;
        }
    }
}

/// # Polynomial<T, 2> - T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1, 1]);
/// let p2 = p1 - 1;
/// assert_eq!(*p2.constant().unwrap(), 0);
/// ```
impl<T: Clone + Sub<Output = T>> Sub<T> for Line<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 2, so the indices are always valid
            unsafe { self.get_unchecked(0).clone() - rhs },
            unsafe { self.get_unchecked(1).clone() },
        ])
    }
}

/// # Polynomial<T, 2> -= T
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1, 0]);
/// p1 -= 1;
/// assert_eq!(*p1.constant().unwrap(), -1);
/// ```
impl<T: SubAssign> SubAssign<T> for Line<T> {
    fn sub_assign(&mut self, rhs: T) {
        // SAFETY: `N` is 2, so the index is always valid
        unsafe {
            *self.get_unchecked_mut(0) -= rhs;
        }
    }
}

macro_rules! impl_line_left_scalar_ops {
    ($($scalar:ty),*) => {
        $(
            impl Add<Line<$scalar>> for $scalar {
                type Output = Line<$scalar>;

                fn add(self, rhs: Line<$scalar>) -> Self::Output {
                    Self::Output::from_data([
                        // SAFETY: `N` is 2, so the indices are always valid
                        unsafe { self - rhs.get_unchecked(0).clone() },
                        unsafe { rhs.get_unchecked(1).clone() },
                    ])
                }
            }
        )*
    };
}

macro_rules! impl_line_left_scalar_sub {
    ($($scalar:ty),*) => {
        $(
            impl Sub<Line<$scalar>> for $scalar {
                type Output = Line<$scalar>;

                fn sub(self, rhs: Line<$scalar>) -> Self::Output {
                    Self::Output::from_data([
                        // SAFETY: `N` is 2, so the indices are always valid
                        unsafe { self - rhs.get_unchecked(0).clone() },
                        unsafe { rhs.get_unchecked(1).clone().neg() },
                    ])
                }
            }

        )*
    };
}

impl_line_left_scalar_ops!(i8, u8, i16, u16, i32, u32, isize, usize, f32, f64);
impl_line_left_scalar_sub!(i8, i16, i32, isize, f32, f64);

// ===============================================================================================
//      Line-Empty Polynomial Arithmatic
// ===============================================================================================

/// # Polynomial<T, 2> + Polynomial<T, 0>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1, 0]);
/// let p2 = Polynomial::new([]);
/// let p3 = p1 + p2;
/// assert_eq!(*p3.constant().unwrap(), 0);
/// assert_eq!(*p3.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone> Add<Polynomial<T, 0>> for Line<T> {
    type Output = Self;

    fn add(self, _rhs: Polynomial<T, 0>) -> Self::Output {
        self.clone()
    }
}

/// # Polynomial<T, 2> += Polynomial<T, 0>
///
/// This function has no effect, it is only implemented for completeness.
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1, 0]);
/// let p2 = Polynomial::new([]);
/// p1 += p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// assert_eq!(*p1.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T> AddAssign<Polynomial<T, 0>> for Line<T> {
    fn add_assign(&mut self, _rhs: Polynomial<T, 0>) {}
}

/// # Polynomial<T, 2> - Polynomial<T, 0>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([1, 0]);
/// let p2 = Polynomial::new([]);
/// let p3 = p1 - p2;
/// assert_eq!(*p3.constant().unwrap(), 0);
/// assert_eq!(*p3.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone> Sub<Polynomial<T, 0>> for Line<T> {
    type Output = Self;

    fn sub(self, _rhs: Polynomial<T, 0>) -> Self::Output {
        self.clone()
    }
}

/// # Polynomial<T, 2> -= Polynomial<T, 0>
///
/// This function has no effect, it is only implemented for completeness.
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([1, 0]);
/// let p2 = Polynomial::new([]);
/// p1 -= p2;
/// assert_eq!(*p1.constant().unwrap(), 0);
/// assert_eq!(*p1.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T> SubAssign<Polynomial<T, 0>> for Line<T> {
    fn sub_assign(&mut self, _rhs: Polynomial<T, 0>) {}
}

// ===============================================================================================
//      Line-Constant Arithmatic
// ===============================================================================================

// ===============================================================================================
//      Constant-Line Arithmatic
// ===============================================================================================

// ===============================================================================================
//      Line-Line Arithmatic
// ===============================================================================================

/// # Polynomial<T, 2> + Polynomial<T, 2>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([0, 0]);
/// let p2 = Polynomial::new([1, 1]);
/// let p3 = p1 + p2;
/// assert_eq!(*p3.constant().unwrap(), 1);
/// assert_eq!(*p3.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Add<Output = T>> Add for Line<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 2, so the index is always valid
            unsafe { self.get_unchecked(0).clone() + rhs.get_unchecked(0).clone() },
            unsafe { self.get_unchecked(1).clone() + rhs.get_unchecked(1).clone() },
        ])
    }
}

/// # Polynomial<T, 2> += Polynomial<T, 2>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0, 0]);
/// let p2 = Polynomial::new([1, 1]);
/// p1 += p2;
/// assert_eq!(*p1.constant().unwrap(), 1);
/// assert_eq!(*p1.leading_coefficient().unwrap(), 1);
/// ```
/// TODO: Unit Test
impl<T: Clone + AddAssign> AddAssign for Line<T> {
    fn add_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.iter_mut().zip(rhs.iter()) {
            *a_i += b_i.clone();
        }
    }
}

/// # Polynomial<T, 2> - Polynomial<T, 2>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0, 0]);
/// let p2 = Polynomial::new([1, 1]);
/// let p3 = p1 - p2;
/// assert_eq!(*p3.constant().unwrap(), -1);
/// ```
/// TODO: Unit Test
impl<T: Clone + Sub<Output = T>> Sub for Line<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_data([
            // SAFETY: `N` is 2, so the index is always valid
            unsafe { self.get_unchecked(0).clone() - rhs.get_unchecked(0).clone() },
            unsafe { self.get_unchecked(1).clone() - rhs.get_unchecked(1).clone() },
        ])
    }
}

/// # Polynomial<T, 2> -= Polynomial<T, 2>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let mut p1 = Polynomial::new([0, 0]);
/// let p2 = Polynomial::new([1, 1]);
/// p1 -= p2;
/// assert_eq!(*p1.constant().unwrap(), -1);
/// ```
/// TODO: Unit Test
impl<T: Clone + SubAssign> SubAssign for Line<T> {
    fn sub_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.iter_mut().zip(rhs.iter()) {
            *a_i -= b_i.clone();
        }
    }
}

/// # Polynomial<T, 2> * Polynomial<T, 2>
///
/// # Example
/// ```
/// use control_rs::polynomial::Polynomial;
/// let p1 = Polynomial::new([2, 2]);
/// let p2 = Polynomial::new([1, 1]);
/// let p3 = p1 * p2;
/// assert_eq!(*p3.constant().unwrap(), 2);
/// assert_eq!(p3.degree().unwrap(), 2);
/// ```
/// TODO: Unit Test
impl<T: Clone + Add<Output = T> + Mul<Output = T>> Mul for Line<T> {
    type Output = Polynomial<T, 3>;

    fn mul(self, rhs: Self) -> Self::Output {
        Polynomial::<T, 3>::from_data([
            // SAFETY: `N` is 2, so the index is always valid
            unsafe { self.get_unchecked(0).clone() * rhs.get_unchecked(0).clone() },
            unsafe {
                self.get_unchecked(0).clone() * rhs.get_unchecked(1).clone()
                    + self.get_unchecked(1).clone() * rhs.get_unchecked(0).clone()
            },
            unsafe { self.get_unchecked(1).clone() * rhs.get_unchecked(1).clone() },
        ])
    }
}
//
// /// # Polynomial<T, 1> / Polynomial<T, 1>
// ///
// /// # Example
// /// ```
// /// use control_rs::polynomial::Polynomial;
// /// let p1 = Polynomial::new([2]);
// /// let p2 = Polynomial::new([2]);
// /// let p3 = p1 / p2;
// /// assert_eq!(*p3.constant().unwrap(), 1);
// /// ```
// /// TODO: Unit Test
// impl<T: Clone + Div<Output = T>> Div for Line<T> {
//     type Output = Self;
//
//     fn div(self, rhs: Self) -> Self::Output {
//         Self::from_data([
//             // SAFETY: `N` is 1, so the index is always valid
//             unsafe { self.get_unchecked(0).clone() / rhs.get_unchecked(0).clone() },
//         ])
//     }
// }
//
// /// # Polynomial<T, 1> /= Polynomial<T, 1>
// ///
// /// # Example
// /// ```
// /// use control_rs::polynomial::Polynomial;
// /// let mut p1 = Polynomial::new([2]);
// /// let p2 = Polynomial::new([2]);
// /// p1 /= p2;
// /// assert_eq!(*p1.constant().unwrap(), 1);
// /// ```
// /// TODO: Unit Test
// impl<T: Clone + DivAssign> DivAssign for Line<T> {
//     fn div_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe {
//             *self.get_unchecked_mut(0) /= rhs.get_unchecked(0).clone();
//         }
//     }
// }
//
// /// # Polynomial<T, 1> % Polynomial<T, 1>
// ///
// /// # Example
// /// ```
// /// use control_rs::polynomial::Polynomial;
// /// let p1 = Polynomial::new([2]);
// /// let p2 = Polynomial::new([2]);
// /// let p3 = p1 % p2;
// /// assert_eq!(*p3.constant().unwrap(), 0);
// /// ```
// /// TODO: Unit Test
// impl<T: Clone + Rem<Output = T>> Rem for Line<T> {
//     type Output = Self;
//
//     fn rem(self, rhs: Self) -> Self::Output {
//         Self::from_data([
//             // SAFETY: `N` is 1, so the index is always valid
//             unsafe { self.get_unchecked(0).clone() % rhs.get_unchecked(0).clone() },
//         ])
//     }
// }
//
// /// # Polynomial<T, 1> %= Polynomial<T, 1>
// ///
// /// # Example
// /// ```
// /// use control_rs::polynomial::Polynomial;
// /// let mut p1 = Polynomial::new([2]);
// /// let p2 = Polynomial::new([2]);
// /// p1 %= p2;
// /// assert_eq!(*p1.constant().unwrap(), 0);
// /// ```
// /// TODO: Unit Test
// impl<T: Clone + RemAssign> RemAssign for Line<T> {
//     fn rem_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe {
//             *self.get_unchecked_mut(0) %= rhs.get_unchecked(0).clone();
//         }
//     }
// }
