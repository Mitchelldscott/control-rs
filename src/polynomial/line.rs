// //! Type alias of polynomial that implements a linear function.
//
// use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
//
// use crate::Polynomial;
// use crate::polynomial::Constant;
//
// /// Specialization of `Polynomial` that has only the constant term.
// pub type Line<T> = Polynomial<T, 2>;
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Add<Output = T>> Add<T> for Line<T> {
//     type Output = Line<T>;
//
//     fn add(self, rhs: T) -> Self::Output {
//         // SAFETY: `N` is 2, so the indices 0 and 1 are always valid
//         unsafe {
//             Self::from_data([
//                 self.coefficients.get_unchecked(0).clone() + rhs,
//                 self.coefficients.get_unchecked(1).clone(),
//             ])
//         }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: AddAssign> AddAssign<T> for Line<T> {
//     fn add_assign(&mut self, rhs: T) {
//         // SAFETY: `N` is 2, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) += rhs; }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Sub<Output = T>> Sub<T> for Line<T> {
//     type Output = Self;
//
//     fn sub(self, rhs: T) -> Self::Output {
//         // SAFETY: `N` is 2, so the indices 0 and 1 are always valid
//         unsafe {
//             Self::from_data([
//                 self.coefficients.get_unchecked(0).clone() - rhs,
//                 self.coefficients.get_unchecked(1).clone(),
//             ])
//         }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: SubAssign> SubAssign<T> for Line<T> {
//     fn sub_assign(&mut self, rhs: T) {
//         // SAFETY: `N` is 2, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) -= rhs; }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Mul<Output = T>> Mul<T> for Line<T> {
//     type Output = Self;
//
//     fn mul(self, rhs: T) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() * rhs]) }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + MulAssign> MulAssign<T> for Line<T> {
//     fn mul_assign(&mut self, rhs: T) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) *= rhs; }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Div<Output = T>> Div<T> for Line<T> {
//     type Output = Self;
//
//     fn div(self, rhs: T) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() / rhs]) }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + DivAssign> DivAssign<T> for Line<T> {
//     fn div_assign(&mut self, rhs: T) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) /= rhs; }
//     }
// }
//
// /// TODO: Repair + Doc + Unit Test + Example
// impl<T: Clone + Rem<Output = T>> Rem<T> for Line<T> {
//     type Output = Self;
//
//     fn rem(self, rhs: T) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() % rhs]) }
//     }
// }
//
// /// TODO: Repair + Doc + Unit Test + Example
// impl<T: Clone + RemAssign> RemAssign<T> for Line<T> {
//     fn rem_assign(&mut self, rhs: T) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) %= rhs; }
//     }
// }
//
// // ===============================================================================================
// //      Polynomial-Polynomial Arithmatic
// // ===============================================================================================
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Add<Output = T>> Add for Line<T> {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() + rhs.coefficients.get_unchecked(0).clone()]) }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + AddAssign> AddAssign for Line<T> {
//     fn add_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) += rhs.coefficients.get_unchecked(0).clone(); }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Sub<Output = T>> Sub for Line<T> {
//     type Output = Self;
//
//     fn sub(self, rhs: Self) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() - rhs.coefficients.get_unchecked(0).clone()]) }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + SubAssign> SubAssign for Line<T> {
//     fn sub_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) -= rhs.coefficients.get_unchecked(0).clone(); }
//     }
// }
//
// /// TODO: Cleanup + Doc + Unit Test + Example
// impl<T: Clone + Mul<Output = T>> Mul for Line<T> {
//     type Output = Self;
//
//     fn mul(self, rhs: Self) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() * rhs.coefficients.get_unchecked(0).clone()]) }
//     }
// }
//
// /// # TODO: Repair + Doc + Unit Test + Example
// impl<T: Clone + MulAssign> MulAssign for Line<T> {
//     fn mul_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) *= rhs.coefficients.get_unchecked(0).clone(); }
//     }
// }
//
// /// # TODO: Doc + Unit Test + Example
// impl<T: Clone + Div<Output = T>> Div for Line<T> {
//     type Output = Self;
//
//     fn div(self, rhs: Self) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() / rhs.coefficients.get_unchecked(0).clone()]) }
//     }
// }
//
// /// # TODO: Doc + Unit Test + Example
// impl<T: Clone + DivAssign> DivAssign for Line<T> {
//     fn div_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) /= rhs.coefficients.get_unchecked(0).clone(); }
//     }
// }
//
// /// # TODO: Doc + Unit Test + Example
// impl<T: Clone + Rem<Output = T>> Rem for Line<T> {
//     type Output = Self;
//
//     fn rem(self, rhs: Self) -> Self::Output {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { Self::from_data([self.coefficients.get_unchecked(0).clone() % rhs.coefficients.get_unchecked(0).clone()]) }
//     }
// }
//
// /// # TODO: Doc + Unit Test + Example
// impl<T: Clone + RemAssign> RemAssign for Line<T> {
//     fn rem_assign(&mut self, rhs: Self) {
//         // SAFETY: `N` is 1, so the index is always valid
//         unsafe { *self.coefficients.get_unchecked_mut(0) %= rhs.coefficients.get_unchecked(0).clone(); }
//     }
// }
