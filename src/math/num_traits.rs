// //! Minimal copy of [num_traits](https://github.com/rust-num/num-traits) with minor adjustments
// //!
// //! Changes:
// //!     * Num renamed to Number
// //!     * NumOps renamed to Arithmatic
// //!     * Number does not require string conversion
// //!     * Zero requires implementing a constant value
// //!     * Zero requires implementing is_zero()
// //!     * One requires implementing a constant value
// //!     * One requires implementing is_one()
//
// #[cfg(feature = "std")]
// use std::ops::{Add, Div, Mul, Rem, Sub};
//
// #[cfg(not(feature = "std"))]
// use core::ops::{Add, Div, Mul, Rem, Sub};
//
// /// Defines an additive identity element for `Self`.
// ///
// /// # Laws
// ///
// /// ```text
// /// a + 0 = a       ∀ a ∈ Self
// /// 0 + a = a       ∀ a ∈ Self
// /// ```
// pub trait Zero: Sized + Add<Self, Output = Self> {
//     /// The additive identity element of `Self`, `0`.
//     const ZERO: Self;
//
//     /// Returns the additive identity element of `Self`, `0`.
//     /// # Purity
//     ///
//     /// This function should return the same result at all times regardless of
//     /// external mutable state. For example, values stored in TLS or in
//     /// `static mut`s.
//     // This cannot be an associated constant, because of bignums.
//     #[inline]
//     fn zero() -> Self {
//         Self::ZERO
//     }
//
//     /// Sets `self` to the additive identity element of `Self`, `0`.
//     #[inline]
//     fn set_zero(&mut self) {
//         *self = Zero::zero();
//     }
//
//     /// Returns `true` if `self` is equal to the additive identity.
//     fn is_zero(&self) -> bool;
// }
//
// macro_rules! zero_impl {
//     ($($t:ty, $v:expr),*) => {
//         $(
//             impl Zero for $t {
//                 const ZERO: Self = $v;
//                 #[inline]
//                 fn is_zero(&self) -> bool { *self == Self::ZERO }
//             }
//         )*
//     };
// }
//
// zero_impl!(usize, 0);
// zero_impl!(u8, 0);
// zero_impl!(u16, 0);
// zero_impl!(u32, 0);
// zero_impl!(u64, 0);
// zero_impl!(u128, 0);
//
// zero_impl!(isize, 0);
// zero_impl!(i8, 0);
// zero_impl!(i16, 0);
// zero_impl!(i32, 0);
// zero_impl!(i64, 0);
// zero_impl!(i128, 0);
//
// zero_impl!(f32, 0.0);
// zero_impl!(f64, 0.0);
//
// /// Defines a multiplicative identity element for `Self`.
// ///
// /// # Laws
// ///
// /// ```text
// /// a * 1 = a       ∀ a ∈ Self
// /// 1 * a = a       ∀ a ∈ Self
// /// ```
// pub trait One: Sized + Mul<Self, Output = Self> {
//     /// The multiplicative identity element of `Self`, `1`.
//     const ONE: Self;
//
//     // Modifying this is a safety feature, prevents use of BigNums
//     // (which can use heap/allocations)
//     /// Returns the multiplicative identity element of `Self`, `1`.
//     ///
//     /// # Purity
//     ///
//     /// This function should return the same result at all times regardless of
//     /// external mutable state. For example, values stored in TLS or in
//     /// `static mut`s.
//     // This cannot be an associated constant, because of bignums.
//     fn one() -> Self {
//         Self::ONE
//     }
//
//     /// Returns `true` if `self` is equal to the multiplicative identity.
//     ///
//     /// For performance reasons, it's best to implement this manually.
//     /// After a semver bump, this method will be required, and the
//     /// `where Self: PartialEq` bound will be removed.
//     fn is_one(&self) -> bool;
// }
//
// macro_rules! one_impl {
//     ($($t:ty, $v:expr),*) => {
//         $(
//             impl One for $t {
//                 const ONE: Self = $v;
//                 #[inline]
//                 fn is_one(&self) -> bool { *self == Self::ONE }
//             }
//         )*
//     };
// }
//
// one_impl!(usize, 1);
// one_impl!(u8, 1);
// one_impl!(u16, 1);
// one_impl!(u32, 1);
// one_impl!(u64, 1);
// one_impl!(u128, 1);
//
// one_impl!(isize, 1);
// one_impl!(i8, 1);
// one_impl!(i16, 1);
// one_impl!(i32, 1);
// one_impl!(i64, 1);
// one_impl!(i128, 1);
//
// one_impl!(f32, 1.0);
// one_impl!(f64, 1.0);
//
// pub trait Arithmatic<Rhs = Self, Output = Self>:
//     Add<Rhs, Output = Output>
//     + Sub<Rhs, Output = Output>
//     + Mul<Rhs, Output = Output>
//     + Div<Rhs, Output = Output>
//     + Rem<Rhs, Output = Output>
// {
// }
//
// impl<T> Arithmatic<T> for T where
//     T: Add<T, Output = T>
//         + Sub<T, Output = T>
//         + Mul<T, Output = T>
//         + Div<T, Output = T>
//         + Rem<T, Output = T>
// {
// }
//
// pub trait Number: PartialEq + Zero + One + Arithmatic {}
//
// impl<T> Number for T where T: PartialEq + Zero + One + Arithmatic {}
