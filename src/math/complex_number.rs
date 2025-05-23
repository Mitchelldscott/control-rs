//! Scalar number types

#[cfg(feature = "std")]
use std::ops::{Add, Mul};

#[cfg(not(feature = "std"))]
use core::ops::{Add, Mul};

use super::num_traits::{One, Zero, Number, Arithmatic};

/// A complex number in Cartesian form.
///
/// ## Representation and Foreign Function Interface Compatibility
///
/// `Complex<T>` is memory layout compatible with an array `[T; 2]`.
///
/// Note that `Complex<F>` where F is a floating point type is **only** memory
/// layout compatible with C's complex types, **not** necessarily calling
/// convention compatible.  This means that for FFI you can only pass
/// `Complex<F>` behind a pointer, not as a value.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug, Default)]
#[repr(C)]
pub struct Complex<T> {
    /// Real portion of the complex number
    real: T,
    /// Imaginary portion of the complex number
    imaginary: T,
}

impl<T> Complex<T> {
    /// Create a new complex number
    #[inline]
    pub fn new(real: T, imaginary: T) -> Self {
        Self { real, imaginary }
    }
}

impl<T: Number> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

impl<T: Clone + Number> Mul<Complex<T>> for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex {
            real: self.real.clone() * rhs.real.clone() - self.imaginary.clone() * rhs.imaginary.clone(),
            imaginary: self.real * rhs.imaginary + self.imaginary * rhs.real,
        }
    }
}

impl<T: Number> Zero for Complex<T> {
    const ZERO: Self = Complex { real: T::ZERO, imaginary: T::ZERO };
    #[inline]
    fn is_zero(&self) -> bool { *self == Self::ZERO }
}

impl<T: Clone + Number> One for Complex<T> {
    const ONE: Self = Complex { real: T::ONE, imaginary: T::ONE };
    #[inline]
    fn is_one(&self) -> bool { *self == Self::ONE }
}

// impl<T: Clone + Number + PartialEq + Arithmatic> Number for Complex<T> {}