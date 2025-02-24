//!
//! # Polynomial struct and default implementations
//!
//!
#[cfg(feature = "std")]
use std::{
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Neg},
};

#[cfg(not(feature = "std"))]
use core::{
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Neg},
};

use nalgebra::{ArrayStorage, Complex, Const, DefaultAllocator, Dim, DimName, DimNameDiff, DimNameSub, Dyn, Matrix, RawStorage, RawStorageMut, SMatrix, VecStorage, U1};
use num_traits::{Float, Num, Zero};

pub mod tools;
pub use tools::*;

pub struct Polynomial<T, D, S> {
    pub coefficients: S,
    _phantom: PhantomData<(T, D)>,
}

impl<T, D: Dim, const N: usize> Polynomial<T, D, ArrayStorage<T, N, 1>> {
    pub const fn new(coefficients: [T; N]) -> Self {
        Self {
            coefficients: ArrayStorage([coefficients]),
            _phantom: PhantomData,
        }
    }

    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}

impl<T> Polynomial<T, Dyn, VecStorage<T, Dyn, Const<1>>> {
    pub fn from_vec(coefficients: Vec<T>) -> Self {
        Self {
            // VecStorage checks that vec.len() == rows * cols
            coefficients: VecStorage::new(Dyn(coefficients.len()), U1, coefficients),
            _phantom: PhantomData,
        }
    }

    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: Copy + Num,
    D: Dim,
    S: RawStorage<T, D>,
{
    pub fn constant(&self) -> T {
        self.coefficient(self.num_coefficients() - 1)
    }

    pub fn num_coefficients(&self) -> usize {
        self.coefficients.shape().0.value()
    }

    pub fn coefficient(&self, i: usize) -> T {
        assert!(i < self.num_coefficients(), "Index out of bounds");
        unsafe { *self.coefficients.get_unchecked(i, 0) }        
    }

    pub fn evaluate<U>(&self, input: U) -> U
    where
        U: Copy + Zero + Add<T, Output = U> + Mul<U, Output = U>,
    {
        let nrows = self.num_coefficients();
        (0..nrows).fold(U::zero(), |acc, irow| {
            // safe becuase nrows is from shape() and irow is in 0..nrows
            acc * input + self.coefficient(irow)
        })
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: Copy + Num,
    D: Dim,
    S: RawStorageMut<T, D>,
{
    pub fn set_coefficient(&mut self, i: usize, value: T) {
        assert!(i < self.num_coefficients(), "Index out of bounds");
        unsafe { *self.coefficients.get_unchecked_mut(i, 0) = value}        
    }
}

// impl<T, D, S> Polynomial<T, D, S>
// where
//     T: 'static + Copy + Num + Neg<Output = T> + fmt::Debug,
//     D: DimNameSub<U1>,
//     S: RawStorage<T, D>,
//     DefaultAllocator: nalgebra::allocator::Allocator<DimNameDiff<D, U1>, DimNameDiff<D, U1>>
// {
//     pub fn companion(&self) -> Matrix<T, _, _, _> {
//         Matrix::<T, DimNameDiff<D, U1>, DimNameDiff<D, U1>, _>::from_fn(|i, j| {
//             if i == 0 {
//                 -self.coefficient(j + 1)
//             } else {
//                 if i + 1 == j {
//                     T::one()
//                 } else {
//                     T::zero()
//                 }
//             }
//         })
//     }
// }

pub fn companion<T, D1, D2, S1, S2>(polynomial: Polynomial<T, D1, S1>) -> Matrix<T, D2, D2, S2> 
where 
    T: 'static + Copy + Num + Neg<Output = T> + fmt::Debug,
    D1: DimNameSub<U1>,
    DimNameDiff<D1, U1>: DimName,
    D2: DimName,
    S1: RawStorage<T, D1>,
    S2: RawStorage<T, D2, D2>,
{
    SMatrix::<T, D2, D2>::from_fn(|i, j| {
        if i == 0 {
            -polynomial.coefficient(j + 1)
        } else {
            if i + 1 == j {
                T::one()
            } else {
                T::zero()
            }
        }
    })
}

// impl<T, D, S> Polynomial<T, D, S>
// where
//     T: 'static + Copy + Float + fmt::Debug,
//     D: DimNameSub<U1>,
//     DimNameDiff<D, U1>: DimName,
//     S: RawStorage<T, D>,
//     DefaultAllocator: nalgebra::allocator::Allocator<DimNameDiff<D, U1>, U1>
// {
//     pub fn roots(&self) -> Matrix<Complex<T>, DimNameDiff<D, U1>, U1, _> {
//         // let mut roots = Matrix::zeros_generic(DimNameDiff::<D, U1>::name(), U1);
//         // tools::roots(self.coefficients(), roots.data.as_mut_slice());
//         // roots

//         // less than 1 coeff has no roots, and len = 1, is nan unless coeff[0] = 0 then root[0] = inf
//         match D::dim() {
//             0 => {
//                 return Matrix::<Complex<T>, DimNameDiff<D, U1>, U1, _>::from_element(Complex::new(T::nan(), T::nan()));
//             }
//             1 => {
//                 if self.coefficient(0).is_zero() {
//                     return Matrix::from_element(Complex::new(T::infinity(), T::nan()));
//                 } else {
//                     return Matrix::from_element(Complex::new(T::nan(), T::nan()));
//                 }
//             }
//             _ => {}
//         }
//         // for efficiency? avoid recursing in the base case: all coeff == 0
//         let num_zero_coeff = (0..self.num_coefficients()).fold(0, |acc, &i| match self.coefficient(i) == T::zero() {
//             true => acc + 1,
//             false => acc,
//         });

//         // all coeff are zero
//         if self.num_coefficients() == num_zero_coeff {
//             // constant 0, everything is a root
//             return Matrix::from_element(Complex::new(T::infinity(), T::nan()));
//         }

//         if self.coefficient(0).is_zero() {
//             let mut root_buffer = Matrix::from_element(Complex::new(T::nan(), T::nan()));
//             let reduced_polynomial = Polynomial { 
//                 coefficients: self.coefficients.fixed_rows::<DimNameDiff<D, U1>>(1),
//                 _phantom: PhantomData,
//             };
//             reduced_polynomial.roots().iter().enumerate().for_each(|(i, root)| {
//                 root_buffer[i] = root;
//             });
//             return root_buffer;
//         } else {
//             // Compute the eigenvalues of the companion matrix
//             match self.companion().eigenvalues() {
//                 Some(eigenvalues) => Matrix::from_fn(|i, j| eigenvalues[i]),
                
//                 None => Matrix::from_element(Complex::new(T::nan(), T::nan())),
//             }
//         }
//     }
// }

impl<T, D: Dim, S: RawStorage<T, D>> fmt::Display for Polynomial<T, D, S>
where
    T: fmt::Display + From<u8> + Copy + Num + PartialOrd + Neg<Output = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ncoeff = self.coefficients.shape().0.value();

        for i in 0..ncoeff {
            let coeff = self.coefficient(i);

            if coeff == T::from(0) {
                continue;
            }

            if i > 0 {
                write!(f, " {} ", if coeff >= T::from(0) { "+" } else { "-" })?;
            } else if coeff < T::from(0) {
                write!(f, "-")?;
            }

            let abs_coeff = if coeff < T::from(0) { -coeff } else { coeff };
            let exp = ncoeff - 1 - i;

            if abs_coeff != T::from(1) || exp == 0 {
                write!(f, "{}", abs_coeff)?;
            }

            if exp > 0 {
                write!(f, "x")?;
                if exp > 1 {
                    write!(f, "^{}", exp)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
#[cfg(feature = "std")]
mod fmt_tests;
