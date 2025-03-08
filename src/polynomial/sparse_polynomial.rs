// //! A polynomial alias that does not have a correlation between
// //! coefficient degree and index.
// use super::*;

// use num_traits::Pow;

// /// An object that mocks a larger (R,C) shaped storage but only stores N.
// #[derive(Clone, Debug, PartialEq)]
// pub struct SparseArrayStorage<T, R: Dim, C: Dim, const N: usize> 
// // where R > D & C = N
// {
//     pub shape: (R, C),
//     pub data: [T; N],
//     pub mask: [usize; N],
// }

// /// A polynomial alias that uses a degree mask rather than degree/index/length relationship.
// pub type SparsePolynomial<T, R: Dim, C: Dim, const N: usize> = Polynomial<T, R, SparseArrayStorage<T, R, C, N>, Const<N>>;
// // pub type SparsePolynomial<T, D> = Polynomial<T, D, U2, S<D, U2>>;

// impl<T, R: Dim, C: Dim, const N: usize> SparsePolynomial<T, R, C, N> {
//     /// Creates a [SPolynomial] that has both coefficients and a degree mask.
//     /// 
//     /// The SparsePolynomial is not restricted to storing a degree + 1 number of coefficients.
//     /// i.e. a 20th, 100th and 1000th order function may be represented with a single coeff.
//     /// 
//     /// # Arguments
//     /// 
//     /// * `variable` - variable of the polynomial
//     /// * `coefficients` - the non-zero coefficients
//     /// * `degree_mask` - matches the size of coefficients and provides the degree for each coeff
//     /// 
//     /// # Returns
//     /// 
//     /// * `SparsePolynomial` - alias type for a Polynomial<T, D, ArrayStorage<T, D, 2>, 2>
//     pub const fn sparse(variable: &'static str, coefficients: [T; D], mask: [usize; D]) -> Self {
//         Self {
//             variable,
//             coefficients: SparseArrayStorage {
//                 shape: (R, C),
//                 data: [coefficients],
//                 mask: [mask], 
//             },
//             _phantom: PhantomData,
//         }
//     }
// }


// impl<T, const D: usize> SparsePolynomial<T, D>
// where
//     T: Copy,
// {
//     /// Evaluate the polynomial at the given value.
//     ///
//     /// # Arguments
//     ///
//     /// * `value` - The value at which to evaluate the polynomial.
//     ///
//     /// # Returns
//     ///
//     /// * `T` - The value of the polynomial at the given value.
//     ///
//     /// # Example
//     ///
//     /// ```rust
//     /// use control_rs::polynomial::Polynomial;
//     ///
//     /// let p = Polynomial::sparse("x", [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]);
//     /// let value = p.evaluate(2.0);
//     /// ```
//     pub fn evaluate<U>(&self, value: U) -> U
//     where
//         U: Copy + Zero + One + Add<U, Output = U> + Mul<T, Output = U> + Pow<T, Output = U>,
//     {
//         (0..self.num_coefficients()).fold(U::zero(), |acc, irow| {
//             acc + value.pow(self.coefficients.0[1][irow]) * self[irow]
//         })
//     }
// }

// #[cfg(test)]
// mod sparse_polynomial_tests {
//     use super::*;
//     #[test]
//     fn basic_init() {
//         let polynomial = Polynomial::sparse("x", [1], [10]);
//         assert_eq!(polynomial.coefficients, ArrayStorage([[1], [10]]));
//         assert_eq!(polynomial.variable, "x");
//     }

//     #[test]
//     fn coeff_as_slice() {
//         let polynomial = Polynomial::sparse("z", [1], [10]);
//         assert_eq!(polynomial.coefficients(), [1, 10]);
//         assert_eq!(polynomial.variable, "z");
//     }   

//     #[test]
//     fn evaluate_unit_quartic() {
//         let polynomial: SparsePolynomial<u32, 1> = Polynomial::sparse("y", [1], [4]);
//         // assert_eq!(polynomial.evaluate(0i16), 0);
//         assert_eq!(polynomial.evaluate(0u32), 0);
//         assert_eq!(polynomial.evaluate(1u32), 1);
//         assert_eq!(polynomial.evaluate(2u32), 16);
//         assert_eq!(polynomial.evaluate(4u32), 256);
//         assert_eq!(polynomial.variable, "y");
//     } 

//     #[test]
//     fn evaluate_constant_one() {
//         let polynomial: SparsePolynomial<u32, 1> = Polynomial::sparse("y", [1], [0]);
//         // assert_eq!(polynomial.evaluate(0i16), 0);
//         assert_eq!(polynomial.evaluate(0u32), 1);
//         assert_eq!(polynomial.evaluate(1u32), 1);
//         assert_eq!(polynomial.evaluate(2u32), 1);
//         assert_eq!(polynomial.evaluate(4u32), 1);
//         assert_eq!(polynomial.variable, "y");
//     } 
// }