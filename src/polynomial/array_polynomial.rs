//! Statically sized Polynomial

use super::*;

/// A static polynomial with D coefficients. *Similar to [nalgebra::SMatrix], this is a type alias
/// so not all of its methods are listed here. See [Polynomial] too*
pub type SPolynomial<T, const D: usize, const N: usize = 1> =
    Polynomial<T, Const<D>, ArrayStorage<T, D, N>, Const<N>>;

impl<T, const D: usize> SPolynomial<T, D> {
    /// Create an [ArrayStorage] based [Polynomial]
    ///
    /// This function uses a static array to initialize the coefficients of a polynomial. It is
    /// assumed the coefficients are sorted from highest to lowest degree, so the largest index
    /// refers to the constant term.
    ///
    /// # Arguments
    ///
    /// * `variable` - the variable of the polynomial
    /// * `coefficients` - an array of the coefficients, length = degree + 1
    ///
    /// # Returns
    ///
    /// * `Polynomial` - static polynomial
    pub const fn new(coefficients: [T; D]) -> Self {
        Self {
            coefficients: ArrayStorage([coefficients]),
            _phantom: PhantomData,
        }
    }
}

impl<T, const D: usize, const N: usize> SPolynomial<T, D, N> {
    /// wrapper for [ArrayStorage::as_slice()]
    ///
    /// # Returns
    ///
    /// * `coeffs` - slice of coefficients sorted from high degree -> low degree
    pub fn coefficients(&self) -> &[T] {
        self.coefficients.as_slice()
    }
}

impl<T, const D: usize, const N: usize> SPolynomial<T, D, N>
where
    T: Scalar + One + Zero + RealField,
{
    /// Fits a polynomial to the given data points using the least squares method.
    ///
    /// # Arguments
    ///
    /// * `x` - An array of input values of shape `[T; N]`, where `N` is the number of data points.
    /// * `y` - An array of corresponding output values of shape `[T; N]`.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D, S>` - that approximates the relationship between `x` and `y`
    ///
    /// # Generic Arguments
    ///
    /// * `N` - The number of data points in the X,y set
    ///
    /// # Example
    /// ```rust
    /// use control_rs::polynomial::SPolynomial;
    /// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    /// let y = [4.0, 1.0, 0.0, 1.0, 4.0];
    /// let poly: SPolynomial<f64, 3> = SPolynomial::fit(x, y);
    /// ```
    pub fn fit<const L: usize>(
        x: [T; L],
        y: [T; L],
    ) -> Polynomial<T, Const<D>, ArrayStorage<T, D, 1>>
    where
        Const<L>: DimMin<Const<D>>,
        DimMinimum<Const<L>, Const<D>>: DimSub<U1>,
        DefaultAllocator: Allocator<DimMinimum<Const<L>, Const<D>>, Const<D>>
            + Allocator<Const<L>, DimMinimum<Const<L>, Const<D>>>
            + Allocator<DimMinimum<Const<L>, Const<D>>>
            + Allocator<DimDiff<DimMinimum<Const<L>, Const<D>>, U1>>,
    {
        let degree = D - 1;
        let h = OMatrix::<T, Const<L>, U1>::from_row_slice(&y);
        let vandermonde = OMatrix::<T, Const<L>, Const<D>>::from_fn(|i, j| {
            (0..degree - j).fold(T::one(), |acc, _| acc * x[i].clone())
        });
        let coeff_estimate = vandermonde
            .svd(true, true)
            .solve(&h, T::RealField::from_f64(1e-15).unwrap())
            .expect("Least squares solution failed");
        Polynomial::from_matrix(coeff_estimate)
    }
}

#[cfg(test)]
mod spolynomial_tests {

    use super::*;

    #[test]
    fn basic_init() {
        let polynomial = SPolynomial::new([1, 0, 0]);
        assert_eq!(polynomial.coefficients, ArrayStorage([[1, 0, 0]]));
    }

    #[test]
    fn coeff_as_slice() {
        let polynomial = SPolynomial::new([1, 0, 0]);
        assert_eq!(polynomial.coefficients(), [1, 0, 0]);
    }
}
