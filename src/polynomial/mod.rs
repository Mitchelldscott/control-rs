//!
//! # Polynomial struct and helpful tools
//!
//! Mostly a copy of [Matrix] implementation with the specifics of a polynomial.
use nalgebra::{
    allocator::Allocator, ArrayStorage, Complex, Const, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Matrix, OMatrix, RawStorage, RawStorageMut, RealField, Scalar, U1
};
use num_traits::{Float, Num, One, Zero};

#[cfg(feature = "std")]
use std::{
    fmt,
    iter,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign},
};

#[cfg(not(feature = "std"))]
use core::{
    fmt,
    iter,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign},
};

// ===============================================================================================
//      Polynomial Sub-Modules
// ===============================================================================================

pub mod array_polynomial;
pub use array_polynomial::SPolynomial;

#[cfg(feature = "std")]
pub mod vec_polynomial;
#[cfg(feature = "std")]
pub use vec_polynomial::DPolynomial;

// pub mod sparse_polynomial;

// ===============================================================================================
//      Polynomial Tests
// ===============================================================================================

#[cfg(test)]
mod edge_case_test;

#[cfg(test)]
mod op_tests;

#[cfg(test)]
#[cfg(feature = "std")]
mod fmt_tests;

// ===============================================================================================
//      Polynomial Base Implementation
// ===============================================================================================

/// Stores coefficients and implements tools for a polynomial degree `D - 1`.
///
/// This struct stores the coefficients of a polynomial a(s):
/// <pre>
/// a(s) = a_0 * s^(D-1) + a_1 * s^(D-2) + ... + a_(D-2) * s + a_(D-1)
/// </pre>
///
/// The coefficients are stored in descending degree order.
///
/// # Generic Arguments
///
/// * `T` - Type of the coefficients
/// * `D` - Length of the coefficient array, ***NOT the degree!***
/// * `S` - The type of the storage
/// * `N` - number of equations (input variables)
///
/// # Example
/// ```rust
/// use control_rs::Polynomial;
/// let quadratic = Polynomial::new([1, 0, 0]);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Polynomial<T, D, S, N = U1> {
    /// coefficients of the polynomial `[a0, a1, ... an]`, stored [highest -> lowest degree]
    pub coefficients: S,
    _phantom: PhantomData<(T, D, N)>,
}

impl<T, D, S, N> Polynomial<T, D, S, N>
where
    D: Dim,
    N: Dim,
    S: RawStorage<T, D, N>,
{
    /// Returns the number of coefficients in the polynomial (degree + 1).
    ///
    /// This function relies on the [RawStorage] trait to access the the number of rows.
    ///
    /// # Returns
    ///
    /// * `num_coeff` - number of coefficients (including zeros)
    pub fn num_coefficients(&self) -> usize {
        self.coefficients.shape().0.value()
    }

    /// Returns the number of equations in the polynomial.
    ///
    /// This function relies on the [RawStorage] trait to access the the number of cols.
    ///
    /// # Returns
    ///
    /// * `num_eqn` - number of equations
    pub fn num_equations(&self) -> usize {
        self.coefficients.shape().1.value()
    }

    /// Construct a polynomial from a [Matrix].
    ///
    /// should this be from_data(data: S) instead?
    ///
    /// # Arguments
    ///
    /// * `matrix` - column matrix containing the coefficients
    ///
    /// # Returns
    ///
    /// * `polynomial` - polynomial built from the matrix's data
    pub fn from_matrix(matrix: Matrix<T, D, N, S>) -> Self {
        Polynomial {
            coefficients: matrix.data,
            _phantom: PhantomData,
        }
    }

    /// Construct a polynomial from data.
    ///
    /// # Arguments
    ///
    /// * `data` - RawStorage implementor
    ///
    /// # Returns
    ///
    /// * `polynomial` - polynomial built from the given data
    pub fn from_data(data: S) -> Self {
        Polynomial {
            coefficients: data,
            _phantom: PhantomData,
        }
    }
}

impl<T, D, S, N> Polynomial<T, D, S, N>
where
    T: Scalar,
    D: Dim,
    N: Dim,
    S: RawStorage<T, D, N>,
    DefaultAllocator: Allocator<D, N, Buffer<T> = S>,
{
    /// Creates a polynomial with all its elements set to `elem`.
    #[inline]
    pub fn from_element_generic(nrows: D, ncols: N, elem: T) -> Self {
        let len = nrows.value() * ncols.value();
        Self::from_iterator_generic(nrows, ncols, iter::repeat(elem).take(len))
    }

    /// Creates a polynomial with all its elements set to 0.
    #[inline]
    pub fn zeros_generic(nrows: D, ncols: N) -> Self
    where
        T: Zero,
    {
        Self::from_element_generic(nrows, ncols, T::zero())
    }

    /// Creates a polynomial with all its elements filled by an iterator.
    #[inline]
    pub fn from_iterator_generic<I>(nrows: D, ncols: N, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_data(DefaultAllocator::allocate_from_iterator(nrows, ncols, iter))
    }
}

impl<T, D, S, N> Polynomial<T, D, S, N>
where
    T: Clone,
    D: Dim,
    N: Dim,
    S: RawStorage<T, D, N>,
{
    /// Returns a copy of the coefficient at the given index.
    ///
    /// This is a wrapper for the index trait impl, the Copy trait bound allows this
    /// function to copy the value at index rather than returning a refernce to the value.
    ///
    /// Coefficients are sorted high degree -> low degree, should add a function to look up terms
    /// by degree.
    ///
    /// This will panic if the index is invalid.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of the coefficient
    ///
    /// # Returns
    ///
    /// * `coeff` - the coefficient at the index
    ///
    /// # Example
    /// ```rust
    /// use control_rs::Polynomial;
    /// let p = Polynomial::new("x", [1, 0, 0]);
    /// // forces T to be Copy, won't compile otherwise
    /// assert_eq!(p.coefficient(0), 1, "reference had the wrong value");
    /// ```
    pub fn coefficient(&self, index: usize) -> T {
        self[index].clone()
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: Clone,
    D: Dim + DimSub<U1>,
    S: RawStorage<T, D>,
{
    /// Returns a copy of the constant coefficient.
    ///
    /// This is a wrapper for the index function, the Copy trait bound allows this
    /// function to copy the value at index rather than returning a refernce to the value.
    ///
    /// The constant term of the polynomial is the last term in the storage. This is just
    /// a convienience function.
    ///
    /// # Returns
    ///
    /// * `coeff` - the constant coefficient
    pub fn constant(&self) -> T {
        self[self.num_coefficients() - 1].clone()
    }
}

impl<T, D, S> Default for Polynomial<T, D, S>
where
    T: Clone,
    S: Default,
{
    /// Create a new polynomial with the default coefficients.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - An array of coefficients of the polynomial in descending degree order.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D>` - A new polynomial with the given coefficients.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    /// use nalgebra::{U2, ArrayStorage};
    /// let p: Polynomial<i32, U2, ArrayStorage<i32, 2, 1>> = Polynomial::default();
    /// ```
    fn default() -> Self {
        Polynomial {
            coefficients: S::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: Clone,
    D: Dim,
    S: RawStorage<T, D>,
{
    /// Evaluate the polynomial at the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value at which to evaluate the polynomial.
    ///
    /// # Returns
    ///
    /// * `T` - The value of the polynomial at the given value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, 2.0, 3.0]);
    /// let value = p.evaluate(2.0);
    /// ```
    pub fn evaluate<U>(&self, value: U) -> U
    where
        U: Copy + Zero + One + Add<T, Output = U> + Mul<U, Output = U>,
    {
        (0..self.num_coefficients()).fold(U::zero(), |acc, irow| {
            // safe becuase nrows is from shape() and irow is in 0..nrows
            acc * value + self[irow].clone()
        })
    }
}

impl<T, D, S1> Polynomial<T, D, S1>
where
    T: Clone + Num + Default,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName,
    S1: RawStorage<T, D>,
{
    /// Compute the derivative of the polynomial.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D - 1>` - The derivative of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    /// use nalgebra::ArrayStorage;
    ///
    /// let p = Polynomial::new([1.0, 2.0, 3.0]);
    /// let derivative = p.derivative::<ArrayStorage<f32, 2, 1>>();
    /// ```
    pub fn derivative<S2>(&self) -> Polynomial<T, DimDiff<D, U1>, S2>
    where
        S2: RawStorageMut<T, DimDiff<D, U1>> + Default,
    {
        let new_dim = self.num_coefficients() - 1;
        let mut derivative = Polynomial {
            coefficients: S2::default(),
            _phantom: PhantomData,
        };
        for i in 0..new_dim {
            derivative[i] = (i..new_dim).fold(T::zero(), |acc, _| acc + self[i].clone());
        }

        derivative
    }

    /// Compute the derivative of the polynomial.
    ///
    /// # Returns
    ///
    /// * `Polynomial<T, D - 1>` - The derivative of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    /// use nalgebra::{U2, ArrayStorage};
    ///
    /// let p = Polynomial::new([1.0, 2.0, 3.0]);
    /// let reduced_p: Polynomial<f32, U2, ArrayStorage<f32, 2, 1>> = p.reduce_order();
    /// ```
    pub fn reduce_order<S2>(&self) -> Polynomial<T, DimDiff<D, U1>, S2>
    where
        S2: RawStorageMut<T, DimDiff<D, U1>> + Default,
    {
        let mut reduced = Polynomial {
            coefficients: S2::default(),
            _phantom: PhantomData,
        };
        for i in 0..reduced.num_coefficients() {
            reduced[i] = self[i + 1].clone();
        }

        reduced
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: 'static + Clone + Num + Neg<Output = T> + fmt::Debug,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName,
    S: RawStorage<T, D>,
    DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>>,
{
    /// Constructs the companion matrix of the polynomial.
    ///
    /// The companion matrix is a square matrix whose eigenvalues are the roots of the polynomial.
    /// It is constructed from an identity matrix, a zero column and a row of the polynomial's
    /// coefficients scaled by the highest term.
    ///
    /// <pre>
    /// |  a[1..D] / a[0]  |
    /// |     I      0     |
    /// </pre>
    /// <pre>
    /// |  -a_1/a_0  -a_2/a_0  -a_3/a_0  ...  -a_(D-2)/a_0  -a_(D-1)/a_0  |
    /// |     1         0         0      ...       0            0         |
    /// |     0         1         0      ...       0            0         |
    /// |     0         0         1      ...       0            0         |
    /// |    ...       ...       ...     ...      ...          ...        |
    /// |     0         0         0      ...       1            0         |
    /// </pre>
    ///
    /// # Returns
    ///
    /// * `OMatrix<T, DimDiff<D, U1>, DimDiff<D, U1>>` - The companion matrix representation of the polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]);
    /// let companion_matrix = p.companion();
    /// ```
    pub fn companion(&self) -> OMatrix<T, DimDiff<D, U1>, DimDiff<D, U1>> {
        // return companion;
        OMatrix::<T, DimDiff<D, U1>, DimDiff<D, U1>>::from_fn(|i, j| {
            if i == 0 {
                -self[j + 1].clone() / self[0].clone()
            } else {
                if i == j + 1 {
                    T::one()
                } else {
                    T::zero()
                }
            }
        })
    }
}

impl<T, D, S> Polynomial<T, D, S>
where
    T: 'static + Clone + Num + Neg<Output = T> + fmt::Debug + RealField + Float,
    D: DimSub<U1>,
    DimDiff<D, U1>: DimName + DimSub<U1>,
    S: RawStorage<T, D>,
    DefaultAllocator: Allocator<DimDiff<D, U1>, DimDiff<D, U1>>
        + Allocator<DimDiff<D, U1>, DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<DimDiff<D, U1>, U1>>
        + Allocator<DimDiff<D, U1>>,
{
    /// Computes the roots of the polynomial.
    ///
    /// Edge cases:
    /// - if the leadng coeff is zero this should reduce the order and recurse, having issues with trait bounds (currently returning NaN)
    /// - all coefficients are zero: all roots are infinite
    /// - if there are two coefficients and the lead is non-zero: the root is `-coeff[1]/coeff[0]`
    /// - if all but the first coeff are zero: all roots are zero
    ///
    ///
    /// For very high order polynomials this may be inefficient, especially for degenerate cases.
    /// User should consider cases where all/many coeff = 0 and avoid calling this. Would be nice if
    /// nalgebra handled large/sparse matrix eigenvalues.
    ///
    /// # Returns
    ///
    /// * `OMatrix<Complex<T>, Const<D>, U1>` - A column vector containing the computed roots.
    ///
    /// # Example
    ///
    /// ```rust
    /// use control_rs::polynomial::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -6.0, 11.0, -6.0]);
    /// let roots = p.roots();
    /// ```
    pub fn roots(&self) -> OMatrix<Complex<T>, DimDiff<D, U1>, U1> {
        if !self[0].is_zero() {
            let num_coeff = self.num_coefficients();
            if num_coeff == 2 {
                OMatrix::<Complex<T>, DimDiff<D, U1>, U1>::from_element(Complex::new(
                    -self[1].clone() / self[0].clone(),
                    T::zero(),
                ))
            } else {
                let num_zeros = (0..num_coeff).fold(0, |acc, i| match self[i] == T::zero() {
                    true => acc + 1,
                    false => acc,
                });

                if num_zeros == num_coeff {
                    // zero/degenerate polynomial, all infinite roots
                    OMatrix::<Complex<T>, DimDiff<D, U1>, U1>::from_element(Complex::new(
                        T::infinity(),
                        T::infinity(),
                    ))
                } else if num_zeros == num_coeff - 1 {
                    // unit case, all zero roots
                    OMatrix::<Complex<T>, DimDiff<D, U1>, U1>::from_element(Complex::new(
                        T::zero(),
                        T::zero(),
                    ))
                } else {
                    // need to know more specifics about what matrices work with complex_eigenvalues,
                    // the current cases fixed an infinite loops in the test, but certianly not a garuanteed solution
                    self.companion().complex_eigenvalues()
                }
            }
        } else {
            // should be able to reduce order and keep trying, but having issues with recursive trait bounds
            OMatrix::<Complex<T>, DimDiff<D, U1>, U1>::from_element(Complex::new(
                T::nan(),
                T::nan(),
            ))
        }
    }
}

// ===============================================================================================
//      Polynomial Operator Implementations
// ===============================================================================================

//
//  Index
//

impl<T, D, S, N> Index<usize> for Polynomial<T, D, S, N>
where
    D: Dim,
    N: Dim,
    S: RawStorage<T, D, N>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        // also need to check if storage is contiguous for sparse polynomial?
        assert!(index < self.num_coefficients(), "Index out of bounds");
        unsafe { self.coefficients.get_unchecked(index, 0) }
    }
}

impl<T, D, S, N> IndexMut<usize> for Polynomial<T, D, S, N>
where
    D: Dim,
    N: Dim,
    S: RawStorageMut<T, D, N>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.num_coefficients(), "Index out of bounds");
        unsafe { self.coefficients.get_unchecked_mut(index, 0) }
    }
}

impl<T, D, S, N> Index<(usize, usize)> for Polynomial<T, D, S, N>
where
    D: Dim,
    N: Dim,
    S: RawStorage<T, D, N>,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        // also need to check if storage is contiguous for sparse polynomial?
        assert!(index.0 < self.num_coefficients(), "Index row out of bounds");
        assert!(index.1 < self.coefficients.shape().1.value(), "Index col out of bounds");
        unsafe { self.coefficients.get_unchecked(index.0, index.1) }
    }
}

impl<T, D, S, N> IndexMut<(usize, usize)> for Polynomial<T, D, S, N>
where
    D: Dim,
    N: Dim,
    S: RawStorageMut<T, D, N>,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.num_coefficients(), "Index row out of bounds");
        assert!(index.1 < self.coefficients.shape().1.value(), "Index col out of bounds");
        unsafe { self.coefficients.get_unchecked_mut(index.0, index.1) }
    }
}

//
//  Scalar Math
//
// provides function implementions for op(Polynomial, U), where U implements Num
//

impl<T, U, D, S> Add<U> for Polynomial<T, D, S>
where
    T: Clone + Add<U, Output = T>,
    U: Clone + Num,
    D: Dim + DimSub<U1>,
    S: RawStorageMut<T, D> + Clone,
{
    type Output = Self;

    fn add(self, rhs: U) -> Self::Output {
        let mut new_poly = self.clone();
        let num_coeff = self.num_coefficients();
        if num_coeff > 0 {
            new_poly[num_coeff - 1] = self.constant() + rhs;
        }
        new_poly
    }
}

impl<T, U, D, S> Sub<U> for Polynomial<T, D, S>
where
    T: Clone + Sub<U, Output = T>,
    U: Clone + Num,
    D: Dim + DimSub<U1>,
    S: RawStorageMut<T, D> + Clone,
{
    type Output = Self;

    fn sub(self, rhs: U) -> Self::Output {
        let mut new_poly = self.clone();
        let num_coeff = self.num_coefficients();
        if num_coeff > 0 {
            new_poly[num_coeff - 1] = self.constant() - rhs;
        }
        new_poly
    }
}

impl<T, U, D, S> Mul<U> for Polynomial<T, D, S>
where
    T: Clone + Mul<U, Output = T>,
    U: Clone + Num,
    D: Dim,
    S: RawStorageMut<T, D> + Clone,
{
    type Output = Polynomial<T, D, S>;

    fn mul(self, rhs: U) -> Self::Output {
        let mut new_poly = self.clone();
        for i in 0..self.num_coefficients() {
            new_poly[i] = self[i].clone() * rhs.clone();
        }
        new_poly
    }
}

impl<T, U, D, S> Div<U> for Polynomial<T, D, S>
where
    T: Clone + Div<U, Output = T>,
    U: Clone + Num,
    D: Dim,
    S: RawStorageMut<T, D> + Clone,
{
    type Output = Self;

    fn div(self, rhs: U) -> Self::Output {
        let mut new_poly = self.clone();
        for i in 0..self.num_coefficients() {
            new_poly[i] = self[i].clone() / rhs.clone();
        }
        new_poly
    }
}

impl<T, D, S> Neg for Polynomial<T, D, S>
where
    T: Clone + Zero + Sub<Output = T>,
    D: Dim,
    S: RawStorageMut<T, D> + Clone,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut new_poly = self.clone();
        for i in 0..self.num_coefficients() {
            new_poly[i] = T::zero() - self[i].clone();
        }
        new_poly
    }
}

macro_rules! impl_left_scalar_arithmetic {
    ($($scalar:ty),*) => {
        $(
            impl<D, S> Add<Polynomial<$scalar, D, S>> for $scalar
            where
                $scalar: Clone + Add<Output = $scalar> + Num,
                D: Dim + DimSub<U1>,
                S: RawStorageMut<$scalar, D> + Clone,
            {
                type Output = Polynomial<$scalar, D, S>;

                fn add(self, rhs: Polynomial<$scalar, D, S>) -> Self::Output {
                    let mut new_poly = rhs.clone();
                    let num_coeff = rhs.num_coefficients();
                    if num_coeff > 0 {
                        new_poly[num_coeff - 1] = self + rhs.constant();
                    }
                    new_poly
                }
            }

            impl<D, S> Sub<Polynomial<$scalar, D, S>> for $scalar
            where
                $scalar: Scalar + Sub<Output = $scalar> + Num,
                D: Dim + DimSub<U1>,
                S: RawStorageMut<$scalar, D> + Clone,
            {
                type Output = Polynomial<$scalar, D, S>;

                fn sub(self, rhs: Polynomial<$scalar, D, S>) -> Self::Output {
                    let mut new_poly = -rhs.clone();
                    let num_coeff = rhs.num_coefficients();
                    if num_coeff > 0 {
                        new_poly[num_coeff - 1] = self - new_poly.constant();
                    }
                    new_poly
                }
            }

            impl<D, S> Mul<Polynomial<$scalar, D, S>> for $scalar
            where
                $scalar: Clone + Mul<$scalar, Output = $scalar> + Num,
                D: Dim + DimSub<U1>,
                S: RawStorageMut<$scalar, D> + Clone,
            {
                type Output = Polynomial<$scalar, D, S>;

                fn mul(self, rhs: Polynomial<$scalar, D, S>) -> Self::Output {
                    let mut new_poly = rhs.clone();
                    let num_coeff = rhs.num_coefficients();
                    for i in 0..num_coeff {
                        new_poly[i] = self * rhs[i];
                    }
                    new_poly
                }
            }

            impl<D, S> Div<Polynomial<$scalar, D, S>> for $scalar
            where
                $scalar: Clone + Div<$scalar, Output = $scalar> + Num,
                D: Dim + DimSub<U1>,
                S: RawStorageMut<$scalar, D> + Clone,
            {
                type Output = Polynomial<$scalar, D, S>;

                fn div(self, rhs: Polynomial<$scalar, D, S>) -> Self::Output {
                    let mut new_poly = rhs.clone();
                    let num_coeff = rhs.num_coefficients();
                    for i in 0..num_coeff {
                        new_poly[i] = self / rhs[i];
                    }
                    new_poly
                }
            }
        )*
    };
}

// Example usage:
impl_left_scalar_arithmetic!(i16, u16, u32, i32, f32, f64);

//
//  Polynomial Math
//
// - NO Variable Checking, this only works for polynomials with the same variable
// and does not error
// - No Multivariate support
// - should do <Op>Assign for Add, sub, Mul, Div
// - steal nalgebra impl_arithmatic() macro?

impl<T, D, S, N> Add<Polynomial<T, D, S, N>> for Polynomial<T, D, S, N>
where
    T: Clone + AddAssign,
    D: Dim + DimSub<U1>,
    S: Clone + RawStorageMut<T, D, N>,
    N: Dim,
{
    type Output = Self;

    fn add(self, rhs: Polynomial<T, D, S, N>) -> Self::Output {
        let mut sum = self.clone();
        for i in 0..self.num_coefficients() {
            for j in 0..self.coefficients.shape().1.value() {
                sum[(i, j)] += rhs[(i, j)].clone();
            }
        }
        sum
    }
}

impl<T, D, S, N> Sub<Polynomial<T, D, S, N>> for Polynomial<T, D, S, N>
where
    T: Clone + SubAssign,
    D: Dim + DimSub<U1>,
    S: Clone + RawStorageMut<T, D, N>,
    N: Dim,
{
    type Output = Self;

    fn sub(self, rhs: Polynomial<T, D, S, N>) -> Self::Output {
        let mut dif = self.clone();
        for i in 0..self.num_coefficients() {
            for j in 0..self.coefficients.shape().1.value() {
                dif[(i, j)] -= rhs[(i, j)].clone();
            }
        }
        dif
    }
}

impl<T, D, S, N> Div<Polynomial<T, D, S, N>> for Polynomial<T, D, S, N>
where
    T: Clone + SubAssign,
    D: Dim + DimSub<U1>,
    S: Clone + RawStorageMut<T, D, N>,
    N: Dim,
{
    type Output = Self;

    fn div(self, rhs: Polynomial<T, D, S, N>) -> Self::Output {
        let mut quotient = self.clone() * T::zero();
        let mut remainder = self.clone();
        let divisor_degree = rhs.degree();
        let divisor_lead_coeff = rhs[0].clone();

        while remainder.degree() >= divisor_degree {
            let degree_diff = remainder.degree() - divisor_degree;
            let coeff = remainder[0].clone() / divisor_lead_coeff.clone();

            let mut term = Polynomial::zero();
            term.set_coefficient(degree_diff, coeff.clone());
            quotient += term.clone();
            remainder -= term * rhs.clone();
        }

        quotient
    }
}

// ===============================================================================================
//      Polynomial Num Traits Implementation
// ===============================================================================================

// impl<T, D, S, N> Zero for Polynomial<T, D, S, N>
// where
//     T: Clone + Zero + AddAssign,
//     D: Dim + DimSub<U1>,
//     S: Clone + RawStorageMut<T, D, N>,
//     N: Dim,
// {
//     fn zero() -> Self {
//         Self {
//             variable: "x",
//             coefficients: S::zero(),
//             _phantom: PhantomData,
//         }
//     }

//     fn is_zero(&self) -> bool {
//         // Assuming `coefficients` implements `iter()` or similar trait
//         self.coefficients.iter().all(|&c| c == T::zero())
//     }
// }

// impl<T, D, S, N> One for Polynomial<T, D, S, N>
// where
//     T: One,
//     S: Default + FromIterator<T>,
// {
//     fn one() -> Self {
//         Self {
//             variable: "x",
//             coefficients: core::iter::once(T::one()).collect(),
//             _phantom: PhantomData,
//         }
//     }
// }

// ===============================================================================================
//      Polynomial Display Implementation
// ===============================================================================================

impl<T, D, S> fmt::Display for Polynomial<T, D, S>
where
    T: Copy + Num + PartialOrd + Neg<Output = T> + fmt::Display,
    D: Dim,
    S: RawStorage<T, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_coeff = self.num_coefficients();
        for i in 0..num_coeff {
            let coeff = self[i];
            if coeff == T::zero() {
                continue;
            }

            if i > 0 {
                write!(f, " {} ", if coeff >= T::zero() { "+" } else { "-" })?;
            } else if coeff < T::zero() {
                write!(f, "-")?;
            }

            let abs_coeff = if coeff < T::zero() { -coeff } else { coeff };
            let exp = num_coeff - 1 - i;

            if abs_coeff != T::one() || exp == 0 {
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