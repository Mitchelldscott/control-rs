//! Simple and safe static sized matrix implementation
//!
//! TODO: 
//!     * constructors
//!         * from_data
//!         * from_element
//!         * from_row
//!         * from_diagonal
//!         * from_column
//!     * meta
//!         * shape
//!         * rows
//!         * cols
//!     * Num Trait
//!         * One
//!         * Zero
//!         * Identity
//!         * Neg
//!     * decompositions
//!         * Lower Upper
//!         * Cholesky
//!         * Singular Value
//!         * QR
//!     * operators
//!         * Add/AddAssign
//!         * Sub/SubAssign
//!         * Mul/MulAssign
//!         * Div/DivAssign
//!         * Index/IndexMut (linear and (row,col))
//!     * slicing
//!         * row slice
//!         * col slice
//!         * stride slice 

/// A static sized 2-dimensional matrix implementation
///
/// # Generic Arguments
/// * `T` - The type of data the matrix contains
/// * `R` - Number of rows
/// * `C` - Number of columns
struct Matrix<T, const R: usize, const C: usize> {
    /// the data of the matrix
    data: [[T; R]; C],
}


// impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
//     /// Create a new matrix from a 2D array
//     ///
//     /// # Arguments
//     /// * `data` - 2D array of values (inner shape = R, outer shape = C)
//     ///
//     /// # Returns
//     /// * `matrix` - a 2-dimensional matrix object
//     pub fn new(data: [[T; R]; C]) -> Self { Matrix { data } }
// }
//
// impl<T, const R: usize, const C: usize> Index<(usize, usize)> Matrix<T, R, C> {}