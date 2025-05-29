//! A safe and statically sized generic vector implementation
//! TODO:
//!     * constructors
//!         * from_data
//!         * from_element
//!     * meta
//!         * length
//!     * Num Trait
//!         * One
//!         * Zero
//!         * Identity
//!         * Neg
//!     * operators
//!         * Add/AddAssign
//!         * Sub/SubAssign
//!         * Mul/MulAssign
//!         * Div/DivAssign
//!         * Index/IndexMut
//!     * slicing
//!         * col slice
//!         * stride slice

/// A safe and statically sized column vector
///
/// # Generic Arguments
/// * `T` - Type of data the vector contains
/// * `N` - Length of the vector
struct Vector<T, const N: usize> {
    data: [T; N],
}