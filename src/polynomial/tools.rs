//! Helper tools for working with polynomials
//!
use nalgebra::{Complex, DMatrix, RealField};
use num_traits::Float;

/// Assembles the companion matrix of a polynomial.
///
/// The companion matrix is a square matrix that is used to: compute the roots of a polynomial and convert
/// [crate::TransferFunction] to [crate::StateSpace].

/// Computes the roots of a univariate polynomial of degree `N` by calculating
/// the eigenvalues of its companion matrix. Also provides some edge case checks
/// on the validity of the solution.
///
/// # Arguments
/// * `coeff` - A slice of polynomial coefficients starting from the highest degree term.
/// * `root_buffer` - A mutable slice of Complex numbers to store the roots of the polynomial
/// (should always have room for coeff.len()-1, the 'worst case').
///
/// # Returns
///
/// * `[T; N]` - a fixed-size array containing the roots of the polynomial.
///   If roots cannot be computed, it returns an array filled with NaNs (Not a Number).
///
/// ## Edge cases
///
/// - If `len(coeff) < 1`, the function will return an array of Complex(NaN, NaN).
/// - If the leading coefficient `coeff[0] == 0` the function will recurse with a reduced degree.
/// - If all coefficients are zero, the function will return an array of Complex(Infinities, 0).?? maybe shoule be Complex(Inf, NaN)
///
/// # Example
/// ```rust
/// use nalgebra::Complex;
/// use control_rs::polynomial::roots;
/// fn main() {
///     let coeff = [1.0, 2.0, 3.0, 4.0];
///     let mut rbuffer = [Complex::new(0.0, 0.0); 3];
///     let roots = roots::<f64>(&coeff, &mut rbuffer);
///     println!("{roots:?}");
/// }
/// ```
///
/// ## Edge Case Examples
/// - see basic_roots_tests
pub fn roots<T: Copy + Float + RealField>(coeff: &[T], root_buffer: &mut [Complex<T>]) {
    // less than 1 coeff has no roots, and len = 1, is nan unless coeff[0] = 0 then root[0] = inf
    match coeff.len() {
        0 => {
            // if root buffer is much bigger than necessary this significantly hurts performance
            root_buffer.fill(Complex::new(T::nan(), T::nan()));
            return;
        }
        1 => {
            if coeff[0].is_zero() {
                root_buffer[0] = Complex::new(T::infinity(), T::zero());
            } else {
                root_buffer.fill(Complex::new(T::nan(), T::nan()));
            }
            return;
        }
        _ => {}
    }

    // for efficiency? avoid recursing in the base case: all coeff == 0
    let num_zero_coeff = coeff.iter().fold(0, |acc, &c| match c == T::zero() {
        true => acc + 1,
        false => acc,
    });

    // all coeff are zero
    if coeff.len() == num_zero_coeff {
        // constant 0, everything is a root
        root_buffer[0] = Complex::new(T::infinity(), T::zero());
        return;
    }

    // first coeff is zero, recurse
    if coeff[0] == T::zero() {
        return roots::<T>(coeff[1..coeff.len()].try_into().unwrap(), root_buffer);
    }

    // Build the companion matrix
    let companion = DMatrix::from_fn(coeff.len() - 1, coeff.len() - 1, |i, j| {
        if i == 0 {
            Complex::new(-coeff[j + 1] / coeff[0], T::zero())
        } else {
            if i - 1 == j {
                Complex::new(T::one(), T::zero())
            } else {
                Complex::new(T::zero(), T::zero())
            }
        }
    });

    // Compute the eigenvalues of the companion matrix
    match companion.eigenvalues() {
        Some(eigenvalues) => eigenvalues
            .iter()
            .enumerate()
            .for_each(|(i, eigenvalue)| root_buffer[i] = *eigenvalue),
        None => root_buffer.fill(Complex::new(T::nan(), T::nan())),
    }
}

/// tests the trait for computing the roots of a polynomial.
#[cfg(test)]
pub mod basic_roots_tests {
    use super::*;

    #[test]
    fn test_standard() {
        let coeff: [f64; 3] = [1.0, 1.0, 1.0]; // x^2 + x + 1
        let mut rbuffer = [Complex::new(0.0, 0.0); 2];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, -0.49999999999999994);
        assert_eq!(rbuffer[0].im, 0.8660254037844386);
        assert_eq!(rbuffer[1].re, -0.5);
        assert_eq!(rbuffer[1].im, -0.8660254037844386);
    }

    #[test]
    fn test_three_two_one() {
        let coeff: [f64; 4] = [1.0, -6.0, 11.0, -6.0]; // x^3 - 6x^2 + 11x - 6 = 0
        let mut rbuffer = [Complex::new(0.0, 0.0); 3]; // x^3 - 6x^2 + 11x - 6 = 0
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, 3.000000000000014);
        assert_eq!(rbuffer[0].im, 0.0);
        assert_eq!(rbuffer[1].re, 1.9999999999999991);
        assert_eq!(rbuffer[1].im, 0.0);
        assert_eq!(rbuffer[2].re, 0.9999999999999999);
        assert_eq!(rbuffer[2].im, 0.0);
    }

    #[test]
    fn test_linear() {
        let coeff = [1.0, 0.0]; // y = x
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, 0.0);
        assert_eq!(rbuffer[0].im, 0.0);
    }

    #[test]
    fn test_constant() {
        let coeff = [1.0]; // y = 1
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_constant_zero() {
        let coeff = [0.0]; // y = 0
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_infinite());
    }

    #[test]
    fn test_leading_zero_constant() {
        let coeff = [0.0, 1.0]; // y = 1
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_multiple_zero_coefficients() {
        let coeff = [0.0, 0.0]; // y = 0
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_infinite());
    }
}
