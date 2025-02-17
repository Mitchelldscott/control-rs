//!
//! # Polynomial trait and default implementations
//!
//!
use nalgebra::{ComplexField, DMatrix};
use num_traits::Float;

/// Computes the roots of a univariate polynomial of degree `N` by calculating
/// the eigenvalues of its companion matrix. Also provides some edge case checks
/// on the validity of the solution.
///
/// # Arguments
/// * `coeff` - A slice of polynomial coefficients starting from the highest degree term.
///
/// # Returns
///
/// * `[T; N]` - a fixed-size array containing the roots of the polynomial.
///   If roots cannot be computed, it returns an array filled with NaNs (Not a Number).
///
/// ## Edge cases
///
/// - If `len(coeff) <= 1`, the function will return a vector of NaNs since no roots can be determined.
/// - If the leading coefficient `coeff[0] == 0`, the polynomial is degenerate, and the function will attempt to handle the case by reducing the degree.
/// - If all coefficients are zero except the first one, the function assumes the polynomial is a constant and returns NaNs for all roots.
///
///
/// # Example
/// ```rust
/// use control_rs::polynomial::roots;
/// fn main() {
///     let coeff = [1.0, -6.0, 11.0, -6.0]; // x^3 - 6x^2 + 11x - 6 = 0
///     let roots = roots::<f64, 3>(&coeff);
///     println!("{:?}", roots); // Roots: [3.0, 2.0, 1.0]
/// }
/// ```
///
/// ## Edge Case Examples
/// - see basic_roots_tests
pub fn roots<T: Copy + Float + ComplexField>(coeff: &[T], root_buffer: &mut [T]) {
    if coeff.len() <= 1 {
        root_buffer.fill(T::nan());
        if coeff.len() == 1 {
            root_buffer[0] = T::zero();
        }
        return;
    }

    // count coefficients equal to zero
    let num_non_leading_zero = coeff.iter().fold(0, |acc, &c| match c == T::zero() {
        true => acc + 1,
        false => acc,
    });

    // all coeff are zero, no roots
    if coeff.len() - 1 == num_non_leading_zero {
        for n in 0..coeff.len() - 1 {
            root_buffer[n] = T::nan();
        }
        return;
    }
    // first coeff is zero, recurse
    else if coeff[0] == T::zero() {
        return roots::<T>(coeff[1..coeff.len()].try_into().unwrap(), root_buffer);
    }
    // all coeff but the first are zero, all roots = 0
    else if coeff.len() - 1 == num_non_leading_zero {
        for n in 0..coeff.len() - 1 {
            root_buffer[n] = T::nan();
        }
        return;
    }

    // Build the companion matrix
    let companion = DMatrix::from_fn(coeff.len(), coeff.len(), |i, j| {
        if i == 0 {
            -coeff[j + 1] / coeff[0]
        } else {
            if i - 1 == j {
                T::one()
            } else {
                T::zero()
            }
        }
    });

    // Compute the eigenvalues of the companion matrix
    match companion.eigenvalues() {
        Some(eigenvalues) => eigenvalues.iter().enumerate().for_each(|(i, eigenvalue)| root_buffer[i] = *eigenvalue),
        None => {
            for n in 0..coeff.len() - 1 {
                root_buffer[n] = T::nan();
            }
        },
    }
}

/// tests the trait for computing the roots of a polynomial.
#[cfg(test)]
pub mod basic_roots_tests {
    use super::*;

    #[test]
    fn test_simple_polynomial() {
        let coeff: [f64; 4] = [1.0, -6.0, 11.0, -6.0]; // x^3 - 6x^2 + 11x - 6 = 0
        let mut rbuffer: [f64; 3] = [0.0, 0.0, 0.0]; // x^3 - 6x^2 + 11x - 6 = 0
        roots(&coeff, &mut rbuffer);
        assert_eq!(
            rbuffer,
            [3.000000000000014, 1.9999999999999991, 0.9999999999999999]
        );
    }

    #[test]
    fn test_linear_polynomial() {
        let coeff = [1.0, 0.0]; // y = x
        let mut rbuffer = [0.0];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer, [0.0]);
    }

    #[test]
    fn test_constant_polynomial() {
        let coeff = [1.0]; // y = 1
        let mut rbuffer = [0.0];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_leading_zero_constant_polynomial() {
        let coeff = [0.0, 1.0]; // y = 1
        let mut rbuffer = [0.0];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_zero_coefficients() {
        let coeff = [0.0, 0.0]; // No polynomial
        let mut rbuffer = [0.0];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }
}
