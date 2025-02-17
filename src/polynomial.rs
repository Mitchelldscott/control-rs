//!
//! # Polynomial trait and default implementations
//!
//!
use nalgebra::{Complex, RealField, DMatrix};
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
pub fn roots<T: Copy + Float + RealField>(coeff: &[T], root_buffer: &mut [Complex<T>]) {
    // less than 1 coeff has no roots unless len = 1 and coeff[0] = 0 then root[0] = inf
    match coeff.len() {
        0 => {
            root_buffer.fill(Complex::new(T::nan(), T::zero()));
            return;
        }
        1 => {
            if coeff[0].is_zero() {
                root_buffer[0] = Complex::new(T::infinity(), T::zero());
            }
            else {
                root_buffer.fill(Complex::new(T::nan(), T::zero()));
            }
            return;
        }
        _ => {},
    }

    // count coefficients equal to zero
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
        Some(eigenvalues) => eigenvalues.iter().enumerate().for_each(|(i, eigenvalue)| root_buffer[i] = *eigenvalue),
        None => root_buffer.fill(Complex::new(T::nan(), T::zero())),
    }
}

/// tests the trait for computing the roots of a polynomial.
#[cfg(test)]
pub mod basic_roots_tests {
    use super::*;

    #[test]
    fn test_simple_polynomial() {
        let coeff: [f64; 3] = [1.0, 1.0, 1.0]; // x^2 + x + 1
        let mut rbuffer = [Complex::new(0.0, 0.0); 2];
        roots(&coeff, &mut rbuffer);
        assert_eq!(
            rbuffer[0].re,
            -0.49999999999999994
        );
        assert_eq!(
            rbuffer[1].re,
            -0.5
        )
    }

    #[test]
    fn test_polynomial() {
        let coeff: [f64; 4] = [1.0, -6.0, 11.0, -6.0]; // x^3 - 6x^2 + 11x - 6 = 0
        let mut rbuffer = [Complex::new(0.0, 0.0); 3]; // x^3 - 6x^2 + 11x - 6 = 0
        roots(&coeff, &mut rbuffer);
        
        assert_eq!(
            rbuffer[0].re,
            3.000000000000014
        );
        assert_eq!(
            rbuffer[1].re,
            1.9999999999999991
        );
        assert_eq!(
            rbuffer[2].re,
            0.9999999999999999
        );
    }

    #[test]
    fn test_linear_polynomial() {
        let coeff = [1.0, 0.0]; // y = x
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert_eq!(rbuffer[0].re, 0.0);
    }

    #[test]
    fn test_constant_polynomial() {
        let coeff = [1.0]; // y = 1
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_nan());
    }

    #[test]
    fn test_constant_zero_polynomial() {
        let coeff = [0.0]; // y = 0
        let mut rbuffer = [Complex::new(0.0, 0.0)];
        roots(&coeff, &mut rbuffer);
        assert!(rbuffer[0].is_infinite());
    }

    #[test]
    fn test_leading_zero_constant_polynomial() {
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
