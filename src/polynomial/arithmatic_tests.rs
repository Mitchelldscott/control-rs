// use super::*;

// fn validate_polynomial_neg<T>() {
//     let empty_p =
// }
//
// #[test]
// fn test_neg() {
//     let p_f32 = Polynomial::new([3.0f32, -2.0f32, 1.0f32]); // 1 - 2x + 3x^2
//     let neg_p_f32 = -p_f32;
//     assert_eq!(neg_p_f32.coefficients, [-1.0f32, 2.0f32, -3.0f32], "neg_p_f32 is not [-1, 2, -3]");
//
//     let p_isize = Polynomial::new([3isize, -2isize, 1isize]);
//     let neg_p_isize = -p_isize;
//     assert_eq!(neg_p_isize.coefficients, [-3isize, 2isize, -1isize], "neg_p_isize is not [-1, 2, -3]");
//
//     let p_i8 = Polynomial::new([0i8, -128i8, 1i8]);
//     let neg_p_i8 = -p_i8;
//     assert_eq!(neg_p_i8.coefficients, [-1i8, 127i8, -0i8], "neg_p_i8 is not [-1, 127, -0]");
// }
//
// #[test]
// fn test_add_scalar() {
//     let p_f32 = Polynomial::new([3.0f32, 2.0f32, 1.0f32]); // 1 + 2x + 3x^2
//     let result_f32 = p_f32 + 5.0f32; // 6 + 2x + 3x^2
//     assert_eq!(result_f32.coefficients, [6.0f32, 2.0f32, 3.0f32], "p_f32 scalar addition failed");
//
//     let p_i128 = Polynomial::new([3i128, 2i128, 1i128]);
//     let result_i128 = p_i128 + 5i128; // 6 + 2x + 3x^2
//     assert_eq!(result_i128.coefficients, [6i128, 2i128, 3i128], "p_i128 scalar addition failed");
//
//     let p_i8 = Polynomial::new([3i8, 2i8, 127i8]);
//     let result_i8 = p_i8 + 5i8; // 6 + 2x + 3x^2
//     assert_eq!(result_i8.coefficients, [-124i8, 2i8, 3i8], "p_i8 scalar addition w/ overflow failed");
// }
//
// #[test]
// fn test_add_assign_scalar() {
//     let mut p = Polynomial::new([3.0, 2.0, 1.0]);
//     p += 5.0;
//     assert_eq!(p.coefficients, [6.0, 2.0, 3.0]);
// }
//
// #[test]
// fn test_mul_scalar() {
//     let p_f64 = Polynomial::new([1.0f64, 2.0f64, 3.0f64]); // 3 + 2x + x^2
//     let result_f64 = p_f64 * 2.0f64; // 6 + 4x + 2x^2
//     assert_eq!(result_f64.coefficients, [6.0f64, 4.0f64, 2.0f64], "p_f64 scalar multiplication failed");
// }
//
// #[test]
// fn test_mul_assign_scalar() {
//     let mut p = Polynomial::new([1.0, 2.0, 3.0]);
//     p *= 2.0;
//     assert_eq!(p.coefficients, [2.0, 4.0, 6.0]);
// }
//
// #[test]
// fn test_add_polynomial_same_size() {
//     let p1 = Polynomial::new([1.0, 2.0]);    // 1 + 2x
//     let p2 = Polynomial::new([3.0, 4.0]);    // 3 + 4x
//     let p_sum = p1 + p2;                     // 4 + 6x
//     assert_eq!(p_sum.coefficients, [4.0, 6.0]);
// }
//
// #[test]
// fn test_add_polynomial_different_sizes() {
//     let p1 = Polynomial::new([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
//     let p2 = Polynomial::new([4.0, 5.0]);      // 4 + 5x
//     let p_sum = p1 + p2;                      // 5 + 7x + 3x^2
//     assert_eq!(p_sum.coefficients, [5.0, 7.0, 3.0]);
//
//     let p3 = Polynomial::new([1.0, 2.0]);      // 1 + 2x
//     let p4 = Polynomial::new([3.0, 4.0, 5.0]); // 3 + 4x + 5x^2
//     let p_sum2 = p3 + p4;                     // 4 + 6x + 5x^2
//     assert_eq!(p_sum2.coefficients, [4.0, 6.0, 5.0]);
// }
//
// #[test]
// fn test_mul_polynomial() {
//     // (1 + x) * (2 + x) = 2 + x + 2x + x^2 = 2 + 3x + x^2
//     let p1 = Polynomial::new([1.0, 1.0]); // 1 + x
//     let p2 = Polynomial::new([2.0, 1.0]); // 2 + x
//     let p_prod = p1 * p2;
//     assert_eq!(p_prod.coefficients, [2.0, 3.0, 1.0, 0.0]); // N+M-1 = 2+2-1 = 3
//
//     // (1 + 2x) * (3 + x^2) = 3 + x^2 + 6x + 2x^3 = 3 + 6x + x^2 + 2x^3
//     let p3 = Polynomial::new([1.0, 2.0]); // 1 + 2x
//     let p4 = Polynomial::new([3.0, 0.0, 1.0]); // 3 + 0x + x^2
//     let p_prod2 = p3 * p4;
//     // Expected size N+M-1 = 2+3-1 = 4
//     assert_eq!(p_prod2.coefficients, [3.0, 6.0, 1.0, 2.0, 0.0, 0.0]);
// }