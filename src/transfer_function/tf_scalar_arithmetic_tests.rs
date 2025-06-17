//! Test arithmetic operation for transfer function
//!
//! These rely heavily on the polynomial arithmetic operations, so check those too.

use super::*;

mod addition {
    use super::*;
    #[test]
    fn test_add() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_plus_one = tf.add(1);
        assert_eq!(tf_plus_one.numerator, [2, 1, 1], "Incorrect numerator");
        assert_eq!(tf_plus_one.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_add_zero() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_plus_zero = tf.add(0);
        assert_eq!(tf_plus_zero.numerator, [1, 0, 0], "Incorrect numerator");
        assert_eq!(tf_plus_zero.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_add_assign() {
        let mut tf = TransferFunction::new([0, 0, 0, 1], [1, 1, 1]);
        tf.add_assign(1);
        assert_eq!(tf.numerator, [2, 1, 1, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_add_assign_zero() {
        let mut tf = TransferFunction::new([0, 0, 1], [1, 1, 1]);
        tf.add_assign(0);
        assert_eq!(tf.numerator, [1, 0, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }
}

mod subtraction {
    use super::*;
    #[test]
    fn test_sub() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_minus_one = tf.sub(1);
        assert_eq!(tf_minus_one.numerator, [0, -1, -1], "Incorrect numerator");
        assert_eq!(tf_minus_one.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_sub_zero() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_minus_zero = tf.sub(0);
        assert_eq!(tf_minus_zero.numerator, [1, 0, 0], "Incorrect numerator");
        assert_eq!(
            tf_minus_zero.denominator,
            [1, 1, 1],
            "Incorrect denominator"
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut tf = TransferFunction::new([0, 0, 0, 1], [1, 1, 1]);
        tf.sub_assign(1);
        assert_eq!(tf.numerator, [0, -1, -1, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_sub_assign_zero() {
        let mut tf = TransferFunction::new([0, 0, 1], [1, 1, 1]);
        tf.sub_assign(0);
        assert_eq!(tf.numerator, [1, 0, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }
}

mod multiplication {
    use super::*;
    #[test]
    fn test_mul() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_times_one = tf.mul(1);
        assert_eq!(tf_times_one.numerator, [1], "Incorrect numerator");
        assert_eq!(tf_times_one.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_mul_zero() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_times_zero = tf.mul(0);
        assert_eq!(tf_times_zero.numerator, [0], "Incorrect numerator");
        assert_eq!(
            tf_times_zero.denominator,
            [1, 1, 1],
            "Incorrect denominator"
        );
    }

    #[test]
    fn test_mul_assign() {
        let mut tf = TransferFunction::new([1], [1, 1, 1]);
        tf.mul_assign(1);
        assert_eq!(tf.numerator, [1], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_mul_assign_zero() {
        let mut tf = TransferFunction::new([0, 0, 1], [1, 1, 1]);
        tf.mul_assign(0);
        assert_eq!(tf.numerator, [0, 0, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [1, 1, 1], "Incorrect denominator");
    }
}

mod division {
    use super::*;
    #[test]
    fn test_div() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_div_two = tf.div(2);
        assert_eq!(tf_div_two.numerator, [1], "Incorrect numerator");
        assert_eq!(tf_div_two.denominator, [2, 2, 2], "Incorrect denominator");
    }

    #[test]
    fn test_div_zero() {
        let tf = TransferFunction::new([1], [1, 1, 1]);
        let tf_div_zero = tf.div(0);
        assert_eq!(tf_div_zero.numerator, [1], "Incorrect numerator");
        assert_eq!(tf_div_zero.denominator, [0, 0, 0], "Incorrect denominator");
    }

    #[test]
    fn test_div_assign() {
        let mut tf = TransferFunction::new([1], [1, 1, 1]);
        tf.div_assign(2);
        assert_eq!(tf.numerator, [1], "Incorrect numerator");
        assert_eq!(tf.denominator, [2, 2, 2], "Incorrect denominator");
    }

    #[test]
    fn test_div_assign_zero() {
        let mut tf = TransferFunction::new([0, 0, 1], [1, 1, 1]);
        tf.div_assign(0);
        assert_eq!(tf.numerator, [1, 0, 0], "Incorrect numerator");
        assert_eq!(tf.denominator, [0, 0, 0], "Incorrect denominator");
    }
}
