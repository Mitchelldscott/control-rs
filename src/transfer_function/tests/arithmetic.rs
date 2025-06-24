use super::*;

mod addition {
    use super::*;
    #[test]
    fn test_add() {
        let tf = TransferFunction::new([1], [1, 1]);
        let sum = tf + TransferFunction::new([1], [1, 2]);
        assert_eq!(sum.numerator, [3, 2], "Incorrect numerator");
        assert_eq!(sum.denominator, [2, 3, 1], "Incorrect denominator");
    }

    #[test]
    fn test_add_one() {
        let tf = TransferFunction::new([1], [1]);
        let sum = tf + TransferFunction::new([1], [1, 2]);
        assert_eq!(sum.numerator, [3, 1], "Incorrect numerator");
        assert_eq!(sum.denominator, [2, 1], "Incorrect denominator");
    }

    #[test]
    fn test_add_zero() {
        let tf = TransferFunction::new([0], [1]);
        let sum = tf + TransferFunction::new([1], [1, 2]);
        assert_eq!(sum.numerator, [1, 0], "Incorrect numerator");
        assert_eq!(sum.denominator, [2, 1], "Incorrect denominator");
    }
}

mod subtraction {
    use super::*;
    #[test]
    fn test_sub() {
        let tf = TransferFunction::new([1], [1, 1]);
        let sum = tf - TransferFunction::new([1], [1, 1]);
        assert_eq!(sum.numerator, [0, 0], "Incorrect numerator");
        assert_eq!(sum.denominator, [1, 2, 1], "Incorrect denominator");
    }

    #[test]
    fn test_sub_one() {
        let tf = TransferFunction::new([1], [1, 1]);
        let sum = tf - TransferFunction::new([1], [1]);
        assert_eq!(sum.numerator, [0, -1], "Incorrect numerator");
        assert_eq!(sum.denominator, [1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_sub_zero() {
        let tf = TransferFunction::new([1], [1, 1]);
        let sum = tf - TransferFunction::new([0], [1]);
        assert_eq!(sum.numerator, [1, 0], "Incorrect numerator");
        assert_eq!(sum.denominator, [1, 1], "Incorrect denominator");
    }
}

mod multiplication {
    use super::*;
    #[test]
    fn test_mul() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf * TransferFunction::new([1], [1, 2]);
        assert_eq!(product.numerator, [1], "Incorrect numerator");
        assert_eq!(product.denominator, [2, 3, 1], "Incorrect denominator");
    }

    #[test]
    fn test_mul_one() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf * TransferFunction::new([1], [1]);
        assert_eq!(product.numerator, [1], "Incorrect numerator");
        assert_eq!(product.denominator, [1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_mul_zero() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf * TransferFunction::new([0], [1]);
        assert_eq!(product.numerator, [0], "Incorrect numerator");
        assert_eq!(product.denominator, [1, 1], "Incorrect denominator");
    }
}

mod division {
    use super::*;
    #[test]
    fn test_div() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf / TransferFunction::new([1], [1, 1]);
        assert_eq!(product.numerator, [1, 1], "Incorrect numerator");
        assert_eq!(product.denominator, [1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_div_one() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf / TransferFunction::new([1], [1]);
        assert_eq!(product.numerator, [1], "Incorrect numerator");
        assert_eq!(product.denominator, [1, 1], "Incorrect denominator");
    }

    #[test]
    fn test_div_zero() {
        let tf = TransferFunction::new([1], [1, 1]);
        let product = tf / TransferFunction::new([0], [1]);
        assert_eq!(product.numerator, [1], "Incorrect numerator");
        assert_eq!(product.denominator, [0, 0], "Incorrect denominator");
    }
}

mod cl {
    use crate::systems::System;
    use crate::TransferFunction;
    #[test]
    fn test_cl() {
        let tf = TransferFunction::new([1], [1, 1]);
        // sign_in * sys1.clone() / (GH::identity() + sign_feedback * sys1.clone() * sys2.clone())
        let ident: TransferFunction<i32, 1, 2> = TransferFunction::identity();
        let num: TransferFunction<i32, 1, 2> = 1 * tf;
        let fb: TransferFunction<i32, 1, 2> = -1 * tf * 1;
        let den: TransferFunction<i32, 2, 3> = ident + fb;
        let cl: TransferFunction<i32, 3, 3> = num / den;
        assert_eq!(cl.numerator, [1, 1, 0], "Incorrect numerator");
        assert_eq!(cl.denominator, [0, 1, 1], "Incorrect denominator");
    }
}
