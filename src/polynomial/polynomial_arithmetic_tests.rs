//! TODO:
//!  - [ ] Scalar Rem
//!  - [ ] Constant Rem
//!
use super::*;
use num_traits::Float;
mod scalar_add {
    use super::*;
    // #[test]
    // fn empty_post_add() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     assert!(!(empty + 1).is_empty());
    // }
    // #[test]
    // fn empty_pre_add() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     assert!(!(1 + empty).is_empty());
    // }
    // #[test]
    // fn empty_add_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     empty += 1;
    //     assert!(!empty.is_empty());
    // }
    #[test]
    fn constant_post_add() {
        let constant: Polynomial<u16, 1> = Polynomial::from_element(1);
        assert_eq!(constant + 1, Polynomial::from_data([2]));
    }
    #[test]
    fn constant_pre_add() {
        let constant: Polynomial<i32, 1> = Polynomial::from_element(1);
        assert_eq!(1 + constant, Polynomial::from_data([2i32]));
    }
    #[test]
    fn constant_add_assign() {
        let mut constant: Polynomial<isize, 1> = Polynomial::from_element(1);
        constant += 1;
        assert_eq!(constant, Polynomial::from_data([2isize]));
    }
    #[test]
    fn line_post_add() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        assert_eq!(line + 1, Polynomial::from_data([2, 1]));
    }
    #[test]
    fn line_pre_add() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
        assert_eq!(1.0 + line, Polynomial::from_data([2.0, 1.0]));
    }
    #[test]
    fn line_add_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        line += 1.0;
        assert_eq!(line, Polynomial::from_data([2.0, 1.0]));
    }
    #[test]
    fn quadratic_post_add() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        assert_eq!(quadratic + 1i64, Polynomial::from_data([2, 1, 1]));
    }
    #[test]
    fn quadratic_pre_add() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(1);
        assert_eq!(1 + quadratic, Polynomial::from_data([2, 1, 1]));
    }
    #[test]
    fn quadratic_add_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        quadratic += 1;
        assert_eq!(quadratic, Polynomial::from_data([2, 1, 1]));
    }
    #[test]
    fn large_post_add() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(1);
        assert_eq!(
            large + 1,
            Polynomial::from_data([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }
    #[test]
    fn large_pre_add() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(1);
        assert_eq!(
            1 + large,
            Polynomial::from_data([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }
    #[test]
    fn large_add_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(1);
        large += 1;
        assert_eq!(
            large,
            Polynomial::from_data([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }
}

mod scalar_sub {
    use super::*;
    // #[test]
    // fn empty_post_sub() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     assert!(!(empty - 1).is_empty());
    // }
    // #[test]
    // fn empty_pre_sub() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     assert!(!(1 - empty).is_empty());
    // }
    // #[test]
    // fn empty_sub_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     empty -= 1;
    //     assert!(!empty.is_empty());
    // }
    #[test]
    fn constant_post_sub() {
        let constant: Polynomial<u16, 1> = Polynomial::from_element(1);
        assert_eq!(constant - 1, Polynomial::from_data([0]));
    }
    #[test]
    fn constant_pre_sub() {
        let constant: Polynomial<i32, 1> = Polynomial::from_element(1);
        assert_eq!(1 - constant, Polynomial::from_data([0i32]));
    }
    #[test]
    fn constant_sub_assign() {
        let mut constant: Polynomial<isize, 1> = Polynomial::from_element(1);
        constant -= 1;
        assert_eq!(constant, Polynomial::from_data([0isize]));
    }
    #[test]
    fn line_post_sub() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        assert_eq!(line - 1, Polynomial::from_data([0, 1]));
    }
    #[test]
    fn line_pre_sub() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
        assert_eq!(1.0 - line, Polynomial::from_data([0.0, -1.0]));
    }
    #[test]
    fn line_sub_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        line -= 1.0;
        assert_eq!(line, Polynomial::from_data([0.0, 1.0]));
    }
    #[test]
    fn quadratic_post_sub() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        assert_eq!(quadratic - 1, Polynomial::from_data([0, 1, 1]));
    }
    #[test]
    fn quadratic_pre_sub() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(0);
        assert_eq!(1 - quadratic, Polynomial::from_data([1, 0, 0]));
    }
    #[test]
    fn quadratic_sub_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        quadratic -= 1;
        assert_eq!(quadratic, Polynomial::from_data([0, 1, 1]));
    }
    #[test]
    fn large_post_sub() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(1);
        assert_eq!(
            large - 1,
            Polynomial::from_data([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }
    #[test]
    fn large_pre_sub() {
        let large: Polynomial<u32, 16> = Polynomial::from_iterator([1]);
        assert_eq!(
            1 - large,
            Polynomial::from_data([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
    }
    #[test]
    fn large_sub_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(1);
        large -= 1;
        assert_eq!(
            large,
            Polynomial::from_data([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        );
    }
}

mod scalar_mul {
    use super::*;
    // #[test]
    // fn empty_post_mul() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     assert!((empty * 1).is_empty());
    // }
    // #[test]
    // fn empty_pre_mul() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     assert!((1 * empty).is_empty());
    // }
    // #[test]
    // fn empty_mul_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     empty *= 1;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_post_mul() {
        let constant: Polynomial<u16, 1> = Polynomial::from_element(1);
        assert_eq!(constant * 4, Polynomial::from_data([4]));
    }
    #[test]
    fn constant_pre_mul() {
        let constant: Polynomial<i32, 1> = Polynomial::from_element(6);
        assert_eq!(3 * constant, Polynomial::from_data([18i32]));
    }
    #[test]
    fn constant_mul_assign() {
        let mut constant: Polynomial<isize, 1> = Polynomial::from_element(1);
        constant *= 0;
        assert_eq!(constant, Polynomial::from_data([0isize]));
    }
    #[test]
    fn line_post_mul() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(3);
        assert_eq!(line * 2, Polynomial::from_data([6, 6]));
    }
    #[test]
    fn line_pre_mul() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
        assert_eq!(0.0 * line, Polynomial::from_data([0.0, 0.0]));
    }
    #[test]
    fn line_mul_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        line *= 0.0;
        assert_eq!(line, Polynomial::from_data([0.0, 0.0]));
    }
    #[test]
    fn quadratic_post_mul() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        assert_eq!(quadratic * 10, Polynomial::from_data([10, 10, 10]));
    }
    #[test]
    fn quadratic_pre_mul() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(2);
        assert_eq!(10 * quadratic, Polynomial::from_data([20, 20, 20]));
    }
    #[test]
    fn quadratic_mul_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        quadratic *= 0;
        assert_eq!(quadratic, Polynomial::from_data([0, 0, 0]));
    }
    #[test]
    fn large_post_mul() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(0);
        assert_eq!(large * 10, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_pre_mul() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(0);
        assert_eq!(10 * large, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_mul_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(1);
        large *= 0;
        assert_eq!(large, Polynomial::from_iterator([]));
    }
}

mod scalar_div {
    use super::*;
    // #[test]
    // fn empty_post_div() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     assert!((empty / 1).is_empty());
    // }
    // #[test]
    // fn empty_pre_div() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     assert!((1 / empty).is_empty());
    // }
    // #[test]
    // fn empty_div_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     empty /= 1;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_post_div() {
        let constant: Polynomial<u16, 1> = Polynomial::from_element(1);
        assert_eq!(constant / 1, Polynomial::from_data([1]));
    }
    #[test]
    fn constant_pre_div() {
        let constant: Polynomial<i32, 1> = Polynomial::from_element(2);
        assert_eq!(2 / constant, Polynomial::from_data([1i32]));
    }
    #[test]
    fn constant_div_assign() {
        let mut constant: Polynomial<isize, 1> = Polynomial::from_element(3);
        constant /= 3;
        assert_eq!(constant, Polynomial::from_data([1isize]));
    }
    #[test]
    fn line_post_div() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(4);
        assert_eq!(line / 4, Polynomial::from_data([1, 1]));
    }
    #[test]
    fn line_div_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(6.0);
        line /= 6.0;
        assert_eq!(line, Polynomial::from_data([1.0, 1.0]));
    }
    #[test]
    fn quadratic_post_div() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(7);
        assert_eq!(quadratic / 7, Polynomial::from_data([1, 1, 1]));
    }
    #[test]
    fn quadratic_div_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(9);
        quadratic /= 9;
        assert_eq!(quadratic, Polynomial::from_data([1, 1, 1]));
    }
    #[test]
    fn large_post_div() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(10);
        assert_eq!(large / 10, Polynomial::from_element(1));
    }
    #[test]
    fn large_div_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(12);
        large /= 12;
        assert_eq!(large, Polynomial::from_element(1));
    }
}

mod const_add {
    use super::*;
    // #[test]
    // fn empty_post_add() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i8]);
    //     assert!(!(empty + constant).is_empty());
    // }
    // #[test]
    // fn empty_pre_add() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1u8]);
    //     assert!(!(constant + empty).is_empty());
    // }
    // #[test]
    // fn empty_add_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i16]);
    //     empty += constant;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_add() {
        let constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        assert_eq!(constant_a + constant_b, Polynomial::from_data([4]));
    }
    #[test]
    fn constant_add_assign() {
        let mut constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        constant_a += constant_b;
        assert_eq!(constant_a, Polynomial::from_data([4]));
    }
    #[test]
    fn line_post_add() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([usize::MAX - 1]);
        assert_eq!(line + constant, Polynomial::from_data([usize::MAX, 1]));
    }
    #[test]
    fn line_pre_add() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([-1.0f32]);
        assert_eq!(constant + line, Polynomial::from_data([0.0, 1.0]));
    }
    #[test]
    fn line_add_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([10.0f64]);
        line += constant;
        assert_eq!(line, Polynomial::from_data([11.0, 1.0]));
    }
    #[test]
    fn quadratic_post_add() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([3i64]);
        assert_eq!(quadratic + constant, Polynomial::from_data([4, 1, 1]));
    }
    #[test]
    fn quadratic_pre_add() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([1u32]);
        assert_eq!(constant + quadratic, Polynomial::from_data([2, 1, 1]));
    }
    #[test]
    fn quadratic_add_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([4i64]);
        quadratic += constant;
        assert_eq!(quadratic, Polynomial::from_data([5, 1, 1]));
    }
    #[test]
    fn large_post_add() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([0i32]);
        assert_eq!(large + constant, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_pre_add() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([u32::MAX]);
        assert_eq!(constant + large, Polynomial::from_iterator([u32::MAX]));
    }
    #[test]
    fn large_add_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([0i64]);
        large += constant;
        assert_eq!(large, Polynomial::from_iterator([]));
    }
}

mod const_sub {
    use super::*;
    // #[test]
    // fn empty_post_sub() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i8]);
    //     assert!(!(empty - constant).is_empty());
    // }
    // #[test]
    // fn empty_pre_sub() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1u8]);
    //     assert!(!(constant - empty).is_empty());
    // }
    // #[test]
    // fn empty_sub_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i16]);
    //     empty -= constant;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_sub() {
        let constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        assert_eq!(constant_a - constant_b, Polynomial::from_data([0]));
    }
    #[test]
    fn constant_sub_assign() {
        let mut constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        constant_a -= constant_b;
        assert_eq!(constant_a, Polynomial::from_data([0]));
    }
    #[test]
    fn line_post_sub() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0usize]);
        assert_eq!(line - constant, Polynomial::from_data([1, 1]));
    }
    #[test]
    fn line_pre_sub() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(10_000_000.0);
        let constant = Polynomial::from_data([1.0f32]);
        assert_eq!(
            constant - line,
            Polynomial::from_data([-9_999_999.0, -10_000_000.0])
        );
    }
    #[test]
    fn line_sub_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([1.0f64]);
        line -= constant;
        assert_eq!(line, Polynomial::from_data([0.0, 1.0]));
    }
    #[test]
    fn quadratic_post_sub() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([1i64]);
        assert_eq!(quadratic - constant, Polynomial::from_data([0, 1, 1]));
    }
    #[test]
    fn quadratic_pre_sub() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([1u32]);
        assert_eq!(constant - quadratic, Polynomial::from_data([1, 0, 0]));
    }
    #[test]
    fn quadratic_sub_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([1i64]);
        quadratic -= constant;
        assert_eq!(quadratic, Polynomial::from_data([0, 1, 1]));
    }
    #[test]
    fn large_post_sub() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([1i32]);
        assert_eq!(large - constant, Polynomial::from_iterator([-1]));
    }
    #[test]
    fn large_pre_sub() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([0u32]);
        assert_eq!(constant - large, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_sub_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([0i64]);
        large -= constant;
        assert_eq!(large, Polynomial::from_iterator([]));
    }
}

mod const_mul {
    use super::*;
    // #[test]
    // fn empty_post_mul() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i8]);
    //     assert!((empty * constant).is_empty());
    // }
    // #[test]
    // fn empty_pre_mul() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1u8]);
    //     assert!((constant * empty).is_empty());
    // }
    // #[test]
    // fn empty_mul_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i16]);
    //     empty *= constant;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_mul() {
        let constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        assert_eq!(constant_a * constant_b, Polynomial::from_data([4]));
    }
    #[test]
    fn constant_mul_assign() {
        let mut constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        constant_a *= constant_b;
        assert_eq!(constant_a, Polynomial::from_data([4]));
    }
    #[test]
    fn line_post_mul() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0usize]);
        assert_eq!(line * constant, Polynomial::from_data([0, 0]));
    }
    #[test]
    fn line_pre_mul() {
        let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([0.0f32]);
        assert_eq!(constant * line, Polynomial::from_data([0.0, 0.0]));
    }
    #[test]
    fn line_mul_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([0.0f64]);
        line *= constant;
        assert_eq!(line, Polynomial::from_data([0.0, 0.0]));
    }
    #[test]
    fn quadratic_post_mul() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0i64]);
        assert_eq!(quadratic * constant, Polynomial::from_data([0, 0, 0]));
    }
    #[test]
    fn quadratic_pre_mul() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0u32]);
        assert_eq!(constant * quadratic, Polynomial::from_data([0, 0, 0]));
    }
    #[test]
    fn quadratic_mul_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0i64]);
        quadratic *= constant;
        assert_eq!(quadratic, Polynomial::from_data([0, 0, 0]));
    }
    #[test]
    fn large_post_mul() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0i32]);
        assert_eq!(large * constant, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_pre_mul() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0u32]);
        assert_eq!(constant * large, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_mul_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0i64]);
        large *= constant;
        assert_eq!(large, Polynomial::from_iterator([]));
    }
}

mod const_div {
    use super::*;
    // #[test]
    // fn empty_post_mul() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i8]);
    //     assert!((empty * constant).is_empty());
    // }
    // #[test]
    // fn empty_pre_mul() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1u8]);
    //     assert!((constant * empty).is_empty());
    // }
    // #[test]
    // fn empty_mul_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([1i16]);
    //     empty *= constant;
    //     assert!(empty.is_empty());
    // }
    #[test]
    fn constant_div() {
        let constant_a: Polynomial<u16, 1> = Polynomial::from_element(2);
        let constant_b = Polynomial::from_data([2u16]);
        assert_eq!(constant_a / constant_b, Polynomial::from_data([1]));
    }
    #[test]
    fn constant_div_assign() {
        let mut constant_a: Polynomial<u16, 1> = Polynomial::from_element(4);
        let constant_b = Polynomial::from_data([2u16]);
        constant_a /= constant_b;
        assert_eq!(constant_a, Polynomial::from_data([2]));
    }
    #[test]
    fn line_post_div() {
        let line: Polynomial<usize, 2> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([0usize]);
        assert_eq!(line / constant, Polynomial::from_data([0, 0]));
    }
    // #[test]
    // fn line_pre_div() {
    //     let line: Polynomial<f32, 2> = Polynomial::from_element(1.0);
    //     let constant = Polynomial::from_data([0.0f32]);
    //     assert_eq!(constant / line, Polynomial::from_data([0.0, 0.0]));
    // }
    #[test]
    fn line_div_assign() {
        let mut line: Polynomial<f64, 2> = Polynomial::from_element(1.0);
        let constant = Polynomial::from_data([0.0f64]);
        line /= constant;
        assert_eq!(
            line,
            Polynomial::from_data([f64::infinity(), f64::infinity()])
        );
    }
    #[test]
    fn quadratic_post_div() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let constant = Polynomial::from_data([2i64]);
        assert_eq!(quadratic / constant, Polynomial::from_data([0, 0, 0]));
    }
    // #[test]
    // fn quadratic_pre_div() {
    //     let quadratic: Polynomial<u32, 3> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([0u32]);
    //     assert_eq!(constant * quadratic, Polynomial::from_data([0, 0, 0]));
    // }
    #[test]
    fn quadratic_div_assign() {
        let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(2);
        let constant = Polynomial::from_data([2i64]);
        quadratic /= constant;
        assert_eq!(quadratic, Polynomial::from_data([1, 1, 1]));
    }
    #[test]
    fn large_post_div() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([1i32]);
        assert_eq!(large / constant, Polynomial::from_iterator([]));
    }
    // #[test]
    // fn large_pre_div() {
    //     let large: Polynomial<u32, 16> = Polynomial::from_element(1);
    //     let constant = Polynomial::from_data([0u32]);
    //     assert_eq!(constant * large, Polynomial::from_iterator([]));
    // }
    #[test]
    fn large_div_assign() {
        let mut large: Polynomial<i64, 16> = Polynomial::from_element(0);
        let constant = Polynomial::from_data([1i64]);
        large /= constant;
        assert_eq!(large, Polynomial::from_iterator([]));
    }
}

mod line_mul {
    use super::*;
    // #[test]
    // fn empty_post_mul() {
    //     let empty: Polynomial<i8, 0> = Polynomial::from_element(1);
    //     let line = Polynomial::from_data([1i8, 1i8]);
    //     assert!((empty * line).is_empty());
    // }
    // #[test]
    // fn empty_pre_mul() {
    //     let empty: Polynomial<u8, 0> = Polynomial::from_element(1);
    //     let line = Polynomial::from_data([1u8, 1u8]);
    //     assert!((line * empty).is_empty());
    // }
    // #[test]
    // fn empty_mul_assign() {
    //     let mut empty: Polynomial<i16, 0> = Polynomial::from_element(1);
    //     let line = Polynomial::from_data([1i16, 1i16]);
    //     empty *= line;
    //     assert!(empty.is_empty());
    // }
    // #[test]
    // fn constant_post_mul() {
    //     let mut constant: Polynomial<u16, 1> = Polynomial::from_element(2);
    //     let line = Polynomial::from_data([1u16, 1u16]);
    //     assert_eq!(constant * line, Polynomial::from_data([2, 2]));
    // }
    // #[test]
    // fn constant_pre_mul() {
    //     let mut constant: Polynomial<u16, 1> = Polynomial::from_element(2);
    //     let line = Polynomial::from_data([1u16, 1u16]);
    //     assert_eq!(line * constant, Polynomial::from_data([2, 2]));
    // }
    // #[test]
    // fn constant_mul_assign() {
    //     let mut constant: Polynomial<u16, 1> = Polynomial::from_element(2);
    //     let line = Polynomial::from_data([1u16, 1u16]);
    //     constant *= line;
    //     assert_eq!(constant, Polynomial::from_data([4]));
    // }
    #[test]
    fn line_mul() {
        let line_a: Polynomial<usize, 2> = Polynomial::from_element(1);
        let line_b = Polynomial::from_data([0usize, 0usize]);
        assert_eq!(line_a * line_b, Polynomial::from_data([0, 0, 0]));
    }
    // #[test]
    // fn line_mul_assign() {
    //     let mut line_a: Polynomial<f64, 2> = Polynomial::from_element(1.0);
    //     let line_b = Polynomial::from_data([0.0f64, 0.0f64]);
    //     line_a *= line_b;
    //     assert_eq!(line_a, Polynomial::from_data([0.0, 0.0]));
    // }
    #[test]
    fn quadratic_post_mul() {
        let quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
        let line = Polynomial::from_data([0i64, 0i64]);
        assert_eq!(quadratic * line, Polynomial::from_data([0, 0, 0, 0]));
    }
    #[test]
    fn quadratic_pre_mul() {
        let quadratic: Polynomial<u32, 3> = Polynomial::from_element(1);
        let line = Polynomial::from_data([0u32, 0u32]);
        assert_eq!(line * quadratic, Polynomial::from_data([0, 0, 0, 0]));
    }
    // #[test]
    // fn quadratic_mul_assign() {
    //     let mut quadratic: Polynomial<i64, 3> = Polynomial::from_element(1);
    //     let line = Polynomial::from_data([0i64, 0i64]);
    //     quadratic *= line;
    //     assert_eq!(quadratic, Polynomial::from_data([0, 0, 0]));
    // }
    #[test]
    fn large_post_mul() {
        let large: Polynomial<i32, 16> = Polynomial::from_element(1);
        let line = Polynomial::from_data([0i32, 0i32]);
        assert_eq!(large * line, Polynomial::from_iterator([]));
    }
    #[test]
    fn large_pre_mul() {
        let large: Polynomial<u32, 16> = Polynomial::from_element(1);
        let line = Polynomial::from_data([0u32, 0u32]);
        assert_eq!(line * large, Polynomial::from_iterator([]));
    }
    // #[test]
    // fn large_mul_assign() {
    //     let mut large: Polynomial<i64, 16> = Polynomial::from_element(1);
    //     let line = Polynomial::from_data([0i64, 0i64]);
    //     large *= line;
    //     assert_eq!(large, Polynomial::from_iterator([]));
    // }
}
