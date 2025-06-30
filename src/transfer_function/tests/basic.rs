//! Basic test cases to make sure the Transfer Function is usable
use crate::{assert_f64_eq, transfer_function::*};

#[test]
fn initialize_integrator() {
    let tf = TransferFunction::new([1.0], [1.0, 0.0]);
    assert_f64_eq!(tf.numerator[0], 1.0);
    assert_f64_eq!(tf.denominator[1], 1.0);
    assert_f64_eq!(tf.denominator[0], 0.0);
}

#[test]
fn tf_as_monic() {
    let tf = TransferFunction::new([2], [2, 0]);
    let monic_tf = as_monic(&tf);
    assert_eq!(monic_tf.numerator[0], 1);
    assert_eq!(monic_tf.denominator[1], 1);
    assert_eq!(monic_tf.denominator[0], 0);
}

#[test]
fn monic_tf_as_monic() {
    let tf = TransferFunction::new([1.0, 1.0], [1.0, 0.0]);
    let monic_tf = as_monic(&tf);
    assert_f64_eq!(monic_tf.numerator[0], 1.0);
    assert_f64_eq!(monic_tf.numerator[1], 1.0);
    assert_f64_eq!(monic_tf.denominator[1], 1.0);
    assert_f64_eq!(monic_tf.denominator[0], 0.0);
}

#[test]
fn test_lhp() {
    let tf = TransferFunction::new([1.0, 1.0], [1.0, 1.0]);
    assert!(lhp(&tf), "TF is not LHP stable");
}
