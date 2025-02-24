use crate::transfer_function::*;

#[cfg(feature = "std")]
use std::fmt;

#[cfg(not(feature = "std"))]
use core::fmt;

use nalgebra::{U1, U2, U3, U4, U20};

fn validate_tf<
    T: fmt::Debug + Copy + RealField + Float,
    D1: Dim,
    D2: Dim,
    S1,
    S2,
    const N: usize,
    const M: usize,
>(
    name: &str,
    tf: TransferFunction<T, D1, D2, S1, S2>,
    expected_gain: T,
    expected_lhp: bool,
    expected_as_monic: ([T; N], [T; M]),
) {
    assert_eq!(dcgain(&tf), expected_gain, "dcgain for {name}");
    assert_eq!(lhp(&tf), expected_lhp, "lhp for {name}");
    assert_eq!(as_monic(&tf), expected_as_monic, "as_monic for {name}");
}

#[test]
fn improper_tests() {
    validate_tf(
        "improper",
        TransferFunction::<f64, U4, U3, _, _>::new([1.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        1.0,
        true,
        ([1.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
    );
}

#[test]
fn differentiator_tests() {
    validate_tf(
        "differentiator",
        TransferFunction::<f64, U2, U1, _, _>::new([1.0, 0.0], [1.0]),
        1.0,
        true,
        ([1.0, 0.0], [1.0]),
    );
}

#[test]
fn integrator_tests() {
    validate_tf(
        "integrator",
        TransferFunction::<f64, U1, U2, _, _>::new([1.0], [1.0, 0.0]),
        1.0,
        false,
        ([1.0], [1.0, 0.0]),
    );
}

#[test]
fn delay_tests() {
    validate_tf(
        "delay",
        TransferFunction::<f64, U2, U2, _, _>::new([1.0], [1.0, 0.0, 1.0]),
        1.0,
        true,
        ([1.0], [1.0, 0.0, 1.0]),
    );
}

#[test]
fn marginally_stable_tests() {
    validate_tf(
        "marginally_stable",
        TransferFunction::<f64, U1, U3, _, _>::new([1.0], [1.0, 0.0]),
        1.0,
        false,
        ([1.0], [1.0, 0.0, 1.0]),
    );
}

#[test]
fn critically_damped_tests() {
    validate_tf(
        "criticaly_damped",
        TransferFunction::<f64, U1, U3, _, _>::new([1.0], [1.0, 2.0, 1.0]),
        1.0,
        true,
        ([1.0], [1.0, 2.0, 1.0]),
    );
}

#[test]
fn unstable_tests() {
    validate_tf(
        "unstable",
        TransferFunction::<f64, U1, U3, _, _>::new([1.0], [1.0, -2.0, 1.0]),
        1.0,
        false,
        ([1.0], [1.0, -2.0, 1.0]),
    );
}

#[test]
fn zp_cancel_tests() {
    validate_tf(
        "zp_cancel",
        TransferFunction::<f64, U2, U2, _, _>::new([1.0, -1.0], [1.0, -1.0]),
        1.0,
        false,
        ([1.0, -1.0], [1.0, -1.0]),
    );
}

#[test]
fn high_order_tests() {
    validate_tf(
        "high_order",
        TransferFunction::<f64, U20, U20, _, _>::new([1.0; 20], [1.0; 20]),
        1.0,
        false,
        ([1.0; 20], [1.0; 20]),
    );
}

#[test]
fn poorly_damped_tests() {
    validate_tf(
        "poorly_damped",
        TransferFunction::<f64, U1, U3, _, _>::new([1.0], [1.0, 0.1, 1.0]),
        1.0,
        true,
        ([1.0], [1.0, 0.1, 1.0]),
    );
}