use crate::transfer_function::*;

const IMPROPER_TF: TransferFunction<f64, 3, 4> =
    TransferFunction::new([1.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0]);

const DIFFERENTIATOR_TF: TransferFunction<f64, 1, 2> = TransferFunction::new([1.0, 0.0], [1.0]);

const INTEGRATOR_TF: TransferFunction<f64, 2, 1> = TransferFunction::new([1.0], [1.0, 0.0]);

const DELAY_TF: TransferFunction<f64, 2, 2> = TransferFunction::new([1.0, -1.0], [1.0, 1.0]);

const MARGINALLY_STABLE_TF: TransferFunction<f64, 3, 1> =
    TransferFunction::new([1.0], [1.0, 0.0, 1.0]);

const CRITICALLY_DAMPED_TF: TransferFunction<f64, 3, 1> =
    TransferFunction::new([1.0], [1.0, 2.0, 1.0]);

const UNSTABLE_TF: TransferFunction<f64, 3, 1> = TransferFunction::new([1.0], [1.0, -2.0, 1.0]);

const ZP_CANCEL_TF: TransferFunction<f64, 2, 2> =
    TransferFunction::new([1.0, -1.0], [1.0, -1.0]);

const HIGH_ORDER_TF: TransferFunction<f64, 20, 20> =
    TransferFunction::new([1.0; 20], [1.0; 20]);

const LOW_DAMPED_TF: TransferFunction<f64, 3, 1> =
    TransferFunction::new([1.0], [1.0, 0.1, 1.0]);

#[test]
fn improper_tests() {
    assert_eq!(dcgain(&IMPROPER_TF), 1.0, "dcgain for IMPROPER_TF");
    assert_eq!(lhp(&IMPROPER_TF), true, "lhp for IMPROPER_TF");
    assert_eq!(
        as_monic(&IMPROPER_TF),
        ([1.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        "as_monic for IMPROPER_TF"
    );
}

#[test]
fn differentiator_tests() {
    assert_eq!(
        dcgain(&DIFFERENTIATOR_TF),
        0.0,
        "dcgain for DIFFERENTIATOR_TF"
    );
    assert_eq!(lhp(&DIFFERENTIATOR_TF), false, "lhp for DIFFERENTIATOR_TF");
    assert_eq!(
        as_monic(&DIFFERENTIATOR_TF),
        ([1.0, 0.0], [1.0]),
        "as_monic for DIFFERENTIATOR_TF"
    );
}

#[test]
fn integrator_tests() {
    assert_eq!(
        dcgain(&INTEGRATOR_TF),
        f64::INFINITY,
        "dcgain for INTEGRATOR_TF"
    );
    assert_eq!(lhp(&INTEGRATOR_TF), false, "lhp for INTEGRATOR_TF");
    assert_eq!(
        as_monic(&INTEGRATOR_TF),
        ([1.0], [1.0, 0.0]),
        "as_monic for INTEGRATOR_TF"
    );
}

#[test]
fn delay_tests() {
    assert_eq!(dcgain(&DELAY_TF), -1.0, "dcgain for DELAY_TF");
    assert_eq!(lhp(&DELAY_TF), true, "lhp for DELAY_TF");
    assert_eq!(
        as_monic(&DELAY_TF),
        ([1.0, -1.0], [1.0, 1.0]),
        "as_monic for DELAY_TF"
    );
}

#[test]
fn marginally_stable_tests() {
    assert_eq!(
        dcgain(&MARGINALLY_STABLE_TF),
        1.0,
        "dcgain for MARGINALLY_STABLE_TF"
    );
    assert_eq!(
        lhp(&MARGINALLY_STABLE_TF),
        false,
        "lhp for MARGINALLY_STABLE_TF"
    );
    assert_eq!(
        as_monic(&MARGINALLY_STABLE_TF),
        ([1.0], [1.0, 0.0, 1.0]),
        "as_monic for MARGINALLY_STABLE_TF"
    );
}

#[test]
fn critically_damped_tests() {
    assert_eq!(
        dcgain(&CRITICALLY_DAMPED_TF),
        1.0,
        "dcgain for CRITICALLY_DAMPED_TF"
    );
    assert_eq!(
        lhp(&CRITICALLY_DAMPED_TF),
        true,
        "lhp for CRITICALLY_DAMPED_TF"
    );
    assert_eq!(
        as_monic(&CRITICALLY_DAMPED_TF),
        ([1.0], [1.0, 2.0, 1.0]),
        "as_monic for CRITICALLY_DAMPED_TF"
    );
}

#[test]
fn unstable_tests() {
    assert_eq!(dcgain(&UNSTABLE_TF), 1.0, "dcgain for UNSTABLE_TF");
    assert_eq!(lhp(&UNSTABLE_TF), false, "lhp for UNSTABLE_TF");
    assert_eq!(
        as_monic(&UNSTABLE_TF),
        ([1.0], [1.0, -2.0, 1.0]),
        "as_monic for UNSTABLE_TF"
    );
}

#[test]
fn zp_cancel_tests() {
    assert_eq!(dcgain(&ZP_CANCEL_TF), 1.0, "dcgain for ZP_CANCEL_TF");
    assert_eq!(lhp(&ZP_CANCEL_TF), false, "lhp for ZP_CANCEL_TF");
    assert_eq!(
        as_monic(&ZP_CANCEL_TF),
        ([1.0, -1.0], [1.0, -1.0]),
        "as_monic for ZP_CANCEL_TF"
    );
}

#[test]
fn high_order_tests() {
    assert_eq!(dcgain(&HIGH_ORDER_TF), 1.0, "dcgain for HIGH_ORDER_TF");
    assert_eq!(lhp(&HIGH_ORDER_TF), false, "lhp for HIGH_ORDER_TF");
    assert_eq!(
        as_monic(&HIGH_ORDER_TF),
        ([1.0; 20], [1.0; 20]),
        "as_monic for HIGH_ORDER_TF"
    );
}

#[test]
fn poorly_damped_tests() {
    assert_eq!(dcgain(&LOW_DAMPED_TF), 1.0, "dcgain for LOW_DAMPED_TF");
    assert_eq!(lhp(&LOW_DAMPED_TF), true, "lhp for LOW_DAMPED_TF");
    assert_eq!(
        as_monic(&LOW_DAMPED_TF),
        ([1.0], [1.0, 0.1, 1.0]),
        "as_monic for LOW_DAMPED_TF"
    );
}

