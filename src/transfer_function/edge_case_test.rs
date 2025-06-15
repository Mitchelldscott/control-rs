use crate::{assert_f64_eq, transfer_function::*};

const IMPROPER_TF: TransferFunction<f64, 4, 3> =
    TransferFunction::new([1.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0]);

const DIFFERENTIATOR_TF: TransferFunction<f64, 2, 1> = TransferFunction::new([1.0, 0.0], [1.0]);

const INTEGRATOR_TF: TransferFunction<f64, 1, 2> = TransferFunction::new([1.0], [1.0, 0.0]);

const DELAY_TF: TransferFunction<f64, 2, 2> = TransferFunction::new([1.0, -1.0], [1.0, 1.0]);

const MARGINALLY_STABLE_TF: TransferFunction<f64, 1, 3> =
    TransferFunction::new([1.0], [1.0, 0.0, 1.0]);

const CRITICALLY_DAMPED_TF: TransferFunction<f64, 1, 3> =
    TransferFunction::new([1.0], [1.0, 2.0, 1.0]);

const UNSTABLE_TF: TransferFunction<f64, 1, 3> = TransferFunction::new([1.0], [1.0, -2.0, 1.0]);

const ZP_CANCEL_TF: TransferFunction<f64, 2, 2> = TransferFunction::new([1.0, -1.0], [1.0, -1.0]);

const HIGH_ORDER_TF: TransferFunction<f64, 20, 20> = TransferFunction::new([1.0; 20], [1.0; 20]);

const LOW_DAMPED_TF: TransferFunction<f64, 1, 3> = TransferFunction::new([1.0], [1.0, 0.1, 1.0]);

#[test]
fn improper_tests() {
    assert_f64_eq!(dc_gain(&IMPROPER_TF), 1.0);
    assert!(lhp(&IMPROPER_TF), "lhp for IMPROPER_TF");
    assert_eq!(
        as_monic(&IMPROPER_TF),
        TransferFunction {
            numerator: [1.0, 2.0, 0.0, 1.0],
            denominator: [1.0, 1.0, 1.0]
        },
        "as_monic for IMPROPER_TF"
    );
}

#[test]
fn differentiator_tests() {
    assert_f64_eq!(dc_gain(&DIFFERENTIATOR_TF), 0.0);
    // assert_eq!(lhp(&DIFFERENTIATOR_TF), false, "lhp for DIFFERENTIATOR_TF");
    assert_eq!(
        as_monic(&DIFFERENTIATOR_TF),
        TransferFunction {
            numerator: [1.0, 0.0],
            denominator: [1.0]
        },
        "as_monic for DIFFERENTIATOR_TF"
    );
}

#[test]
fn integrator_tests() {
    assert_f64_eq!(dc_gain(&INTEGRATOR_TF), f64::INFINITY);
    assert!(!lhp(&INTEGRATOR_TF), "lhp for INTEGRATOR_TF");
    assert_eq!(
        as_monic(&INTEGRATOR_TF),
        TransferFunction {
            numerator: [1.0],
            denominator: [1.0, 0.0]
        },
        "as_monic for INTEGRATOR_TF"
    );
}

#[test]
fn delay_tests() {
    assert_f64_eq!(dc_gain(&DELAY_TF), -1.0);
    assert!(lhp(&DELAY_TF), "lhp for DELAY_TF");
    assert_eq!(
        as_monic(&DELAY_TF),
        TransferFunction {
            numerator: [1.0, -1.0],
            denominator: [1.0, 1.0]
        },
        "as_monic for DELAY_TF"
    );
}

#[test]
fn marginally_stable_tests() {
    assert_f64_eq!(dc_gain(&MARGINALLY_STABLE_TF), 1.0);
    assert!(!lhp(&MARGINALLY_STABLE_TF), "lhp for MARGINALLY_STABLE_TF");
    assert_eq!(
        as_monic(&MARGINALLY_STABLE_TF),
        TransferFunction {
            numerator: [1.0],
            denominator: [1.0, 0.0, 1.0]
        },
        "as_monic for MARGINALLY_STABLE_TF"
    );
}

#[test]
fn critically_damped_tests() {
    assert_f64_eq!(dc_gain(&CRITICALLY_DAMPED_TF), 1.0);
    assert!(lhp(&CRITICALLY_DAMPED_TF), "lhp for CRITICALLY_DAMPED_TF");
    assert_eq!(
        as_monic(&CRITICALLY_DAMPED_TF),
        TransferFunction {
            numerator: [1.0],
            denominator: [1.0, 2.0, 1.0]
        },
        "as_monic for CRITICALLY_DAMPED_TF"
    );
}

#[test]
fn unstable_tests() {
    assert_f64_eq!(dc_gain(&UNSTABLE_TF), 1.0);
    assert!(!lhp(&UNSTABLE_TF), "lhp for UNSTABLE_TF");
    assert_eq!(
        as_monic(&UNSTABLE_TF),
        TransferFunction {
            numerator: [1.0],
            denominator: [1.0, -2.0, 1.0]
        },
        "as_monic for UNSTABLE_TF"
    );
}

#[test]
fn zp_cancel_tests() {
    assert_f64_eq!(dc_gain(&ZP_CANCEL_TF), 1.0);
    assert!(!lhp(&ZP_CANCEL_TF), "lhp for ZP_CANCEL_TF");
    assert_eq!(
        as_monic(&ZP_CANCEL_TF),
        TransferFunction {
            numerator: [1.0, -1.0],
            denominator: [1.0, -1.0]
        },
        "as_monic for ZP_CANCEL_TF"
    );
}

#[test]
fn high_order_tests() {
    assert_f64_eq!(dc_gain(&HIGH_ORDER_TF), 1.0);
    assert!(!lhp(&HIGH_ORDER_TF), "lhp for HIGH_ORDER_TF");
    assert_eq!(
        as_monic(&HIGH_ORDER_TF),
        TransferFunction {
            numerator: [1.0; 20],
            denominator: [1.0; 20]
        },
        "as_monic for HIGH_ORDER_TF"
    );
}

#[test]
fn poorly_damped_tests() {
    assert_f64_eq!(dc_gain(&LOW_DAMPED_TF), 1.0);
    assert!(lhp(&LOW_DAMPED_TF), "lhp for LOW_DAMPED_TF");
    assert_eq!(
        as_monic(&LOW_DAMPED_TF),
        TransferFunction {
            numerator: [1.0],
            denominator: [1.0, 0.1, 1.0]
        },
        "as_monic for LOW_DAMPED_TF"
    );
}
