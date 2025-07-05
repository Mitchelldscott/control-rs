use crate::{
    state_space::*,
    transfer_function::{TransferFunction, tf2ss},
};

#[test]
fn initialize_velocity_statespace() {
    let ss = StateSpace::new(
        nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
        nalgebra::Matrix2x1::new(0.0, 1.0),
        nalgebra::Matrix1x2::new(1.0, 0.0),
        [[0.0]],
    );

    assert_eq!(
        ss.a,
        nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
        "System matrix incorrect"
    );
    assert_eq!(
        ss.b,
        nalgebra::Matrix2x1::new(0.0, 1.0),
        "Input matrix incorrect"
    );
    assert_eq!(
        ss.c,
        nalgebra::Matrix1x2::new(1.0, 0.0),
        "Output matrix incorrect"
    );
}

#[test]
fn velocity_model_zoh_and_stability() {
    let ss = StateSpace::new(
        nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
        nalgebra::Vector2::new(0.0, 1.0),
        nalgebra::Matrix1x2::new(1.0, 0.0),
        nalgebra::Matrix1::new(0.0),
    );

    let ssd = zoh(&ss, 0.1_f32);

    assert_eq!(
        ssd.a,
        nalgebra::Matrix2::new(1.0, 0.099_501_66, 0.0, 0.990_049_84),
        "Discrete System matrix incorrect"
    );

    // check if the eigen values are marginally stable
    if let Some(eigenvalues) = ssd.a.eigenvalues() {
        assert!(
            eigenvalues[0].abs() <= 1.0,
            "unstable eigen value ({}) in F {:}",
            eigenvalues[0],
            ssd.a
        );
        assert!(
            eigenvalues[1].abs() < 1.0,
            "unstable eigen value ({}) in F {:}",
            eigenvalues[0],
            ssd.a
        );
    } else {
        assert!(
            ss.a[(0, 0)] < -1.0,
            "discrete state-space model matrix does not have eigen values"
        );
    }
    // else { panic!("discrete state-space model matrix does not have eigen values") }
}

#[test]
fn control_canonical_test() {
    let tf = TransferFunction::new([2.0, 4.0], [1.0, 1.0, 4.0, 0.0, 0.0]);
    let ss = tf2ss(&tf);

    assert_eq!(
        ss.a,
        nalgebra::Matrix4::from_row_slice(&[
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, -1.0
        ]),
        "System matrix incorrect"
    );
    assert_eq!(
        ss.b,
        nalgebra::Matrix4x1::new(0.0, 0.0, 0.0, 1.0),
        "Input matrix incorrect"
    );
    assert_eq!(
        ss.c,
        nalgebra::Matrix1x4::new(4.0, 2.0, 0.0, 0.0),
        "Output matrix incorrect"
    );
}
