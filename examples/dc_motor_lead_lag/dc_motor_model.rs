//! Implementation of a dc motor
//!
//! # Reference:
//! * [Equivalent Circuit Encompassing Electrical and Mechanical Parts of Direct Current Motors](https://ieeexplore.ieee.org/document/9332747)

use super::Scalar;
use control_rs::{StateSpace, TransferFunction};

use nalgebra::{Matrix1x2, Matrix2, Vector1, Vector2};

//  Model Parameters

// Moment of inertia of the rotor [Kgm^2]
const J: Scalar = 0.01;
// Viscous friction of the rotor [Nms/rad]
const B: Scalar = 0.1;
// Motor Constant (back-emf & torque constants are assumed equal) [Nm/A]/[V/rad/s]
const K: Scalar = 0.01;
// Electrical resistance of the armature coil [ohms]
const R: Scalar = 1.0;
// Electrical inductance of the armature coil [H]
const L: Scalar = 0.5;

// The KVL differential equation of the motor's electronics is
//      vin = i*R + L*di/dt + e
// Where Vin is the input voltage, i is the current, and e is the back emf.
//
// The differential equation of the mechanical parts is
//      J*dw/dt = T - B*w
// Where w is the rotational speed [rad/s], T is rotor torque.
//
// The relationship between the two equations is given by
//      e = K*w
//      T = K*i
//
// This example will design a compensator that takes a reference speed wr as input and outputs a
// voltage. The parameters of this compensator will be determined from a frequency response of the
// transfer function G that relates the motor's input voltage Vin and rotor speed w.
//
// The transfer function is defined as G(s) = W(s) / Vin(s)
// 1. Convert dynamics to Laplace domain
//      B*w(t) + J*dw/dt(t) = K*i(t) => W(s)*(B + J*s) = K*I(s)
//      vin(t) = i(t)*R + L*di/dt(t) + K*w(t) => Vin(s) = I(s)*(R + L*s) + K*W(s)
// 2. Substitute I(s) into Vin(s)
//      Vin(s) = ([(B + J*s)(R + L*s) / K] + K) * W(s)
// 3. Rearrange + Simplify
//      Vin(s) / W(s) = [(B + J*s)(R + L*s) + K^2] / K
//      G(s) = K / [(B + J*s)(R + L*s) + K^2]
//      G(s) = K / [J*Ls^2 + (J*R + B*L)s + B*R + K^2]
type MotorTF = TransferFunction<Scalar, 1, 3>;
#[allow(non_upper_case_globals)]
pub const Motor_TF: MotorTF = TransferFunction::new([K], [J * L, J * R + B * L, (B * R) + (K * K)]);

// This can be converted to a state space model where:
//      x = [i; w], u = vin
//      y = w
//
// Then dx is
// [di; dw] = [vin - R*i - K*w; K*i/J - B*w/J]
pub type MotorInput = Scalar;
// The state of the motor is the
pub type MotorState = Vector2<Scalar>;
pub type MotorOutput = Vector1<Scalar>;
pub type MotorSS = StateSpace<Matrix2<Scalar>, Vector2<Scalar>, Matrix1x2<Scalar>, Vector1<Scalar>>;
#[allow(non_upper_case_globals)]
pub const Motor_SS: MotorSS = StateSpace::new(
    Matrix2::new(-R, -K, K / J, -B / J),
    Vector2::new(1.0, 0.0),
    Matrix1x2::new(0.0, 1.0),
    Vector1::new(0.0),
);
