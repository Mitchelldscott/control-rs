/// # DC Motor Position Control with a Lead-Lag Compensator
///
/// This example demonstrates the use of a lead-lag compensator for controlling the
/// rotational speed of a DC motor. The primary goal is to improve the transient
/// and steady-state response of the system based on frequency-domain specifications.
///
/// The process is based on examples in *Feedback Control of Dynamic Systems*, Franklin et al.,
/// Ch. 6.7
///
/// ## Problem Description
///
/// We model the DC motor as a second-order transfer function, `G(s)`, which relates the input
/// voltage to the rotational speed of the motor shaft. The open-loop system may exhibit undesirable
/// characteristics such as slow response time, excessive overshoot, or significant steady-state
/// error.
///
/// To enhance the performance, we will design a lead-lag compensator, `C(s)`. The
/// design process involves shaping the open-loop frequency response to meet the
/// following typical specifications:
///
/// * **Phase Margin**: To ensure a stable system with adequate damping.
/// * **Gain Crossover Frequency**: To achieve a desired bandwidth and response speed.
/// * **Steady-State Error**: To minimize the error for a step input, often addressed
///     by adjusting the low-frequency gain.
///
/// This example will walk through the process of defining the motor's transfer
/// function, designing the lead-lag compensator based on frequency-domain analysis
/// (e.g., Bode plots), and simulating the closed-loop step response to verify that
/// the performance criteria are met.
///
/// The final system will consist of the compensator `C(s)` in series with the
/// plant `G(s)` in a standard negative feedback configuration.
///
/// ## Plant
/// <pre> G(s) = Km / ((Js + b)(Ls + R) + Km^2) (rad/sec / V) </pre>
///
/// ## Resources
/// * [DC Motor Control (Lead-Lag)](https://www.mathworks.com/help/sps/ug/dc-motor-control-lead-lag.html)
///
/// TODO: simulator + Lag compensator
pub use super::Scalar;

mod dc_motor_model;
pub use dc_motor_model::{MotorInput, MotorOutput, MotorSS, MotorState, Motor_SS, Motor_TF};

mod lead_lag_compensator;
pub use lead_lag_compensator::{lead_compensator, LeadCompensator};
