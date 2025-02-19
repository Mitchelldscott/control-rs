//! ## [Example: RLC Circuit Modeling a DC Motor](https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling)
//!
//! This example creates a transfer function of a DC Motor and uses the toolbox to design a
//! controller.
//!
//! ### Plant
//! P(s) = Km / ((Js + b)(Ls + R) + Km^2) (rad/sec / V)
//!
//! ### Requirements
//!
//! * Settling time less than 2 seconds
//! * Overshoot less than 5%
//! * Steady-state error less than 1%
//!
//! ### Expected Output (untested)
//!
//! ```text
//! DC Motor TransferFunction:
//!     1
//! ----------
//! 0.5s + 1.1
//!
//! DC Gain: 0.9090909090909091
//! LHP: true
//! ```

use control_rs::{transfer_function::*, state_space::*, DynamicModel};

// Define motor parameters
const J: f64 = 0.01;
#[allow(non_upper_case_globals)]
const b: f64 = 0.1;
#[allow(non_upper_case_globals)]
const Km: f64 = 0.01;
const R: f64 = 1.0;
const L: f64 = 0.5;

fn main() {
    // Numerator: [Km]
    // Denominator: JLs^2 + (JR + bL)s + bR + Km^2
    let motor_tf = TransferFunction::new([Km], [J * L, (J * R + L * b), (R * b) + (Km * Km)]);

    println!("DC Motor {motor_tf}");
    println!("DC Gain: {:?}", dcgain(&motor_tf));
    println!("System Poles: {:?}", poles(&motor_tf));

    let (num, den) = as_monic(&motor_tf);
    let motor_ss: StateSpace<f64, 2, 1, 1> = control_canonical(num, den);

    println!("DC Motor {motor_ss}");

    let mut x = nalgebra::Vector2::new(0.0, 0.0);

    // simulate for 100 steps
    let dt = 0.1;
    let steps = 100;
    let mut sim = [(0.0, x); 100];
    for i in 0..100 {
        x = motor_ss.rk4(0.01, 0.0, 0.1, x, 0.0);
        sim[i+1] = (i as f64 * dt, x);
    }



    // let combined_tf = feedback(motor_tf, 1.0);

    // println!("CL system {motor_tf}");
    // println!("CL system gain: {:?}", dcgain(&motor_tf));
    // println!("CL system Poles: {:?}", poles(&motor_tf));
}
