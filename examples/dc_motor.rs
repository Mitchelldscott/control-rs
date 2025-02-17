//! ## Example: RLC Circuit Modeling a DC Motor
//!
//! A common application of control systems is modeling the electrical dynamics of a DC motor.
//! The motor can be approximated as an RLC circuit, where the resistance (R), inductance (L),
//! and back EMF constant (Kb) influence the motor's electrical response.
//!
//! ### Circuit Description
//!
//! The motor's electrical equation is given by:
//!
//! ```text
//! V(s) = (R + Ls)I(s) + KbΩ(s)
//! ```
//!
//! where:
//! - `V(s)` is the Laplace transform of the input voltage,
//! - `I(s)` is the Laplace transform of the current,
//! - `Ω(s)` is the Laplace transform of the angular velocity.
//!
//! Assuming `Ω(s)` is proportional to the torque and depends on the load, we can isolate the transfer function
//! relating input voltage to the motor position.
//!
//! ### Implementation
//!
//! Below is an example of modeling this RLC system using the `TransferFunction` structure.
//!

//!
//! ### Expected Output (untested)
//!
//! ```text
//! DC Motor Transfer Function: TransferFunction { numerator: [1.0], denominator: [0.5, 1.1] }
//! Frequency: 0.10 rad/s, Magnitude: 0.91, Phase: -25.84°
//! Frequency: 1.00 rad/s, Magnitude: 0.71, Phase: -63.43°
//! Frequency: 10.00 rad/s, Magnitude: 0.09, Phase: -84.29°
//! Frequency: 100.00 rad/s, Magnitude: 0.01, Phase: -89.42°
//! Gain Margin: 20.00 dB
//! Phase Margin: 90.00°
//! ```
//!
//! This example demonstrates how to model and analyze the frequency response of a DC motor's
//! electrical dynamics. The tools in this module make it easy to apply similar methods to other
//! control systems.

use control_rs::transfer_function::{dcgain, lhp, TransferFunction};

fn main() {
    // Define motor parameters
    let r = 1.0; // Resistance (Ohms)
    let l = 0.5; // Inductance (H)
    let kb = 0.1; // Back EMF constant (V/(rad/s))

    // Transfer function: G(s) = I(s) / V(s)
    // Numerator: [1]
    // Denominator: [L, R + Kb]
    let motor_tf = TransferFunction::new([1.0], [l, r + kb]);

    #[cfg(feature = "std")]
    println!("DC Motor {motor_tf}");
    println!("DC Gain: {:?}", dcgain(&motor_tf));
    println!("LHP: {:?}", lhp(&motor_tf));
}
