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
use nalgebra::{Vector1, Vector2};

// Define motor parameters
const J: f64 = 0.01;
#[allow(non_upper_case_globals)]
const b: f64 = 0.1;
#[allow(non_upper_case_globals)]
const Km: f64 = 0.01;
const R: f64 = 1.0;
const L: f64 = 0.5;

type MotorInput = f64;
type MotorState = Vector2<f64>;
type MotorOutput = Vector1<f64>;

const SIMSTEPS: usize = 100;
type SimData = [(f64, f64, f64, f64); SIMSTEPS];

fn sim<M: DynamicModel<f64, MotorInput, MotorState, MotorOutput>>(model: M, dt: f64, mut x: MotorState) -> SimData {
    let mut sim = [(0.0, x[0], x[1], model.h(x, 1.0)[(0, 0)]); SIMSTEPS];
    for i in 1..SIMSTEPS {
        x = model.rk4(0.01, 0.0, 0.1, x, 1.0);
        sim[i] = (i as f64 * dt, x[0], x[1], model.h(x, 1.0)[(0, 0)]);
    }

    sim
}

#[cfg(feature="std")]
fn plot(sim: SimData) {
    use plotly::{Plot, Scatter, Layout};

    // Extract time and state values
    let time: Vec<f64> = sim.iter().map(|(t, _, _, _)| *t).collect();
    let state1: Vec<f64> = sim.iter().map(|(_, x, _, _)| *x).collect();
    let state2: Vec<f64> = sim.iter().map(|(_, _, x, _)| *x).collect();
    let output: Vec<f64> = sim.iter().map(|(_, _, _, x)| *x).collect();
    
    // Create plot traces
    let trace1 = Scatter::new(time.clone(), state1).name("State 1");
    let trace2 = Scatter::new(time.clone(), state2).name("State 2");
    let trace3 = Scatter::new(time, output).name("Output");
    
    // Create subplots
    let mut plot = Plot::new();
    let layout = Layout::new();
    
    plot.set_layout(layout);
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    
    plot.show();
}

fn main() {
    // Numerator: [Km]
    // Denominator: JLs^2 + (JR + bL)s + bR + Km^2
    // let motor_tf = TransferFunction::new([Km], [J * L, (J * R + L * b), (R * b) + (Km * Km)]);
    let motor_tf = TransferFunction::new([10.0 * Km], [J * L, (J * R + L * b), (R * b) + (Km * Km)]);

    println!("DC Motor {motor_tf}");
    println!("DC Gain: {:?}", dcgain(&motor_tf));
    println!("System Poles: {:?}", poles(&motor_tf));

    let (num, den) = as_monic(&motor_tf);
    let motor_ss: StateSpace<f64, 2, 1, 1> = control_canonical(num, den);

    println!("DC Motor {motor_ss}");

    // simulate for 100 steps
    let sim_data = sim(motor_ss, 0.1, MotorState::new(0.0, 0.0));

    #[cfg(feature="std")]
    plot(sim_data);

    // let combined_tf = feedback(motor_tf, 1.0);

    // println!("CL system {motor_tf}");
    // println!("CL system gain: {:?}", dcgain(&motor_tf));
    // println!("CL system Poles: {:?}", poles(&motor_tf));
}
