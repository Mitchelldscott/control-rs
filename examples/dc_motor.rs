/// # DC Motor Position Control with a Lead-Lag Compensator
///
/// This example demonstrates the design of a lead-lag compensator for controlling the
/// angular position of a DC motor. The primary goal is to improve the transient
/// and steady-state response of the system based on frequency-domain specifications.
///
/// ## Problem Description
///
/// We model the DC motor as a second-order transfer function, `G(s)`, which relates the
/// input voltage to the angular position of the motor shaft. The open-loop system
/// may exhibit undesirable characteristics such as slow response time, excessive
/// overshoot, or poor steady-state error.
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
/// <pre> P(s) = Km / ((Js + b)(Ls + R) + Km^2) (rad/sec / V) </pre>
use control_rs::{
    integrators::runge_kutta4, transfer_function::*,
    math::systems::DynamicalSystem,
};
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

const SIM_STEPS: usize = 100;
type SimData = [(f64, f64, f64, f64); SIM_STEPS];

fn sim<M: DynamicalSystem<MotorInput, MotorState, MotorOutput>>(
    model: M,
    dt: f64,
    mut x: MotorState,
) -> SimData {
    let mut sim = [(0.0, x[0], x[1], model.output(x, 1.0)[(0, 0)]); SIM_STEPS];
    for i in 1..SIM_STEPS {
        x = runge_kutta4(&model, x, 1.0, 0.0, 0.1, 0.01);
        sim[i] = (i as f64 * dt, x[0], x[1], model.output(x, 1.0)[(0, 0)]);
    }

    sim
}

#[cfg(feature = "std")]
fn plot(sim: SimData) {
    use plotly::{Layout, Plot, Scatter};

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
    let motor_tf = TransferFunction::new([Km], [J * L, J * R + L * b, (R * b) + (Km * Km)]);

    println!("DC Motor {motor_tf}");
    println!("DC Gain: {:?}", dc_gain(&motor_tf));
    println!("System Poles: {:?}", poles(&motor_tf).ok());

    // simulate for 100 steps
    let _sim_data = sim(tf2ss(motor_tf), 0.1, MotorState::new(0.0, 0.0));

    #[cfg(feature = "std")]
    plot(sim_data);

    // Simulates adding a simple feedforward controller that scales the input by the inverse of the
    // dc_gain, resulting in a new dc_gain = 1. In reality, this drives the motor state to the value
    // of the input voltage. An additional gain can scale the output value to an appropriate speed.
    let compensated_motor_tf = TransferFunction::new(
        [Km / dc_gain(&motor_tf)],
        [J * L, J * R + L * b, (R * b) + (Km * Km)],
    );

    let compensated_motor_ss = tf2ss(compensated_motor_tf);
    println!("DC Motor with gain compensation {compensated_motor_ss}");

    // simulate for 100 steps
    let _compensated_sim_data = sim(compensated_motor_ss, 0.1, MotorState::new(0.0, 0.0));

    #[cfg(feature = "std")]
    plot(_compensated_sim_data);

    // let combined_tf = feedback(motor_tf, 1.0);

    // println!("CL system {motor_tf}");
    // println!("CL system gain: {:?}", dc_gain(&motor_tf));
    // println!("CL system Poles: {:?}", poles(&motor_tf));
}
