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
use control_rs::{
    frequency_tools::*,
    // integrators::runge_kutta4,
    math::systems::feedback, // DynamicalSystem},
    transfer_function::*,
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

// fn step<ModelInput, ModelState: Clone + Default, ModelOutput, G: DynamicalSystem<ModelInput, ModelState, ModelOutput>>(
//     plant: G,
//     dt: f64,
// ) -> SimData {
//     let mut x = ModelState::default();
//     let mut sim = [(0.0, x[0], x[1], plant.output(x.clone(), 1.0)[(0, 0)]); SIM_STEPS];
//     for i in 1..SIM_STEPS {
//         x = runge_kutta4(&plant, x.clone(), 1.0, 0.0, 0.1, 0.01);
//         sim[i] = (i as f64 * dt, x[0], x[1], plant.output(x.clone(), 1.0)[(0, 0)]);
//     }
//
//     sim
// }

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

    let mut fr = FrequencyResponse::<f64, 100, 1, 1>::logspace([-10.0], [5.0]);
    motor_tf.frequency_response(&mut fr);
    #[cfg(feature = "std")]
    std::fs::create_dir_all("target/plots").expect("Failed to creat plots directory");
    #[cfg(feature = "std")]
    bode("DC Motor Transfer Function", &motor_tf, fr)
        .write_html("target/plots/dc_motor_ol_bode.html");

    // Simulates adding a simple feedforward controller that scales the input by the inverse of the
    // dc_gain, resulting in a new dc_gain = 1. In reality, this drives the motor state to the value
    // of the input voltage. An additional gain can scale the output value to an appropriate speed.
    let gain_compensated_tf = motor_tf / dc_gain(&motor_tf);
    assert_eq!(dc_gain(&gain_compensated_tf), 1.0);

    #[cfg(feature = "std")]
    let fr = FrequencyResponse::<f64, 100, 1, 1>::logspace([-10.0], [5.0]);
    #[cfg(feature = "std")]
    bode(
        "DC Motor gain compensated Transfer Function",
        &gain_compensated_tf,
        fr,
    )
    .write_html("target/plots/dc_motor_gain_compensated_bode.html");

    #[allow(non_snake_case)]
    let Td = 0.4; // TODO: use poles + fr to compute these
    let alpha = 0.2;
    let compensator_tf = TransferFunction::new([Td, 1.0], [alpha * Td, 1.0]);
    let lead_compensated_tf = gain_compensated_tf * compensator_tf;
    println!("Compensated System Zeros: {:?}", zeros(&lead_compensated_tf).ok());
    println!("Compensated System Poles: {:?}", poles(&lead_compensated_tf).ok());

    #[cfg(feature = "std")]
    let fr = FrequencyResponse::<f64, 100, 1, 1>::logspace([-10.0], [5.0]);
    #[cfg(feature = "std")]
    bode(
        "DC Motor lead compensated Transfer Function",
        &lead_compensated_tf,
        fr,
    )
    .write_html("target/plots/dc_motor_lead_compensated_bode.html");

    // #[cfg(feature = "std")]
    let cl_motor_tf = feedback(&lead_compensated_tf, &1.0, 1.0, -1.0);
    // println!("Closed loop tf: {cl_motor_tf}");
    // println!("Closed loop poles: {:?}", poles(&cl_motor_tf).ok()); // hanging
    // #[cfg(feature = "std")]
    // plot(step(tf2ss(&cl_motor_tf), 0.1)); // simulation needs to know the state-shape of cl system
}
