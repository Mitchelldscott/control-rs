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
///   by adjusting the low-frequency gain.
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
use control_rs::{frequency_tools::*, transfer_function::*};

#[cfg(feature = "std")]
use control_rs::{
    integrators::runge_kutta4, math::systems::DynamicalSystem, state_space::utils::zoh,
};

#[cfg(feature = "std")]
use plotly::{Layout, Plot, Scatter};

// A Convenient way to set the precision of the models.
pub type Scalar = f64;

mod dc_motor_lead_lag;
pub use dc_motor_lead_lag::*;

#[cfg(feature = "std")]
const SIM_STEPS: usize = 100;
#[cfg(feature = "std")]
type SimData = (
    [Scalar; SIM_STEPS],
    [MotorInput; SIM_STEPS],
    [MotorState; SIM_STEPS],
);

#[cfg(feature = "std")]
fn step(plant: MotorSS, compensator: LeadCompensator, x0: MotorState, dt: Scalar) -> SimData {
    let mut x = x0;
    let mut x_c = nalgebra::Vector1::zeros();
    let compensator_ss = zoh(&tf2ss(&compensator), dt);
    let mut y = 0.0;
    let mut sim = (
        [0.0; SIM_STEPS],
        [compensator_ss.output(x_c, 1.0)[(0, 0)]; SIM_STEPS],
        [x0; SIM_STEPS],
    );
    for i in 1..SIM_STEPS {
        x_c = compensator_ss.dynamics(x_c, 1.0 - y);
        let u = compensator_ss.output(x_c, 1.0 - y)[(0, 0)];
        x = runge_kutta4(&plant, x, u, 0.0, dt, dt / 10.0);
        y = plant.output(x, u)[(0, 0)];
        sim.0[i] = i as Scalar * dt;
        sim.1[i] = u;
        sim.2[i] = x;
    }

    sim
}

#[cfg(feature = "std")]
fn plot(sim: SimData) -> Plot {
    // Extract time and state values
    let time: Vec<f64> = sim.0.to_vec();
    let voltage: Vec<f64> = sim.1.to_vec();
    let current: Vec<f64> = sim.2.iter().map(|x| x[0]).collect();
    let speed: Vec<f64> = sim.2.iter().map(|x| x[1]).collect();

    // Create plot traces
    let trace0 = Scatter::new(time.clone(), vec![1.0; time.len()]).name("reference");
    let trace1 = Scatter::new(time.clone(), voltage).name("voltage");
    let trace2 = Scatter::new(time.clone(), current).name("Current");
    let trace3 = Scatter::new(time.clone(), speed).name("Speed");

    // Create subplots
    let mut plot = Plot::new();
    let layout = Layout::new();

    plot.set_layout(layout);
    plot.add_trace(trace0);
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);

    plot
}

fn main() {
    println!("DC Motor {Motor_TF}");
    println!("DC Gain: {:?}", dc_gain(&Motor_TF));
    println!("System Poles: {:?}", poles(&Motor_TF).ok());

    let mut fr = FrequencyResponse::<f64, 1, 1, 100>::logspace(-1.0, 5.0);
    Motor_TF.frequency_response(&mut fr);
    #[cfg(feature = "std")]
    std::fs::create_dir_all("target/plots").expect("Failed to creat plots directory");
    #[cfg(feature = "std")]
    bode("DC Motor Transfer Function", &Motor_TF, fr)
        .write_html("target/plots/dc_motor_ol_bode.html");

    // Simulates adding a simple feedforward controller that scales the input by the inverse of the
    // dc_gain, resulting in a new dc_gain = 1. In reality, this drives the motor state to the value
    // of the input voltage. An additional gain can scale the output value to an appropriate speed.
    assert_eq!(dc_gain(&(Motor_TF / dc_gain(&Motor_TF))), 1.0);

    #[allow(non_snake_case)]
    let compensator_tf = lead_compensator(5.0, 60.0) / dc_gain(&Motor_TF);
    let lead_compensated_tf = Motor_TF * compensator_tf;
    println!(
        "Compensated System Zeros: {:?}",
        zeros(&lead_compensated_tf).ok()
    );
    println!(
        "Compensated System Poles: {:?}",
        poles(&lead_compensated_tf).ok()
    );

    #[cfg(feature = "std")]
    let fr = FrequencyResponse::<f64, 1, 1, 100>::logspace(-1.0, 5.0);
    #[cfg(feature = "std")]
    bode(
        "DC Motor lead compensated Transfer Function",
        &lead_compensated_tf,
        fr,
    )
    .write_html("target/plots/dc_motor_lead_compensated_bode.html");

    #[cfg(feature = "std")]
    plot(step(
        Motor_SS,
        compensator_tf / dc_gain(&Motor_TF),
        MotorState::zeros(),
        0.1,
    ))
    .write_html("target/plots/dc_motor_sim.html");
}
