# Control-rs

[![Rust](https://github.com/Mitchelldscott/control-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/Mitchelldscott/control-rs/actions/workflows/rust.yml)

`control-rs` is a native Rust library for numerical modeling and synthesis.
It is intended for developing robotics, autonomous vehicles, UAVs and other real-time embedded
systems that rely on advanced control algorithms.

### Long-term goals

1. **Components & Design Toolboxes** for Robotics
    - Common robot components (i.e., ESCs, BMS and odometry systems).
    - System level integration and design toolboxes (i.e. Controller tuning, Sensor Calibration, etc.).
2. **Data-Driven Model-Based Design**
    - Tools for model-based design (i.e., Model-Based Optimization, Verification & Validation,
      System Identification and Control Synthesis).
3. **Simulation Environments** for Robotics
    - ODE solvers for continuous and discrete time systems.
    - Monte-Carlo methods for model randomization and random event selection.
    - Emulator-based Software-in-the-loop (SITL) for release build verification.
    - Hardware-in-the-loop (HITL) for functional verification.

This list covers a few projects that are in the works:

- DC Motor lead-lag compensator
- Brushless DC Electronic Speed Control (FOC or fancy 6-stage commuter)
- Lithium Polymer Battery model adaptive estimator
- Quadcopter attitude/altitude controller (3-loop autopilot)
- Visual-Inertial Odometry

## Model Types

* `Polynomial` - Dense univariate polynomial
* `TransferFunction` - Intended to be a laplace domain input/output model but could potentially be
  used as a rational function
* `StateSpace` - Standard linear-algebra representation for a system of differential equations
* `NLModel` - A trait for custom models that provides a more flexible structure/implementation

## Analysis Tools

* `FrequencyTools` - Classical frequency-response methods
* `RobustTools` - hopefully coming soon

## Synthesis Tools

* `LeastSquares` - a trait that's still in the works (should be available for statically sized 
  models).
* `GradientDescent` - also in the works but should provide a trait to perform backpropagation of
  error on models.

## Simulation Tools

* `integrators` - Various types of integration for precision simulations (RK4 + Dormand-Prince)
* `response` - Classic system response implementations: step, ramp, sine, impulse...

# Getting Started

## Installation (Not Supported... haven't published the crate yet, clone this repo instead)

Add this to your `Cargo.toml`:

```toml
[dependencies]
control-rs = "0.0.0"
```

or run

```bash
cargo add control-rs
```

## Example

Here's a simple example to get you started:

```rust
use control_rs::{
  transfer_function::{TransferFunction, utils::{dc_gain, tf2ss}},
  state_space::{StateSpace, utils::zoh},
  math::systems::DynamicalSystem,
};
fn run_sim() {
    // transfer function 1 / s(s + 0.1)
    let mut tf = TransferFunction::new([1.0], [1.0, 0.1, 0.0]);
    println!("{tf}");
    println!("DC Gain of TF: {}", dc_gain(&tf));
    let ss = zoh(&tf2ss(&tf), 0.1);
    println!("{ss}");
    let mut x = nalgebra::Vector2::new(0.0, 0.0);
    for i in 0..100 {
        // x_k+1 = F*x_k + G*u_k
        x = ss.dynamics(x, 1.0);
    }
}
```

# Testing

## Doc Tests

Doc tests will provide useful examples and a basic functionality check.

* **Run all doc tests**: `cargo test --doc`
* **Run specific doc test**: `cargo test --doc <test_name>`
* **List all doc tests**: `cargo test --doc -- --list`

## Unit Tests

* Located within the module containing the unit being tested (e.g., `src/<module>/tests/`).
* Test files will be named after the specific unit they are testing.
* Each test module should contain a description of the tests.

* **Run unit tests**: `cargo test <module>::tests::<unit_test_file_name>`

## Integration Tests

* Located in the `tests/` directory.
* Designed to verify high-level functionality across the tools.
  * Example: Confirm that a simple model maintains consistency across different model formats.

# Documentation

The documentation provides theoretical references and specific user guidance. Each module should include:

* A conceptual description of the module (similar to [MathWorks TransferFunction docs](https://www.mathworks.com/help/control/ug/transfer-functions.html))
* Links to theoretical references for more in-depth understanding
* Example of how to use the module

# Book

In the future, it would be great to have a book that walks through a series of smaller projects that lead to a larger 
one. For example, designing analytical models for a few sensors and actuators and integrating them to design an 
estimator and controller for an RC car or Quadcopter.

# Contributing

We welcome contributions! If you are a control theory student/enthusiast doing a project leave an issue on git with the 
functions you need so `control-rs` can be your one-stop-shop solution for design and implementation.

# License

This project should have a license, but vscode kept complaining. I'll bring it back soon.

Thank you for using `control-rs`!

# Acknowledgements

This project is heavily inspired by the `MATLab Control Systems Toolbox`, many of the functions were written to have the
same call signatures. Also, the core models are almost an exact copy of `nalgebra`'s matrix and many of the trait bounds 
wouldn't be possible without their crate.

# References
* [Feedback control of Dynamic Systems](https://mrce.in/ebooks/Feedback%20Control%20of%20Dynamic%20Systems%208th%20Ed.pdf)