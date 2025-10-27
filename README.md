# Control-rs

[![Rust](https://github.com/Mitchelldscott/control-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/Mitchelldscott/control-rs/actions/workflows/rust.yml)

`control-rs` is a native Rust library for numerical modeling and synthesis of embedded systems. It leverages Rust's 
high-level safety and low-level control to provide a powerful and flexible, open-source toolbox for model-based design 
and implementation. `control-rs` is meant for developing robotics, autonomous vehicles, UAVs and other real-time 
embedded systems that rely on advance control algorithms. Unlike other MBD tools `control-rs` is designed specifically 
for developing firmware. This focus allows for more efficient and customized implementations, with the drawback that 
users must be more comfortable with firmware development.

The inspiration for this project comes from my enthusiasm for embedded Rust and interest in realtime system 
identification and control.

The crate is `no_std` by default (but has a `std` feature flag for plotting) and intends to support 
both fixed-point and floating-point numeric types.

This project has two long-term goals:
1. Implementations of components for robotics (i.e., ESCs, BMS and odometry systems). These templates will use the
[awesome embedded rust crates](https://github.com/rust-embedded/awesome-embedded-rust) to provide detailed guides to 
implementing and operating a variety of components that common to robotics.
2. Wrapper crates for specific control system design tools (i.e., autopilot, self-driving or chemical process controls). 
These toolboxes will have specific models, types and routines to help design and deploy more complex control and 
estimation systems.

This list covers a few projects that are in the works:
- [ ] DC Motor lead-lag compensator
- [ ] BLDC ESC (FOC or fancy 6-stage commuter)
- [ ] LiPo Battery model adaptive estimator
- [ ] Quadcopter attitude/altitude controller (3-loop autopilot)
- [ ] Visual-Inertial Odometry

# Features
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

fn main() {
    // transfer function 1 / s(s + 0.1)
    let mut tf = TransferFunction::new([1.0], [1.0, 0.1, 0.0]);
    println!("{tf}");
    println!("DC Gain of TF: {}", dc_gain(&tf));
    // convert the tf to a state-space model and discretize the model
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

This project is heavily inspired by the MATLab Control Systems Toolbox, many of the functions were written to have the
same call signatures. Also, the core models are almost an exact copy of `nalgebra`'s matrix and many of the trait bounds 
wouldn't be possible without their crate.

# References
* [Feedback control of Dynamic Systems](https://mrce.in/ebooks/Feedback%20Control%20of%20Dynamic%20Systems%208th%20Ed.pdf)