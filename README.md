# Control-rs

Rust-native numerical modeling and synthesis library for embedded projects.

The crate is `no_std` by default and supports both fixed-point and floating-point numeric types, making it suitable 
for deployment on a wide range of microcontrollers. The goal is to provide Cargo project templates to enable a reliable 
and high performance, open-source foundation for embedded control software.

The vision is that users can use the templates to either fabricate their own robot components or integrate the 
templates into more complex modules.

This list covers a few projects that are in the works:
- [ ] DC Motor lead-lag compensator
- [ ] BLDC ESC (FOC or fancy 6-stage commuter)
- [ ] LiPo Battery model adaptive estimator
- [ ] Quadcopter attitude/altitude controller
- [ ] Visual-Inertial Odometry
- [ ] Orbit Determination/Visual localization (EKF, UKF using Nadir pointing pinhole camera and known landmarks)

## Features

* **Modeling**: Support for Polynomial, Transfer Function, State-Space, and custom nonlinear structs
* **Analysis**: Tools for classical, modern and robust system analysis
* **Synthesis**: Direct and data-driven methods to construct models
* **Simulation**: Precision model integration (and in the future time-series/episodic dataset tools)

## Getting Started

### Installation (Not Supported... haven't published the crate yet, clone this repo instead)

Add this to your `Cargo.toml`:

```toml
[dependencies]
control-rs = "0.0.0"
```

or run

```bash
cargo add control-rs
```

### Example

Here's a simple example to get you started:

```rust
use control_rs::transfer_function::{TransferFunction, linear_tools::dc_gain};

fn main() {
    let dt = 0.1;
    // create a transfer function 1 / s
    let mut tf = Transferfunction::new([1.0], [1.0, 0.1, 0.0]);
    println!("{tf}");
    println!("DC Gain of TF: {}", dc_gain(tf));
    let ss = zoh(tf2ss(tf), dt);
    println!("{ss}");
    let mut x = nalgebra::Vector2(0.0, 0.0);
    for i in 0..100 {
        x = ss.dynamics(x, 1.0);
    }
}
```

## Testing

### Doc Tests

To keep unit tests purposeful and concise, the doc-tests should provide a majority of the code coverage. These will
double as useful examples and a basic functionality check.

* **Run all doc tests**: `cargo test --doc`
* **Run specific doc test**: `cargo test --doc <test_name>`
* **List all doc tests**: `cargo test --doc -- --list`

### Unit Tests

* Located within the module containing the unit being tested (e.g., `src/<module>/...`).
* test files will be named after the specific unit they are testing and end in `_test.rs`.
* Each test module should contain a description of the tests

* **Run unit tests**: `cargo test <module>::<unit_test_file_name>`

### Integration Tests

* Located in the `tests/` directory.
* Designed to verify high-level functionality across the tools.
  * Example: Confirm that a simple model maintains consistency across different model formats.

## Documentation

The documentation provides theoretical references and specific user guidance. Each module should include:

* A conceptual description of the module (similar to [MathWorks TransferFunction docs](https://www.mathworks.com/help/control/ug/transfer-functions.html))
* Links to theoretical references for more in-depth understanding
* Example of how to use the module

## Project Templates

In addition to providing the foundational blocks for implementing control systems, `control-rs` should have templates
that integrate control systems with crates from the embedded rust community. These templates can then provide users a
starting point for developing their own products.

## Book

In the future, it would be great to have a book that walks through a series of smaller projects that lead to a larger 
one. For example, designing analytical models for a few sensors and actuators and integrating them to design an 
estimator and controller for an RC car or Quadcopter.

## Contributing

We welcome contributions! If you are a controls student/enthusiast doing a project leave an issue on git with the 
functions you need so `control-rs` can be your one-stop-shop solution for design and implementation.

## License

This project should have a license, but vscode kept complaining. I'll bring it back soon.

Thank you for using `control-rs`!

## Acknowledgements

This project is heavily inspired by the MATLab Control Systems Toolbox, many of the functions were written to have the
same call signatures. Also, the core models are almost an exact copy of `nalgebra`'s matrix and many of the trait bounds 
wouldn't be possible without their crate.