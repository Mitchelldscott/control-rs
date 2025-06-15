# Control-rs

`control-rs` is a Rust library for control system modeling and design, built for real-time embedded applications. It 
offers a familiar interface, inspired by MATLAB’s Control System Toolbox, while embracing Rust’s compile-time safety 
guarantees and memory model. All data structures are statically sized, with dimensionality and type constraints 
enforced at compile time—eliminating the need for heap allocation and enabling deterministic behavior. 

The crate is `no_std` compatible and supports both fixed-point and floating-point numeric types, making it suitable 
for deployment on a wide range of microcontrollers. In the future `control-rs` hopes to provide template projects for 
components like motor controllers, battery management systems, and autonomous navigation logic. The goal is to provide 
a reliable and high performance, open-source foundation for embedded control software in robotics and aerospace.

## Features

* **Modeling**: Support for Polynomial, Transfer Function, State-Space, and custom nonlinear structs
* **Analysis**: Tools for classical, modern and robust system analysis
* **Synthesis**: Direct and data-driven methods to create models
* **Simulation**: Easy model integration and data visualization

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

## Usage

Here's a simple example to get you started:

```rust
use control_rs::transfer_function::{TransferFunction, linear_tools::dc_gain};

fn main() {
  // create a transfer function 1 / s
    let mut tf = Transferfunction::new([1.0], [1.0, 0.0]);
    println!("DC Gain of TF: {}", dc_gain(tf));
}
```

```rust
use control_rs::{TransferFunction, transferfunction::linear_tools::tf2ss};

fn main() {
    let tf = TransferFunction::new([2.0, 4.0], [1.0, 1.0, 4.0, 0.0, 0.0]);
    let ss = tf2ss(tf);
}
```

## Examples

Examples are either based on a textbook problem or demo a practical application. This list covers a few examples that
are in the works:
- [ ] DC Motor lead-lag compensator
- [ ] BLDC ESC (FOC or fancy 6-stage commuter)
- [ ] LiPo Battery model adaptive estimator
- [ ] Orbit Determination (EKF, UKF using Nadir pointing pinhole camera and known landmarks)
- [ ] Visual-Inertial Odometry

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

* A concise conceptual description (similar to [MathWorks TransferFunction docs](https://www.mathworks.com/help/control/ug/transfer-functions.html))
* Links to theoretical references
* Examples

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
