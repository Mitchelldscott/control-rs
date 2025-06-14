# Control-rs

`control-rs` is a numerical modeling and control synthesis library tailored for embedded systems. Inspired by MATLAB’s 
Control System Toolbox, it offers a structured framework for system modeling, analysis, and controller design. 
Leveraging Rust’s ownership model and static type system, Control-rs ensures memory safety, concurrency guarantees, and 
deterministic behavior—making it well-suited for real-time, resource-constrained applications.

## Features

* **Modeling**: Support for Polynomial, Transfer Function, State-Space, and other nonlinear representations
* **Analysis**: Tools for classical, modern and robust system analysis
* **Synthesis**: Direct and data-driven methods to create models
* **Simulation**: Easy model integration and data visualization

## Installation (Not Supported... haven't published the crate yet, clone this repo instead)

Add this to your `Cargo.toml`:

```toml
[dependencies]
control-rs = "0.1.0"
```

or run

```bash
cargo add control-rs
```

## Usage

Here's a simple example to get you started:

```rust
use control_rs::transfer_function::{TransferFunction, dcgain};

fn main() {
    let mut tf = Transferfunction::new([1.0], [1.0, 0.0]);
    println!("DC Gain of TF: {}", dcgain(tf));
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
- [ ] BLDC ESC (FOC)
- [ ] LiPo Battery model adaptive estimator
- [ ] Orbit Determination
- [ ] Visual-Inertial Odometry

## Testing

### Doc Tests

To keep unit tests purposeful and concise, the doc-tests should provide a majority of the code coverage. These will
double as useful examples and a basic functionality check.

### Unit Tests

* Located within the module containing the unit being tested (e.g., `src/<module>/...`).
* Simple tests may reside in mod.rs, while more complex tests will have dedicated files to keep mod.rs concise.
* Each test case should be:
  * Short
  * Independent
  * Clearly descriptive of the test

### Integration Tests

* Located in the `tests/` directory.
* Designed to verify high-level functionality across the tools.
  * Example: Confirm that a simple model maintains consistency across different model formats.

## Documentation

The documentation provides theoretical references and specific user guidance. Each module should include:

* A concise conceptual description (similar to [MathWorks TransferFunction docs](https://www.mathworks.com/help/control/ug/transfer-functions.html))
* Links to theoretical references
* Use cases (verified with doctests)

## Book

In the future, it would be great to have a book that walks through a series of smaller projects that lead to a larger 
one. For example, designing analytical models for a few sensors and actuators and integrating them to design an 
estimator and controller for an RC car or Quadcopter.

## Contributing

We welcome contributions!

## License

This project should have a license, but vscode kept complaining. I'll bring it back soon.

Thank you for using `control-rs`!
