# Ferrocene Compliance and Safety-Critical Practices
## Control Systems Toolbox
### 1. Introduction

This document outlines the software development lifecycle, practices, and procedures for the `control-rs` project. 
The primary goal is to ensure compliance with safety-critical standards by leveraging the Ferrocene toolchain. 
Adherence to these guidelines is mandatory for all contributors to maintain the integrity, reliability, and safety of 
the codebase.

Ferrocene is a qualified version of the Rust compiler toolchain, suitable for use in safety- and mission-critical 
applications. By adopting Ferrocene, we commit to a higher standard of code quality, formal verification, and 
documentation.

### 2. Core Principles
* **Safety First**: The primary concern is preventing failures that could lead to harm. Every decision must be 
weighed against its impact on the system's safety.
* **Explicitness and Clarity**: Code should be clear, well-documented, and easy to understand. Ambiguity is a risk.
* **Verifiability**: All requirements and code must be testable and verifiable. This includes formal methods where 
applicable.
* **Minimalism**: Both the code and its dependencies should be as simple as possible. Unnecessary complexity increases 
the attack surface and the likelihood of bugs.

### 3. Development Practices & Procedures
#### 3.1. Documentation
Comprehensive documentation is non-negotiable. Every part of the toolbox must be documented to ensure it can be 
understood, reviewed, and maintained correctly.
* **Public API Documentation**: All public functions, structs, enums, traits, and modules must have thorough 
documentation using Rustdoc `///`.
  * Explain the purpose of the item.
  * Describe each parameter and the return value.
  * Detail any possible panic! scenarios.
  * Provide usage examples within the documentation using rust blocks:
    ```rust
    // Cargo test will run these examples
    ```
* **Module-Level Documentation**: Each module (mod.rs or the file itself) must start with `#![doc = "..."]` or `//!` 
that explains the module's role and scope.
* **unsafe Code Documentation**: Every unsafe block must be accompanied by a `// Safety:` comment explaining the 
invariants that make the block sound.
* **Generating Docs**: Project documentation is generated and reviewed using `cargo doc --open`.

#### 3.2. Testing
A multi-layered testing strategy is required to ensure correctness from the unit level to the system level.
* **Unit Tests**: Each module should contain a `#[cfg(test)]` section with unit tests that cover all functions and 
types. The unit tests should cover expected and unexpected inputs as well as document why a case belongs in either
category.
* **Integration Tests**: The `/tests` directory will contain integration tests that verify the interaction and 
consistency between different modules of the toolbox.
* **Code Coverage**: We will track test coverage to identify untested code paths. The CI pipeline will enforce a 
minimum coverage threshold. Tools like `grcov` will be used for this purpose, or a custom x-task may be developed.

#### 3.3. Benchmarking
Performance is critical in control systems, especially for real-time applications.
* **Benchmark Suite**: A comprehensive benchmark suite must be maintained in the `/benches` directory.
* **Critical Paths**: All performance-critical algorithms and code paths must have associated benchmarks.
* **Execution Time**: Benchmarks help enforce worst-case execution time analysis. Any regressions in performance must 
be justified and reviewed.

#### 3.4. Examples
Clear examples are essential for demonstrating correct usage and for high-level integration testing.
* **Usage Examples**: The `/examples` directory must contain standalone, runnable examples for each major feature of 
the toolbox.
* **Clarity and Simplicity**: Examples should be well-commented and easy to follow.

#### 3.5. Formal Methods & Verification
For the most safety-critical parts (e.g., core control-loop algorithms, state estimators), we will use formal 
verification techniques.
* **Model Checking**: Tools like `Kani` may be used to mathematically prove properties about the code, such as the 
absence of panics, overflows, or race conditions.
* **Pre- /Post-Conditions**: Where applicable, tools like `Prusti` can be used to verify function contracts 
(pre-conditions and post-conditions).
* **Process**: The adoption of formal verification for a specific component requires a formal proposal and review 
process.

### 4. Code Quality and Dependency Management
#### 4.1. Code Formatting and Linting
* `rustfmt`: All code must be formatted with `rustfmt` using the project's `rustfmt.toml` configuration. This is 
enforced by the CI pipeline.
* `clippy`: A strict set of Clippy lints is enforced. The following attributes should be at the crate root:
```rust
#![deny(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    warnings,
)]
```
#### 4.2. unsafe Code Policy
The use of unsafe is heavily restricted.
* **Avoidance**: unsafe code should be avoided whenever a safe alternative exists.
* **Justification**: Any use of unsafe requires a detailed, written justification that is approved by at least two 
senior developers.
* **Encapsulation**: All unsafe blocks must be wrapped in a safe abstraction layer. The public API of a module 
containing unsafe code should not expose the unsafety to its callers.`// Safety:` Comments: As stated before, every 
unsafe block must be rigorously documented.

#### 4.3. Dependency Management
* **Minimalism**: Only add dependencies that are necessary.
* **Vetting**: All third-party crates must undergo a vetting process that assesses: 
  * Maintainer reputation and activity.
  * Code quality and API stability. 
  * Security advisories (`cargo audit`).
  * License compatibility.
* `cargo-deny`: We use `cargo-deny` to enforce policies on the dependency tree. The `deny.toml` file in the repository 
root defines the rules for licenses, duplicate dependencies, and security advisories.

### 5. Continuous Integration (CI)
The CI pipeline is the gatekeeper for our main branch. A pull request cannot be merged unless all the following checks 
pass:
1. `cargo check`
2. `cargo fmt -- --check`
3. `cargo clippy -- -D warnings`
4. `cargo test --all-targets`
5. `cargo doc` (to ensure documentation builds correctly)
6. `cargo deny check`
7. `cargo audit`
8. (Optional) Run formal verification tools on specified components.

[1] [“FLS — FLS.” Accessed: Jul. 10, 2025. [Online].](https://rust-lang.github.io/fls/)
[2] [“minirust/spec at master · minirust/minirust,” GitHub. Accessed: Jul. 10, 2025. [Online].](https://github.com/minirust/minirust/tree/master/spec)
