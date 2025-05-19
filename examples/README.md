# Decentralized approach to numerical modelling and synthesis tools for embedded control and estimation systems

## 1. Introduction/Background

### 1.a. Motivation

The growing complexity and requirements of modern robotics demands modular hardware components that are not only robust and adaptable, but also transparent and computationally capable. Many commercial robotic modules, especially in cost-sensitive domains, operate as opaque "black boxes", limiting the ability of engineers and researchers to analyze, customize, or adapt modules for specific use cases. This project proposes the development of a new class of programmable modules—such as motor drivers, sensor hubs, and power monitors—that tightly integrate advanced numerical tools. These components will enable real-time data analysis, on-device model adaptation and decentralized control directly at the hardware level.

### 1.b. Objectives

The key objectives of this project are to develop and demonstrate a new paradigm for robotic hardware modules by focusing on integrated computational capabilities and transparency. Specifically, we aim to achieve the following:

- Develop embedded design tools and middleware specifically engineered for precision numerical control and estimation tasks.
- Provide clear examples and templates for programming common microcontrollers (such as ARM Cortex-M architectures) and illustrate methods for synthesizing control parameters (gains) both online and offline based on real-time sensor feedback.
- Design and build prototype hardware modules that embody these principles, including:
  - A safe battery module providing advanced protection and monitoring for battery systems and connected electronics.
  - Motor drivers capable of high-precision control and enabling detailed modeling of motor dynamics.
  - A positioning module implementing sophisticated sensor fusion techniques, potentially utilizing learned dynamics derived from the motor drivers and battery system.
- Use the designed modules as a stepping stone to building more integrated and specialized robot mother-board.
- Implement the core software as a bare-metal solution to ensure ease of portability across a wide range of different processors and hardware configurations.
- Create a development ecosystem that removes the dependency on monolithic design applications like MATLAB or ROS, which often necessitate massive installations and complex environment setups, thereby streamlining the development workflow.

### 1.c. Scope

The scope of this project is designed to impact robotics development at multiple levels. At the most fundamental level, the core tools and middleware developed will empower advanced engineers, providing them with the capabilities needed to produce highly precise and specialized designs for individual robotic modules. These specialized modules are intended to offer significantly more transparency and functionality compared to existing "black box" solutions, allowing less technical users to gain valuable insight into the module's internal operations. Ultimately, by integrating these sophisticated yet transparent modules, the project aims to create a cohesive hardware and software platform for building specific types of robots – akin to how platforms like PixHawk serve as a dedicated foundation for drone development – making it possible for users with even minimal technical background to configure and work with advanced robotic systems.

### 1.d. Target Components

The project will focus on the development of the following key robotic modules:

- **Numerical synthesis and simulation**: Advanced math tools to analyze and implement control and estimation systems
- **Embedded LCM**: Middleware tool to make communicating between modules reliable, efficient and scalable (i.e. users can ask what varaiables/models a module has before asking for the value, user's may also instruct a module to execute a routine like synthesize the controller gains).
- **Battery Management System (BMS)**: Responsible for the safe and efficient management of the robot's power source.
- **Electronic Speed Controller (ESC)**: Responsible for precisely controlling the motors that actuate the robot's movements.
- **Positioning Module**: Responsible for providing accurate and reliable estimates of the robot's position and orientation within its environment.
- **Specialized cargo project templates for specific processors**: These templates will provide user’s with a basic setup to program their own modules using available Cargo toolchains, controls tools and the embedded lcm middleware.
- **Analytical vs Operational interfaces**: Each module should provide an interface for a user to operate it, this should be identical to existing versions (BMS -> I2C/SPI, ESC -> PWM, Position -> UART). Modules will also provide an interface for analyzing/reconfiguring the modules internal values (USB, UART, ETH).
- **Integrated Systems**: Using more advanced processors will open the door to providing systems of integrated modules (i.e. F1-tenth car, 1/5 baja truck, Quadcopter, Aircraft, ...).

These components are fundamental to a wide range of robotics applications, including:

- Mobile Robots: Autonomous navigation, mapping, and exploration.
- Industrial Robots: Precise manipulation, assembly, and material handling.
- Service Robots: Human-robot interaction, assistance, and delivery.
- UAVs: Flight control, stabilization, and payload delivery
- Scientific Measurement/Data Aquisition Systems: Not all application involve actuators, allowing users to monitor and measure physical systems.

## 2. Methods

Each module developed in this project will have a set of internal variables, dynamic models, operational interfaces and analytical interfaces (operational interfaces will be specific to the module while analytical interfaces should be common across modules).

### 2.a. Controls/System ID Toolbox

A core element of this project is the development and integration of an embedded control system toolbox.  An effective toolbox should provide:

- **Numerical Models for simulation and analysis**: The crate will provide a large amount of tools to work with these models. The models implement a similar interface to their MATLAb versions to allow user to transfer some ideas/implementations easily.
  - Functionality will be built around a few types: Polynomial, Transfer Function and StateSpace
  - Each type should have
    - data and model driven synthesis tools
    - analytical tools
    - data filtering/cleaning tools

- **System Identification (SysID)**: Algorithms for estimating mathematical models of blackbox systems from measured data. This includes methods like:
  - Least Squares Estimation
  - Recursive Least Squares
  - Kalman/Information Filtering
  - Feedback Driven Stochastic Gradient Descent

- **Controller/Estimator Design**: Techniques for designing controllers and estimators from identified models, such as:
  - PID Control
  - Lead/Lag Compensator
  - Optimal Control
  - Adaptive modelling
  - Robust analysis/synthesis
  - Model Predictive Control (MPC)
  - Nonlinear stability tools

- **Model Validation**: Tools for assessing the accuracy and reliability of the identified models.

### 2.b. Embedded Light-weight Communications and Marshaling (LCM)

- Manages all variables, models and tasks on a device.
- Provides a ROS topic like interface to each components internal variables and models.
- External device addressing (may require hardware integration).
- Does not transmit data! only packages/unpackages message headers, timestamps and data
- Provides integration for various hardware communications (USB, Unix socket, UART, CAN...)?

### 2.c. Battery Estimation

The BMS will employ advanced algorithms for accurate battery state estimation, including:

- **State of Charge (SOC) Estimation**: Methods such as Coulomb counting, voltage-based estimation, and Kalman filtering will be used.
- **State of Health (SOH) Estimation**: Techniques for assessing battery degradation, such as internal resistance measurement and capacity estimation.
- **Battery Modeling**: Equivalent circuit models and electrochemical model architectures will be investigated to capture the battery's dynamic behavior.

### 2.d. Motor Controller

Each motor controller will utilize an appropriate control algorithm to achieve high-performance motor control.

- recieve a PWM logic signal from the microcontroller
- generate appropriate output signal to drive the motor
- seperate hardware interface for feedback regarding the motor's speed, position and internal variables (motors can be driven with this unconnected)

### 2.e. Multi-Sensor Localization with Uncertainty

The Positioning Module will employ multi-sensor fusion techniques to achieve robust and accurate localization. This involves:

- Sensor Data Acquisition: Integrating data from multiple sensors, such as:
  - Encoders
  - Inertial Measurement Units (IMUs)
  - GPS
  - LiDAR
- Uncertainty Modeling: Representing the uncertainty associated with each sensor measurement.
- Sensor Fusion: Combining the sensor data using algorithms such as:
  - Kalman Filters (LKF, EKF, UKF)
  - Particle Filter
  - Graph-based SLAM
- Map Building: Optionally, the module may incorporate Simultaneous Localization and Mapping (SLAM) techniques.

## 3. Hardware

This project will involve the design and implementation of embedded hardware systems for each component.

### 3.a Battery Management Systems

- [Infineon BMS](https://www.infineon.com/cms/en/product/battery-management-ics/?adgroupid=173025474631&gad_source=1&gbraid=0AAAAADpmf9cMYUeC982yD4M4iqWBaStL-&gclid=Cj0KCQjwtpLABhC7ARIsALBOCVqOwedUNWbeqqT7SX7fN0N-fRtjgjoXqG_I233OYAfsFf6FQlbS2xkaAptCEALw_wcB&gclsrc=aw.ds): offers a wide range of scalable BMS solutions for various applications, focusing on efficiency and safety.
- [STM L9961](https://www.st.com/en/power-management/l9961.html?icmp=tt40241_gl_lnkon_aug2024): is a battery monitoring and protection IC suitable for automotive and industrial battery management systems.
- [DW01A](https://www.xecor.com/blog/dw01a-ic-circuit-datasheet-alternatives-working): is a very common and cost-effective single-cell lithium-ion battery protection IC.

### 3.b Motor Controllers

- **L298N** (STMicroelectronics): A popular and widely available dual full-bridge driver IC capable of controlling two DC motors or one stepper motor. It can handle relatively high voltages and currents, making it suitable for small to medium-sized brushed DC motors.
- **TB6612FNG** (Toshiba): A compact dual DC motor driver IC offering good efficiency and capable of driving two brushed DC motors with moderate current. It's commonly used in robotics platforms due to its small size and ease of use.
- **A4988** (Allegro MicroSystems): A microstepping driver IC for stepper motors. It allows for precise control of stepper motor position by dividing each full step into smaller microsteps, reducing vibration and increasing resolution.
- **DRV8825** (Texas Instruments): Another popular stepper motor driver IC similar to the A4988, offering microstepping capabilities and integrated current regulation. It's often chosen for its higher current handling capacity compared to the A4988.

### 3.c Positioning Module

**Bosch Sensortec IMUs** (e.g., BMI270, BNO055): Compact and high-performance Inertial Measurement Units providing acceleration, angular velocity, and magnetic field data.
**Garmin GPS Modules** (e.g., GPS 18x LVC): Reliable Global Positioning System receivers for outdoor localization.
**Velodyne Puck** LITE/VLP-16: Compact and affordable 3D LiDAR sensors for environmental perception and mapping.
**Intel RealSense D4xx Series**: Depth cameras utilizing structured light or stereo vision for 3D perception and object recognition.
**PixArt Optical Flow Sensors** (e.g., PMW3901): Low-cost sensors for measuring relative motion, useful for stabilization and visual odometry.
**Various RGB Cameras** (e.g., Raspberry Pi Camera Module, ArduCam): Standard color cameras for visual input and image processing.

## 4. Applications

The proposed components and their embedded control system toolbox enable several advanced robotics applications:

### 4.a. Online/Offline Calibration and model Training

Each module provides an analytical interface (USB, UART, ETH, I2C, SPI) to attach sensors that provide full observability. As a result, the BMS can refine its battery model as the battery ages, the ESC can adapt to changes in motor parameters or load conditions and the positioning model can utilize actuator dynamics to improve estimates.

Models may also be produced on a host that is connected to the target module and an external sensor. The user can then implement their system design, verification and deployment using a script on the host. This is safer method to estimate models and then verify the models before synthesizing and deploying a control system to a robot.

### 4.b. System Degradation Monitoring

By tracking the parameters of the learned models, the components can detect and predict system degradation. For instance, the BMS can monitor the increase in battery internal resistance over time, and the ESC can detect changes in motor efficiency. This enables proactive maintenance and prevents unexpected failures and behaviors.

### 4.c. Adaptive Feedback Control

The components can use the learned models to improve the performance of feedback control systems. For example, the Positioning Module can use the ESC's motor model to compensate for actuator dynamics, leading to more accurate and responsive motion control. This also enables the robot to adapt to changing environmental conditions or payloads.

## 5. Market Survey

- [PixHawk Flight Control/Sensor Integration/Battery Management](https://pixhawk.org/)
- **Texas Instruments MSP430 Family**: Some MSP430 microcontrollers offer integrated analog-to-digital converters (ADCs) and timers that can be used for basic motor control applications. While not as feature-rich as dedicated motor driver ICs, they can be suitable for simpler brushed DC motor control with external power stages.
- **NXP Kinetis Family**: Similar to STM32, the Kinetis family offers a wide range of ARM Cortex-M microcontrollers with various peripherals. Some Kinetis devices include integrated motor control timers, PWM generators, and ADCs suitable for more advanced motor control techniques like Field-Oriented Control (FOC) when paired with external drivers.
- **Infineon XMC Family**: These microcontrollers are specifically designed for industrial applications, including motor control. Many XMC devices feature dedicated motor control peripherals like PWM units with dead-time insertion, capture/compare units, and high-resolution ADCs for precise current and voltage sensing.
- **Renesas RA Family**: This family of ARM Cortex-M microcontrollers offers a diverse set of features, with some series including peripherals suitable for motor control, such as high-resolution PWM timers and ADCs.
- **Allegro MicroSystems A89xxx Family**: While primarily known for motor driver ICs, Allegro also offers integrated solutions like the A89224, which combines an ARM Cortex-M4F microcontroller with a 90V BLDC gate driver and current sensing, offering a highly integrated solution for automotive and industrial BLDC applications.
- **T-Motor ESCs with Telemetry**: While Electronic Speed Controllers (ESCs) are typically external to flight controllers, some high-end T-Motor ESCs offer telemetry data (e.g., current, voltage, RPM) that can be fed back to the flight controller. Future integration could see ESCs with more onboard processing for control and system identification.
- **CubePilot Ecosystem**: Building upon the Pixhawk standard, CubePilot offers more rugged and integrated flight control solutions, sometimes incorporating carrier boards with integrated power management and sensor options.
**Dedicated Robot Motor Controller Boards** (e.g., from Pololu, Cytron): These boards often integrate motor drivers (for brushed DC, BLDC, or stepper motors) with microcontrollers (like Arduino or STM32) and sometimes include basic current sensing or encoder interfaces. They simplify motor control for robotic platforms but typically don't have the full suite of sensors and flight control capabilities of a Pixhawk.

## 6. Budget and Resources

## 7. Risk Mitigation
