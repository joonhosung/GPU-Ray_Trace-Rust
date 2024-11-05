<!-- omit in toc -->
# Project Proposal

Project proposal for ECE1724 - Performant Software Systems with Rust

Team members: Jackson Nie (1005282409) Jun Ho Sung (1004793262)

<!-- omit in toc -->
## Table of Contents
- [Motivation](#motivation)
  - [Performance Testing](#performance-testing)
- [Objectives](#objectives)
  - [Primary Objectives](#primary-objectives)
  - [Secondary Objectives](#secondary-objectives)
- [Key features](#key-features)
  - [Primary Features](#primary-features)
  - [Secondary Features](#secondary-features)
- [Tentative plan](#tentative-plan)
  - [Project Timeline](#project-timeline)
    - [Week 1 (11/04 - 11/10) - Performance Analysis \& GPU Experimentation](#week-1-1104---1110---performance-analysis--gpu-experimentation)
    - [Week 2 (11/11 - 11/17) - Core Optimization Implementation](#week-2-1111---1117---core-optimization-implementation)
    - [Week 3 (11/18 - 11/24) - Performance Optimization \& Animation](#week-3-1118---1124---performance-optimization--animation)
    - [Week 4 (11/25 - 12/01) - Feature Expansion](#week-4-1125---1201---feature-expansion)
    - [Week 5 (12/02 - 12/08) - UI \& Rendering Enhancement](#week-5-1202---1208---ui--rendering-enhancement)
    - [Week 6 (12/09 - 12/15) - Documentation \& Demo](#week-6-1209---1215---documentation--demo)

## Motivation
Our project combines our passion for gaming and computer graphics with an opportunity to enhance an existing ray tracer. Ray tracing technology, which creates photorealistic images by simulating light-object interactions, is increasingly crucial in modern graphics applications. While the current implementation delivers high-quality output through multithreading and kd-tree acceleration, performance, and a lack of features remains a key challenge.

This motivated us to propose an endeavor that aims to accelerate the ray-tracing process leveraging modern GPUs. Both team members have experience in GPU programming with C++, but neither has experience implementing GPU solutions in Rust. This project therefore offers an ideal intersection of learning and practical application. As a language that promises "blazingly fast" performance while maintaining memory safety, Rust presents an excellent platform for high-performance computing tasks like ray tracing. By accelerating a computationally intensive graphics application, we will gain hands-on experience and learn the techniques of writing high-performance applications in Rust. This aligns perfectly with the course's goal of developing performant and scalable systems.

This project currently has one major algorithmic optimization in the form of K-D trees. By subdividing the bounding box of objects within a scene, the program is able minimize ray tracing in areas that can be confirmed to not have any interecting objects. However, from our testing, we found that increasing the K-D tree depth can lead to longer rendering times for simpler scenes, likely due to the size of the tree exploding unnecessarily as the depth increases. This means that even though the bounding box volume of the objects within the scene decrease, the time required to access all the nodes of the tree will increase. Therefore, we wanted to explore ways to further optimize this project by implementing other software optimization techniques on the CPU.

Beyond pure performance optimization, we aim to expand the ray tracer's capabilities by implementing new features such as short scene generation and an interactive, visual user interface. These additions will make the project more engaging, and this combination of optimization and feature development will also provide additional challenges in maintaining performance at scale.

To summarize, through this project, we expect to:
* Significantly improve ray tracing performance through GPU acceleration
* Gain practical experience with Rust in high-performance computing
* Implement new features that showcase intriguing capabilities of the ray tracer

### Performance Testing
To establish baseline performance metrics and justify our optimization efforts, we conducted rendering tests across two distinct scenes using varying parameters. Tests were performed on both Windows 11 and MacOS systems to evaluate cross-platform performance. These benchmarks will serve as comparison points for measuring improvements after our optimizations.

Test Scenes:

* Wada: A relatively simple scene containing 8 wada basin spheres
* Biplane: A complex scene depicting a plane over snowy terrain, composed of 267,958 triangles

Key Findings:
* Low iteration or depth parameters frequently resulted in grainy output
* Pure ray-tracing requires each light ray to interact with every object point for smooth rendering, resulting in extensive computing
* Rendering times were prohibitive:
  * Biplane base case (kd-tree depth 2): ~995 seconds per ray-trace iteration (~115 days for 10,000 iterations)
  * Biplane with deep kd-tree optimization (depth 17): ~8 hours for 10,000 iterations
These results clearly demonstrate the need for significant performance optimization. The current rendering times, particularly for complex scenes like the biplane, are impractical for real-world applications. While kd-tree acceleration provides some improvement, additional optimization strategies are essential for achieving usable performance.

| Test Case | Samples/Pixel | Assured Depth | KD Tree Depth | Jun Ho's Time | Jackson's Time |
|-----------|---------------|---------------|---------------|---------------|----------------|
| wada | 100 | 1 | 2 | 21.9s | 18.1s |
| wada | 100 | 1 | 8 | 27.6s | 27.1s |
| wada | 100 | 1 | 17 | 28.4s | 38.3s |
| wada | 100 | 2 | 2 | 25.7s | 29.2s |
| wada | 100 | 5 | 2 | 41.0s | 39.0s |
| wada | 1000 | 1 | 2 | 191.2s | 202.4s |
| wada | 1000 | 2 | 8 | 372.6s | 341.2s |
| wada | 10000 | 1 | 2 | 1928.6s | 2034.1s |
| wada | 10000 | 5 | 17 | 6914.1s | 8543.6s |
| biplane | 100 | 1 | 2 | DNF (~885s/it) | DNF (~483.64s/it) |
| biplane | 100 | 1 | 8 | 2316.1s | 2093.6s |
| biplane | 100 | 1 | 17 | 328.3s | 409.3s | 
| biplane | 100 | 2 | 2 | DNF (~995s/it) | DNF (~506.52s/it) | 
| biplane | 100 | 5 | 2 | DNF | DNF |
| biplane | 1000 | 1 | 2 | - | - |
| biplane | 1000 | 2 | 8 | - | - |
| biplane | 10000 | 1 | 2 | - | - |
| biplane | 10000 | 5 | 17 | 37628.7s | DNF (4.54s/it) |

Note: DNF = Did Not Finish, "-" = No data available

Machine Specifications:
| Specification | Jun Ho's Machine | Jackson's Machine |
|--------------|------------------|-------------------|
| CPU | AMD Ryzen 5800x | Apple M1 Pro |
| GPU | AMD Radeon RX6800XT | Apple M1 Pro |
| RAM | 32GB 3200MHz | 16GB |
| OS | Windows 11 | Ventura 13.1 |

## Objectives
### Primary Objectives
1. Performance Optimization
   * Port the existing CPU-based ray tracer to utilize GPU acceleration in Rust.
   * Implement parallel processing algorithms for various GPU architectures. Develop a flexible GPU backend that supports multiple architectures through generic GPU computation crates.
   * Explore NVIDIA CUDA-specific implementation. 
   * Conduct thorough performance analysis to:
     * Identify computational bottlenecks in the current implementation.
     * Determine optimal GPU-accelerated algorithms for ray tracing operations.
       * Since the bulk of the operations are simple math applied to a wide range of pixels, we believe that GPU acceleration fits this problem extremely well.
   * Based on the test results, target a minimum 5x speedup over current CPU implementation.
   * Implement comprehensive benchmarking suite to:
     * Compare CPU vs. GPU performance metrics.
     * Identify potential drawbacks/limitations of the CPU/GPU implementations, and analyse their impact on performance.
       * Document these optimization impacts and trade-offs.

2. Animation System Integration
   * Develop a kinematic-based animation system.
   * Implement efficient frame generation pipeline leveraging GPU accelerated algorithms implemented previously.


### Secondary Objectives
1. Animation System Optimization
   * Analyze bottlenecks/slowdowns in the animation pipeline.
   * Identify potential improvements that can be applied to speed up the process.
   * Integrate the animation generation process into the UI to provide real-time render previews.
  
2. Interactive User Interface
   * The current implementation utilizes the command line for generating ray traces.
   * Implement an interactive user interface that allows the user to preview images to trace, along with interactive hyperparameter toggling capabilities.
  
3. Pre-Rasterization Enhancement
   * Design and implement a GPU-accelerated pre-rasterization pass.
   * Integrate rasterization output with the ray tracing system.
   * Evaluate the performance impact of pre-rasterization.
   * Analyse image quality with/without pre-rasterization with the same number of iterations.

## Key features
### Primary Features
1. Performance Optimization
   * GPU Acceleration
     * Ray intersection calculations can be sent off to the GPU for ultra-quick processing.
     * The following crates will be explored to add GPU acceleration:
       * emu - procedural macro uses OpenCL to automatically determine what parts of the provided code can be accelerated with a GPU. As it's automatic, it will be the easiest to use, but will likely not provide as much optimization compared to the latter two options.
       * WebGPU - will need to compartmentalize the parallelizable code into compute shaders to then provide as a shader module for the program.
       * opencl3 - vanilla OpenCL API that can target a wide range of heterogeneous devices.
       * rust-cuda - crate that allows executing CUDA kernels on NVIDIA gpus. Kernel programming interface is extremely similar to C++.
   * CPU Enhancement
     * Further optimization of existing multi-threaded CPU implementation by identifying sub-optimal execution patterns and optimizable performance bottlenecks.

2. Animation System Integration
   * Physics-driven object motion.
   * Smooth camera path interpolation and rotation of object.    
    1. Keyframe
    2. cv::videoio::VideoWriter
    3. Camera movement
    4. Object movement (JUN HO FINISH)

### Secondary Features
1. Animation System Optimization
   * UI Integration and Monitoring
     * Real-time animation parameter controls in the interactive interface.
     * Live performance metrics and profiling data.
     * Frame-by-frame preview and adjustment capabilities.
   * Performance Enhancement
     * Systematic analysis of animation pipeline bottlenecks.
     * Implementation of caching strategies for repeated calculations.
  
2. Interactive User Interface
   * Real-time render and preview capabilities.
     * Preview functionality for image selection, and interactive selection opposed to command-line argument input.
   * Configuration of render settings and parameter adjustment.
     * Visual controls for render setting and parameters.
     * Real-time parameter adjustment without configuration file editing.
     * Intuitive preset management.
   * Performance monitoring
     * Live progress tracking and estimated completion time.
     * Detailed performance metrics denoting time spent in exhaustive regions of calculation and bottlenecks. 
     * Can be interactively toggled on/off.
  
3. Pre-Rasterization Enhancement
   * Quality Improvement Pipeline
     * Implementation of GPU-accelerated pre-rasterization stage.
     * Integration with existing ray tracing pipeline.
     * This will allow the render to generate a clear, serviceable image with much less compute required than a purely ray-traced render. This will produce a high-quality render in a shorter time by taking the best of both of rasterization (fast compute for a complete image) and ray tracing (realistic reflections). 


## Tentative plan
### Project Timeline
Secondary objectives may be deprioritized based on progress with primary goals.
#### Week 1 (11/04 - 11/10) - Performance Analysis & GPU Experimentation

* Performance Profiling
  * Research profiling crates and identify bottlenecks with them (Both)

* GPU Acceleration Framework Evaluation
  * emu implementation (Jun Ho)
  * wgpu implementation (Both)
  * CUDA exploration if needed (Jackson)
  * opencl3 exploration if needed (Both)

#### Week 2 (11/11 - 11/17) - Core Optimization Implementation

* Compute Shader Development (Both)
  * Prioritize on highest-impact bottlenecks identified in profiling
  * Continuous performance benchmarking whilst optimizations are applied
  * Jackson will take on more work here as Jun Ho will work on Keyframe Integration

* Keyframe Integration (Jun Ho)
  * Begin implementation of frame interpolation
  * At this point, the animations won't be rendered, but the object for each frame should be calculated

#### Week 3 (11/18 - 11/24) - Performance Optimization & Animation

* Complete Core GPU Implementation (Both)
  * Target: 50x speedup on Jun Ho's machine
    * Will also experiment on Jackson's machine, but currently unclear how well Macbook GPUs are supported

* Animation Development (Pending speedup goals)
  * Complete keyframe generation (Jun Ho)
  * Implement multi-frame generation
  * Begin VideoWriter module implementation (Jackson)

#### Week 4 (11/25 - 12/01) - Feature Expansion

* Ongoing Optimization Refinement (Both)
  * Optimize any missed/new code regions that are easily optimizable
  * Not the top priority here, more so low-hanging-fruit cleanup
* Complete Video Generation Feature (Both)
* Interactive UI Design Initiation (Jackson)
  * Research crates to use
  * Implement a prototype that can do basic preview and parameter toggling
* Rasterization Implementation (Jun Ho)

#### Week 5 (12/02 - 12/08) - UI & Rendering Enhancement

* UI Finalization (Jackson)
  * All features including preview and parameter toggling should be functional

* Hybrid Rendering System (Jun Ho)
  * Combine rasterization with ray tracing to form final pipeline

#### Week 6 (12/09 - 12/15) - Documentation & Demo
* Further optimizations if applicable (Both)
  * Low priority

* Project Documentation (Both)
  * Document findings made during optimization changes
  * Analyse and summarize all optimizations made and why they were effective/uneffective
  * Describe video generation feature implementation and how it can be used
  * Demonstrate interactive UI and what features it has
  * Show improvements achieved by adding a pre-rasterization step to the pipeline

* Demo Video Creation That Portrays:
  * Speedup achieved from optimizations
  * Animation/video rendering feature
  * Interactive UI (if added)
  * Speedup/quality improvement from pre-rasterization (if added)
  * Jun Ho to create the video, Jackson to provide all results and steps to reproduce
