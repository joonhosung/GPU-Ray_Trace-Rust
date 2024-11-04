<!-- omit in toc -->
# Project Proposal

Project proposal for ECE1724 - Performant Software Systems with Rust

Team members: Jackson Nie (1005282409) Jun Ho Sung (1004793262)

<!-- omit in toc -->
## Table of Contents
- [Motivation](#motivation)
    -[Testing](#testing)
- [Objectives](#objectives)
  - [Primary Objectives](#primary-objectives)
  - [Secondary Objectives](#secondary-objectives)
- [Key features](#key-features)
  - [Primary Features](#primary-features)
  - [Secondary Features](#secondary-features)
- [Tentative plan](#tentative-plan)

## Motivation
Our project proposal stems from a shared passion for gaming and computer graphics, combined with a fascinating opportunity to enhance an existing ray tracing implementation.

Ray tracing, which creates photorealistic images by simulating light-object interactions, has become increasingly relevant in modern graphics applications, especially in gaming and visual effects. While the current ray-tracer implementation produces high-quality images through multithreading and kd-tree acceleration, its performance remains a significant bottleneck. This limitation presents an exciting opportunity for optimization, particularly through GPU acceleration. 

This motivated us to propose an endeavor that aims to accelerate the ray-tracing process leveraging modern GPUs. Both team members have experience in GPU programming with C++, but neither has experience implementing GPU solutions in Rust. This project therefore offers an ideal intersection of learning and practical application. As a language that promises "blazingly fast" performance while maintaining memory safety, Rust presents an excellent platform for high-performance computing tasks like ray tracing. By accelerating a computationally intensive graphics application, we will gain hands-on experience and learn the techniques of writing high-performance applications in Rust. This aligns perfectly with the course's goal of developing performant and scalable systems.

This project currently has one major algorithmic optimization in the form of K-D trees. By subdividing the bounding box of objects within a scene, the program is able minimize ray tracing in areas that can be confirmed to not have any interecting objects. However, from our testing, we found that increasing the K-D tree depth can lead to longer rendering times, likely due to the size of the tree exploding as the depth increases (talk about how it did improve for more complex scenes). This means that even though the bounding box volume of the objects within the scene decrease, the time required to access all the nodes of the tree will increase. Therefore, we wanted to explore ways to further optimize this project by implementing common software optimization techniqes.

Beyond pure performance optimization, we aim to expand the ray tracer's capabilities by implementing new features such as an interactive, visual user interface as well as short scene generation. These additions will make the project more engaging, and this combination of optimization and feature development will also provide additional challenges in maintaining performance at scale.

To summarize, through this project, we expect to:
* Significantly improve ray tracing performance through GPU acceleration
* Gain practical experience with Rust in high-performance computing
* Implement new features that showcase intriguing capabilities of the ray tracer

### Testing
To get a baseline performance profile of the project at its current state and to further motivate performance optimizations, we rendered two different scenes with varying parameters. 
We used one Windows 11 and one MacOS machine to help see how different systems perform. By the end of the project, we will re-run the same tests after the project to determine what speedup we were able to achieve.

During the test runs, we noticed that the image turned out grainy a lot of the time with low-iteration or low-depth parameters. This was due to the renders being purely ray-traced, meaning that every light 'rays' generated had to hit every single point in the object to create a smooth render, making it highly compute intensive. Due to this effect, we were determined to find any ways that could make the quality of the image higher while keeping the number of computations the same.

| Test Case | Samples/Pixel | Assured Depth | KD Tree Depth | Jun Ho's Time | Jackson's Time |
|-----------|---------------|---------------|---------------|---------------|----------------|
| wada_100_1_2 | 100 | 1 | 2 | 21.9s | 18.1s |
| wada_100_1_8 | 100 | 1 | 8 | 27.6s | 27.1s |
| wada_100_1_17 | 100 | 1 | 17 | 28.4s | 38.3s |
| wada_100_2_2 | 100 | 2 | 2 | 25.7s | 29.2s |
| wada_100_5_2 | 100 | 5 | 2 | 41.0s | 39.0s |
| wada_1000_1_2 | 1000 | 1 | 2 | 191.2s | 202.4s |
| wada_1000_2_8 | 1000 | 2 | 8 | 372.6s | 341.2s |
| wada_10000_1_2 | 10000 | 1 | 2 | 1928.6s | 2034.1s |
| wada_10000_5_17 | 10000 | 5 | 17 | 6914.1s | 8543.6s |
| biplane_100_1_2 | 100 | 1 | 2 | DNF (too long ~885s/it) | - |
| biplane_100_1_8 | 100 | 1 | 8 | 2316.1s | - |
| biplane_100_1_17 | 100 | 1 | 17 | 328.3s | - |
| biplane_100_2_2 | 100 | 2 | 2 | DNF (too long ~995s/it) | - |
| biplane_100_5_2 | 100 | 5 | 2 | DNF | - |
| biplane_1000_1_2 | 1000 | 1 | 2 | DNF (too long ~885s/it) | - |
| biplane_1000_2_8 | 1000 | 2 | 8 | - | - |
| biplane_10000_1_2 | 10000 | 1 | 2 | DNF | - |
| biplane_10000_5_17 | 10000 | 5 | 17 | 37628.7s | - |

Note: DNF = Did Not Finish, "-" = No data available

Machine Specifications:
| Specification | Jun Ho's Machine | Jackson's Machine |
|--------------|------------------|-------------------|
| CPU | AMD Ryzen 5800x | Apple M1 PRO |
| GPU | AMD Radeon RX6800XT | Apple M1 PRO |
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

2. Interactive User Interface
   * The current implementation utilizes the command line for generating ray traces.
   * Implement an interactive user interface that allows the user to preview images to trace, along with interactive hyperparameter toggling capabilities.

3. Animation System Integration
   * Develop a kinematic-based animation system.
   * Implement efficient frame generation pipeline leveraging GPU accelerated algorithms implemented previously.


### Secondary Objectives
1. Animation System Optimization
   * Analyze bottlenecks/slowdowns in the animation pipeline.
   * Identify potential improvements that can be applied to speed up the process.
   * Integrate the animation generation process into the UI to provide real-time render previews.

2. Pre-Rasterization Enhancement
   * Design and implement a GPU-accelerated pre-rasterization pass.
   * Integrate rasterization output with the ray tracing system.
   * Evaluate the performance impact of pre-rasterization.
   * Analyse image quality with/without pre-rasterization with the same number of iterations.

## Key features
### Primary Features
1. Performance Optimization
   * Multi-architecture GPU Backend
     * Generic gpu backend that utilizes existing crates to run on different gpu architectures.
     * The utilization of the [emu](https://github.com/calebwin/emu) crate will be our first experiment, because emu provides a clean macro interface for running programs on the GPU and has a relatively simple programming pattern.
     * Alternatively, implementation will explore wgpu for cross-platform compatibility [wgpu](https://github.com/gfx-rs/wgpu).
   * Nvidia-specific CUDA implementation
     * Targets Nvidia GPUs specifically.
     * Currently aiming to utilize the [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA/tree/master) project for implementing CUDA kernels.
   * CPU Performance Enhancement
     * Further optimization of existing multi-threaded CPU implementation by identifying sub-optimal execution patterns and optimizable performance bottlenecks.

Ray intersection calculations can be sent off to the GPU for ultra-quick processing.

The following crates will be explored to add GPU acceleration:
* emu - procedural macro uses OpenCL to automatically determine what parts of the provided code can be accelerated with a GPU. As it's automatic, it will be the easiest to use, but will likely not provide as much optimization compared to the below two options.
* WebGPU - Will need to compartmentalize the parallelizable code into compute shaders to then provide as a shader module for the program.
* opencl3

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

3. Animation System Integration
   * Physics-driven object motion.
   * Smooth camera path interpolation and rotation of object.    
    1. Keyframe
    2. cv::videoio::VideoWriter
    1. Camera movement
    2. Object movement (JUN HO FINISH)

### Secondary Features
1. Animation System Optimization
   * UI Integration and Monitoring
     * Real-time animation parameter controls in the interactive interface.
     * Live performance metrics and profiling data.
     * Frame-by-frame preview and adjustment capabilities.
   * Performance Enhancement
     * Systematic analysis of animation pipeline bottlenecks.
     * Implementation of caching strategies for repeated calculations.
2. Pre-Rasterization Enhancement
   * Quality Improvement Pipeline
     * Implementation of GPU-accelerated pre-rasterization stage.
     * Integration with existing ray tracing pipeline.
     * This will allow the render to generate a clear, serviceable image with much less compute required than a purely ray-traced render. This will produce a high-quality render in a shorter time by taking the best of both of rasterization (fast compute for a complete image) and ray tracing (realistic reflections). 


## Tentative plan

Submission deadline: Monday December 16th (6 weeks from proposal due)

The secondary objective 'good-to-haves' will be added to our plan, but will be removed if previous primary objetives will take longer than expected. Each point will have the details about person responsible added in the end (Jun Ho/Jackson/both).

* Week 1 (11/04 - 11/10) - 
  * We will meet regularly during the first week to complete the following tasks:
    * Performance profiling. We will work on identifying the performance bottlenecks of the current implementation. (to look into crates that help do this) (Both)
    * Explore GPU acceleration methods with emu, wgpu, and CUDA. Starting with emu, we will try offloading some loops to the GPU to see how they will work. By the end of the week, one method will be chosen for the hardware acceleration for the rest of the project.
      * emu (Jun Ho)
      * wgpu (Both)
      * cuda - if needed (Jackson)

* Week 2 (11/11 - 11/17) - 
  * Start writing compute shaders for the chosen crate. (Both - separate based on optimization points)
    * This will be advised by the performance profiling of the week before, as these bottlenecks will be able to be accelerated. (Both - based on shaders written)
    * The most computationally intensive portions of the code will be accelerated first. (Jackson expand)
    * While writing the shaders, we will profile the program to see what kind of speedup we get.
  * Start implementing the Keyframe crate into the current program for frame interpolation. (Jun Ho - do less optimizations to focus on this as well)
    * Progress on this point will largely depend on how well the GPU programming goes.
    * By this point, the animation won't be rendered, but the object locations of each frame will be able to be determined.

* Week 3 (11/18 - 11/24) - 
  * Further optimizations if they're found (Both)
  * Finish the bulk of the GPU shader programming to begin more detailled profiling. (Both)
    * If we are able to achieve more than 50x speedup on Jun Ho's machine, we will have reached our speedup goal.
  * Once our speedup goal is reached, then we will begin focusing on animation generation.
    * Finish keyframe generation implementation. (Jun Ho)
    * Add multiple frame generation based on the keyframes. This will make the program run for n-times the frames needed compared to the base image rendering
    * Start implementing the VideoWriter module to compile all the frames into one video (Jackson)

* Week 4 (11/25 - 12/01) - 
  * Further optimizations if they're found (Both)
  * Complete video generation feature (Both - depends on Keyframe and VideoWriter implementation progress)
  * Start designing the UI (Jackson - to expand)
  * Start adding rasterization calculations for the renderer (Jun Ho)
    * Instead of tracing rays from light sources, a fixed light source will be used where rays casted from the camera will be intersected with the objects on the scene.
* Week 5 (12/02 - 12/08) - 
  * Further optimizations if they're found (Both)
  * UI (Jackson to expand)
    * Render info to be added to the UI for easy render configuration
  * Combine rasterization with ray tracing to create a pre-rasterized render (Jun Ho)
    * Experiment with optimal combinations of ray tracing with pre-rasterization to determine what the better parameters will be
* Week 6 (12/09 - 12/15) - 
  * Further optimizations if they're found (Both)
  * Report writing based on our findings (Both)
  * Create video demo portraying: (Jun Ho to make video, Jackson to provide his machine's speedup info)
    * Speedup achieved from optimizations
    * Animation rendering feature
    * UI (if added)
    * Speedup from pre-rasterization (if added)