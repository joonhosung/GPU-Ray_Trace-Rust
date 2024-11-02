<!-- omit in toc -->
# Project Proposal

Project proposal for ECE1724 - Performant Software Systems with Rust

Team members: Jackson Nie (1005282409) Jun Ho Sung ()

<!-- omit in toc -->
## Table of Contents
- [Motivation](#motivation)
- [Objectives](#objectives)
  - [Primary Objectives](#primary-objectives)
  - [Secondary Objectives](#secondary-objectives)
- [Key features](#key-features)
- [Tentative plan](#tentative-plan)

## Motivation
Our project proposal stems from a shared passion for gaming and computer graphics, combined with a fascinating opportunity to enhance an existing ray tracing implementation.

Ray tracing, which creates photorealistic images by simulating light-object interactions, has become increasingly relevant in modern graphics applications, especially in gaming and visual effects. While the current ray-tracer implementation produces high-quality images through multithreading and kd-tree acceleration, its performance remains a significant bottleneck. This limitation presents an exciting opportunity for optimization, particularly through GPU acceleration. 

This motivated us to propose an endeavor that aims to accelerate the ray-tracing process leveraging modern GPUs. Both team members have experience in GPU programming with C++, but neither has experience implementing GPU solutions in Rust. This project therefore offers an ideal intersection of learning and practical application. As a language that promises "blazingly fast" performance while maintaining memory safety, Rust presents an excellent platform for high-performance computing tasks like ray tracing. By accelerating a computationally intensive graphics application, we will gain hands-on experience and learn the techniques of writing high-performance applications in Rust. This aligns perfectly with the course's goal of developing performant and scalable systems.

Beyond pure performance optimization, we aim to expand the ray tracer's capabilities by implementing new features such as an interactive, visual user interface as well as short scene generation. These additions will make the project more engaging, and this combination of optimization and feature development will also provide additional challenges in maintaining performance at scale. 

To summarize, through this project, we expect to:
* Significantly improve ray tracing performance through GPU acceleration
* Gain practical experience with Rust in high-performance computing
* Implement new features that showcase intriguing capabilities of the ray tracer

//// TODO: Update
(Some testing metrics)
* Which tests to run
1. wada_w_front
2. james_webb
3. biplane

* Which parameters to sweep
1. samps_per_pix: 100 - 5000 - 50000
2. assured_depth: 1 - 2 - 5
3. kd_tree_depth: 2 - 8 - 17
    
* Try it out and post results (time taken) below

Jun Ho's machine:
    CPU: AMD Ryzen 5800x
    GPU: AMD Radeon RX6800XT
    RAM: 32GB 3200MHz
    SSD: 2TB 
Test
1. 
2. 
3. 

Jackson's machine:
    CPU: (jackson to add)
    GPU:
    RAM:
    SSD:
Test
1. 
2. 
3. 
//// End TODO

## Objectives
### Primary Objectives
1. Performance Optimization
   * Port the existing CPU-based ray tracer to utilize GPU acceleration in RUst.
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
   * Evaluate the performance impact of pre-rasterization and also the affect on the final output image.

## Key features
1. Multithreading (Jackson)
Parts of the program can be multithreaded ()

2. GPU Acceleration (Jun Ho)
Ray intersection calculations can be sent off to the GPU for ultra-quick processing.

The following GPU acceleration options will be explored
* Using EMU (to test out and see how well it works)
* 

3. Animation (Jun Ho)
    a. Camera movement
    b. Object movement

** Good to have **
4. Rasterization as a baseline 

## Tentative plan

Submission deadline: Monday December 16th (6 weeks from proposal due)
To talk about in the weekend
Week 1 (11/04 - 11/10) - 
Week 2 (11/11 - 11/17) - 
Week 3 (11/18 - 11/24) - 
Week 4 (11/25 - 12/01) - 
Week 5 (12/02 - 12/08) - 
Week 6 (12/09 - 12/15) - 