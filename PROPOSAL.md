# Project Proposal

Project proposal for ECE1724 - Performant systems in rust

## Motivation
Friend had rust project that renders scenes. Wanted to add more.
* Importance of generating realistic images in a scalable fashion is important (Jackson to expand)

This project currently has one major algorithmic optimization in the form of K-D trees. By subdividing the bounding box of objects within a scene, the program is able minimize ray tracing in areas that can be confirmed to not have any interecting objects. However, from our testing, we found that increasing the K-D tree depth can lead to longer rendering times, likely due to the size of the tree exploding as the depth increases. This means that even though the bounding box volume of the objects within the scene decrease, the time required to access all the nodes of the tree will increase. Therefore, we wanted to explore ways to further optimize this project in different ways, by accessing more hardware available to the system (GPUs), and implementing common software optimization techniqes.

More optimizations can be done, specifically parallelism.

We also wanted to take this project one step further. Of course, the natural step to make is to render animations. By leveraging the significant speedup we plan on achieving, we hope to create a scalable video rendering program.

We want to add a usable UI as well, to make it easier to use for anyone who wants to try rendering a scene of their own.

To get a baseline performance profile of the project at its current state, we rendered two different scenes with varying parameters. 
We used one Windows 11 and one MacOS machine to help see how different systems perform. By the end of the project, we will re-run the same tests after the project to determine what speedup we were able to achieve.

During the test runs, we noticed that the image turned out grainy a lot of the time with low-iteration or low-depth parameters. This was due to the renders being purely ray-traced, meaning that every light 'rays' generated had to hit every single point in the object to create a smooth render, making it highly compute intensive. Due to this effect, we were determined to find any ways that could make the quality of the image higher while keeping the number of computations the same.

* Parameters to sweep
1. samps_per_pix: 100 - 1000 - 10000
2. assured_depth: 1   - 2    - 5
3. kd_tree_depth: 2   - 8    - 17
    
Jun Ho's machine:
* CPU: AMD Ryzen 5800x
* GPU: AMD Radeon RX6800XT
* RAM: 32GB 3200MHz
* OS: Windows 11

Test
1. Wada
    1. wada_100_1_2: 21.9s
    2. wada_100_1_8: 27.6s
    3. wada_100_1_17: 28.4s
    4. wada_100_2_2: 25.7s
    5. wada_100_5_2: 41.0s
    6. wada_1000_1_2: 191.2s
    7. wada_10000_1_2: 1928.6s
    8. wada_1000_2_8: 372.6s
    9. wada_10000_5_17: 6914.1s

2. Biplane
    1. biplane_100_1_2: DNF (too long per iteration - ~885s/it)
    2. biplane_100_1_8: 2316.1s
    3. biplane_100_1_17: 328.3s
    4. biplane_100_2_2: DNF (too long per iteration - ~995s/it)
    5. biplane_100_5_2: DNF (too long per iteration - ~s/it)
    6. biplane_1000_1_2: DNF (too long per iteration - ~885s/it)
    7. biplane_10000_1_2: DNF
    8. biplane_1000_2_8: 
    9. biplane_10000_5_17: 37628.7s

3. 

Jackson's machine:
    CPU: (jackson to add)
    GPU:
    RAM:
    OS:
Test
1. 
2. 
3. 

## Objective
Improve performance and add fun features i.e. Animations!
(*Update based on profiling results above!!*)
* X% speedup
    * Identify the parallelizable portion of code to get a theoretical speedup limit, then give reasonable speedup objective (Jackson)
* Render a 24fps animation with object kinematics for a 2-second video within X minutes


## Key features
1. Multithreading (Jackson)
Parts of the program can be multithreaded ()

2. GPU Acceleration (Jun Ho)

Ray intersection calculations can be sent off to the GPU for ultra-quick processing.

The following crates will be explored to add GPU acceleration:
* emu - procedural macro uses OpenCL to automatically determine what parts of the provided code can be accelerated with a GPU. As it's automatic, it will be the easiest to use, but will likely not provide as much optimization compared to the below two options.
* WebGPU - Will need to compartmentalize the parallelizable code into compute shaders to then provide as a shader module for the program.
* opencl3

3. Animation - Two main crates will be used for this feature:
    1. Keyframe: 
    2. cv::videoio::VideoWriter:
    1. Camera movement
    2. Object movement

** Good-to-haves **
4. Rasterization (ray-casting) as a baseline, then ray-traced above
    a. This will allow the render to generate a clear, serviceable image with much less compute required than a purely ray-traced render. This will produce a high-quality render in a shorter time by taking the best of both worlds of rasterization () and ray tracing (realistic reflections). 
5. Simple UI to load configurations and run renders.
    a. This will allow

## Tentative plan

Submission deadline: Monday December 16th (6 weeks from proposal due)
To talk about in the weekend
Week 1 (11/04 - 11/10) - 
Week 2 (11/11 - 11/17) - 
Week 3 (11/18 - 11/24) - 
Week 4 (11/25 - 12/01) - 
Week 5 (12/02 - 12/08) - 
Week 6 (12/09 - 12/15) - 