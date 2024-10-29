# Project Proposal

Project proposal for ECE1724 - Performant systems in rust

## Motivation
Friend had rust project that renders scenes. Wanted to add more.
* Importance of generating realistic images in a scalable fashion is important (Jackson to expand)
The program currently has minimal optimizations. There are some algorithmic improvements he made, specifically using K-D trees (Jun Ho to describe more).
More optimizations can be done, specifically parallelism.

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