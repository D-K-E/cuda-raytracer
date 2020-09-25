# cuda-raytracer
Simple Port of Raytracing in Series to CUDA

This project is largely insipired from 
[Roger Allen's work](https://github.com/rogerallen/raytracinginoneweekendincuda).

It humbly diverges from it with respect to 2 aspects:

- Build system is now CMake instead of a simple `Makefile`.
- `thrust` is used for handling host side pointers.

With `thrust` the code looks a little more readable in my opinion, 
but for the relatively small scope of this project, I'm not sure if it
makes that much of a difference.

Beware that you would need a relatively new version of CMake (3.18).

The branches of the repository is divided into small chunks to facilitate
the digestion of newly coded material as per the project of Roger Allen.

The overall project structure is also arranged to make it easier to
integrate with other projects.

All of the images below are realized with the following specs:

- Device name: Quadro M2000M
- Memory Clock Rate (KHz): 2505000
- Memory Bus Width (bits): 128
- Peak Memory Bandwidth (GB/s): 80.16


Renders the screenshot below at 800 x 600 in 32.3065 seconds with 50 samples
per pixel and 50 bounces per ray.

<img src="images/final.png" alt="final screenshot"/>

The one in below at 640x360 in 30.4133 seconds with 20 samples and 10 bounces.
<img src="images/final2.png" alt="final screenshot second version"/>


I don't remember the settings for this one. But it includes perlin noise so.
<img src="images/final3.png" alt="final screenshot third version"/>


The one in below at 800x450 in 49.5292 seconds with 500 samples and 500
bounces.
<img src="images/final4.png" alt="final screenshot fourth version"/>

The one in below at 640x360 in 106.365 seconds with 500 samples and 500
bounces
<img src="images/cornell_smoke.png" alt="final screenshot fifth version"/>

## Features

- Most of what you get in Ray Tracing In One Weekend and in Next Week.

- Multiple images in different sizes and channels.

- Perlin Noise

- Constant Medium Volume Rendering

## Planned

- Asset loading with assimp.
- A simple acceleration structure.
- Hopefully pdf handling in near feature.
- Background handling.


## Known Issues

- A Better RNG.

Cheers
