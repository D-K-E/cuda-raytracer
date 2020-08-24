# cuda-raytracer
Simple Port of Raytracing in One Weekend to CUDA

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

Cheers
