# Boltzmann-Tucker
Boltzmann-Tucker is a Python implementation of the tensorized discrete velocity method (DVM) for the numerical solution of the Boltzmann equation with a model collision integral in 3D spatial domains. The use of the Tucker tensor decomposition drastically (up to 100 times) reduces required storage compared to the standard DVM. This allows to solve essentially 3D problem on a computer with rather small RAM size.

Short description of the current version:
* Shakhov collision integral for a monoatomic gas 
* Unstructured hexahedral meshes stored in the ASCII StarCD format
* 1st order finite-volume explicit and implicit methods 
* Implemented boundary conditions:
  * free stream
  * wall with a constant temperature
  * plane of symmetry

Solver is based on the customized part of the library https://github.com/rakhuba/tucker3d.

