Repository Hierarchy:

- `torch`: stores the necessary python files for generating the MNIST datasets we used for our measurements

- `typehint`: contains Python driver to process lines that contains `\\typehint`; see Manual for details

- `vgrad`: contains the VGrad library with all necessary functions definitions
  - `include`: install library by downloading the include folder, which contains all necessary headers
  - `examples`: contains examples used for demonstrating VGrad functionality
  - `measurements`: contains code used to collect runtime measurements

See the VGrad Tutorial document for compilation instructions.
