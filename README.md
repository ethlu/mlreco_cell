# mlreco_cell

- This repository is for reconstruction of wire-based TPC using neural network.
The targeted experiment is ProtoDUNE though the design is modular so that another experiment can be accomodated rather easily.

- The functionalities are divided into packages in the subdirectories. Enter them for more info and to run their code. The main workflow is *data* -> *ml* -> *analysis* 

	*geom* and *tiling* are "helper" packages that don't need running explicitly. Though the APA geometry files should be generated once beforehand (see *geom/README.md*)

- The overall dependency is just python3 and the standard scientific packages. Read the package docs for additional dependencies if any. 

	This top level directory should be in the Python path. e.g. `export PYTHONPATH=$PYTHONPATH:~/mlreco_cell`

