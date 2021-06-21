# Analysis

This directory is for visualization and evaluation of the network output as well as the comparisons to other algorithms. 

## Scripts
The scripts contain core analysis features.

- *util.py* has common utilities that manipulate, analyse, and plot "voxels" which are dictionaries that each encapsulate an image. Every key in a voxel is a coordinate tuple `(x, y, z)` and they each correspond to a number that may represent a sigmoid value, charge, energy, or is `None`. 

	One analysis function is `SP_curve` which computes the sensitivity-purity curve of an inference voxel (with sigmoid values) w.r.t. True voxel. There are also track related functions that extract features like direction and dQ/dX.
	
- *products.py* is used to examine data extracted from the ROOT files as well as the result of tiling called "active" voxels. Note: "mc" refers specifically to particle level information, like PDG code, applicable to BeamCosmic events.

- *yinf.py* is for examining network outputs. Other than plotting individual events, the "stats" functions analyse the sensitivity and purity of the (deghosting) network.

## Notebooks

The following notebooks have sample outputs saved:

- *analysis.ipynb* is for general analysis that mainly runs the Python scripts above. Analysis on the latest (deghosting) network is shown.

- *charge.ipynb* has displays of Electron, Muon and BeamCosmic events reconstructed with the network, SpacePointSolver and Wirecell

- *protoDUNE.ipynb* has displays of protoDUNE events reconstructed with the network, SpacePointSolver, Wirecell, and (Pandora) Calorimetry. 

- *track.ipynb* examines dQ/dX of stopping muon tracks reconstructed with the network, (Pandora) Calorimetry, SpacePointSolver, and MC truth (where the true deposition is track-fitted to extract dQ/dX).

To run them, may need to point `ML_DIR` to where the network outputs are stored and `SIM_DIR` to where the "parsed" and "xy" directories are. Some code also uses "stats" files which contain SP curves or "mc" files with particle level info (see above), though they can be generated using the scripts from the former.
