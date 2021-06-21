# Machine Learning
This directory contains code to preprocess (including "tiling") data and to
perform the training and inference of the neural network. 

## Preprocessing: *make_batches.py*
The goal is to create "xy" files which contain ("tiled") input events and (possibly)
MC truth to feed into the network.

### Main function:`process_xy`
- **Input**
	- Inputs are the "parsed" files inside `parsed_dir` (created with the *data* package). This directory can be specified at command line: see **Run**
	- The product types are specified at the top of *make_batches.py*: `X_TYPE` is the type of the TPC channel file (default is "wire" i.e. recob::Wire). `Y_TYPE` is that of the truth information. (Note: for sake of convenience the types can also be specified as tuples, in which case they are searched in order). Setting `Y_TYPE` to `None` will include no truth info (necessary for experimental data).
	- The TPC number (`tpc_num`) should also be specified: see **Run**.
	- (Optional) By default all relevant files in `parsed_dir` (that have not already been processed to the `xy_dir`) will be processed, but filtering can be done by tweaking the `in_fd` variable. (See the commented out line for an example of selecting a range of file indices.)
 
- **Output**
	- Outputs are "xy" files written to `xy_dir`: see **Run**.
	- The "xy" files can be generated at a lower resolution than the common voxel size (defined in *geom* package). This is configured by the `downsample` argument which, by default, reduces the x, y, z dimensions by a factor of 8, 4, 4 respectively. This mainly lowers computation cost of the neural net, but to some extent also increases its efficacy.
	- (Optional) It is possible to split each input file into multiple "xy" files each containing `batch_size` events: see **Run**. This is primarily used to adjust batch size for training.
	- (Optional) Also possible to save the tiled image (at the default resolution) per se in another file of type "pixel" to the `parsed_dir` by setting the `save_pix` argument to True. 

- **Run**
	-  Dependency: `numba 0.50.1` to speed up and parallelize. (Compatibility with other versions not tested)
	- *scripts/make_batches.sh* can be used to run the function, which sets up the environment and then parses command line options (with some default values). Note: modify the environmental setup in the beginning as needed. 

		Example: `scripts/make_batches.sh -i parsed_dir -o out_dir -t tpc_num`
		
		Note: the actual output "xy" directory is *BASE_DIR/out\_dir* where BASEDIR is a variable in *scripts/make_batches.sh* currently set to *SCRATCH/larsim*.
		
		Additional options are `-b batch_size` and `-n num_threads`, the latter is for OpenMP parallelization, defaulting to 32.
	- *scripts/make\_batches_cori.sh* can be used to submit Slurm job on Cori:

		Example: `sbatch -N num_nodes -t time scripts/make_batches_cori.sh`
		
### Joining TPCs: `join_tpcs`
The xy files generated above are per TPC, the natural unit of tiling. It might be useful to merge them into a single image using this function. The argument `tpcs` specifies a list of tpc numbers to include, default is `[1, 5, 9]`, the three along the protoDUNE beam. 

To run, first uncomment the `join_tpcs` call at the bottom of *make_batches.py* (and probably comment out the *process_xy* call). 

Then `python3 make_batches.py xy_dir a a` (the two "a"s are just fillers because  *make\_batches.py* expects three command line arguments)
		
## Neural Network

The boilerplate used is [NERSC Pytorch Example](https://github.com/NERSC/pytorch-examples).

### Models: *models/*

- The UResNet model `SparseCellNet` in *models/cellnet_sparse.py* is based on [SLAC group's code](https://github.com/DeepLearnPhysics/lartpc_mlreco3d/blob/develop/mlreco/models/uresnet_lonely.py). Its main arguments are
	- `nChannels`: the number of "features" in the first layer of the net.
	- `nStrides`: the depth (i.e. number of blocks of different spatial sizes)
	- `n_2D`: number of "2D" blocks where the first spatial dimension (drift direction) is ignored, as if the 3D image is just a batch of 2D images. The idea was that the drift direction is supposed to have less ambiguity and so need less processing. In practice, however, `n_2D = 0` seems to work best (in which case the network is just a standard 3D UResNet).

- `BinarySparseCellNet` in *models/cellnet_binary.py* is a wrapper around `SparseCellNet` with sigmoid activation at the output in order to predict whether each voxel contains charge or not.
- `ChargeSparseCellNet` in *models/cellnet_charge.py* is another wrapper with linear layer(s) at the output instead. It is used to predict the charges in the voxels. The number of linear layers is specified by `nHiddenLayers`.

### Configuration: *configs/*

Each configuration yaml file specifies nearly all parameters used to run a network, including data, model, loss function etcs. 

Let's learn by examples:

- *ghost3d.yaml* is configured for training a "deghosting" net to predict whether each voxel contains charge or not. Some noteworthy features:
	- `output_dir` is where the outputs, e.g. trained network weights, summaries, validation sample outputs, will be saved (or retrieved).
	-  `store_inference` stores the inference data outputs (see `data`) as "yinf" files. The options `everyNepoch` and `everyNsample` stores periodically to save space. The default directory is *inference* but can be overriden with `outdir`.
	- `data` is the dataset configuration. 
		- The `sparse` dataset refers to *datasets/sparse.py*
		- `data_path` refers to the "xy" directories containing the data. 
		- `train_i` has ranges of file indices (inclusive) used as training data (here files 701-950 of the Electron "xy" dir and files 101-840 of the BeamCosmic dir are selected for training). 
		- `inference_i` has ranges of file indices used as validation data. They are the ones saved if the `store_inference` option is toggled.
		- (Optional) `tpcs` is an optional argument that can filter out TPCs (applied to both training and validation data). Here we select TPC 1 of the BeamCosmic samples but place no restriction on the Electron samples.
		- `threshold` sets the threshold of the voxel value above which we deem it a "true" voxel, i.e. one that contains charge.
		- `batch_size` should always be 1 when using the `sparse` dataset for syntatic reasons. However the actual batch size is that of a single "xy" file.
	- `model` defines the model, here the binary model. See above for more details, but two additional parameters are important:
		- `spatial_size` is the size of the input image
		- `downsample_t` should be set to [2, 2] when n_2D == 0
	- `trainer` is in *trainers/*. `loss`, `optimizer`, `metric` are in either Pytorch natively or in *utils/*. `n_epochs` of the `train` configuration is the number of epochs to train for.	  

- *charge.yaml* is configured for training a charge net. It's pretty much identical to *ghost3d.yaml* except with `model` and `loss` modified, and the `threshold` field of `data` removed. 
	- (Optional) Transfer learning can be done by setting `transfer` to True under `train` (e.g. from the ghost net to the charge net). Though it requires copying over the checkpoint file and *summaries_0.csv* to the new output_dir so that the new network can "resume" from the trained weights.
	
- *ghost_inf.yaml* is configured to just run inference on a trained network: simply remove the `train_i` field of `data` configuration to perform inference only.
	- Note: `optimizer` and other training specific fields are "required" but simply ignored. 
	- Note:  should set `--resume` option (see **Run**) to use the trained network. Also, note that under `train` field, `start_epoch` specifies the checkpoint file (by default it will be the last epoch). It can also be passed in command line: `--resume start_epoch`

### Run

- Dependency: Pytorch, [sparseconvnet](https://github.com/facebookresearch/SparseConvNet)

- *scripts/train_local.sh* can be used to run on Cori login node, for example to run inference:

	`scripts/train_local.sh configs/ghost_inf.yaml --resume -v`
	
	`--resume` is necessary when working with a trained network. It should also be used to continue training. `-v` is verbose output.

- *scripts/train_cori.sh* is the Slurm batch script, e.g.:

	`sbatch -N 5 -t 60 scripts/train_cori.sh configs/ghost3d.yaml`
	
	
 


