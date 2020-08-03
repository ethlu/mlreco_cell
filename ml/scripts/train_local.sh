module load pytorch/v1.5.0

export OMP_NUM_THREADS=32
python train.py $@
