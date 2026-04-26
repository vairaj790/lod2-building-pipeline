docker run --gpus all -it --shm-size=1g -v C:/Users/fo37nor/job_work/heat:/opt/heat vaibhavrajan79/heat_deformable-detr-image:v1 bash


**********************************Train********************************************************

for training:

docker run --gpus all -it --shm-size=1g -p 6006:6006 -v C:/Users/fo37nor/job_work/heat:/opt/heat vaibhavrajan79/heat_deformable-detr-image:with_tensorboard bash


for monitoring the training:

tensorboard --logdir ./checkpoints/my_finetune_256/tensorboard_logs --port 6006 --bind_all


to run it in a different terminal:

docker exec -it objective_thompson bash

to train:

CUDA_VISIBLE_DEVICES=0 python train.py --run_validation


**************************************************************************************************






*****************************************************************************************************************************
For Training on HPC with tensorboard

srun -p gpu-test --gres=gpu:1 --time=12:00:00 --pty bash

After getting into a gpu node

singularity exec --cleanenv --nv -B /home/fo37nor/assets/heat:/opt/heat -B /cluster:/cluster /home/fo37nor/assets/heat/heat_with_tensorboard.sif bash

bash /opt/heat/start_train_512.sh

Open a new terminal on the same hpc, but it will again be in the login node, so enter the gpu node which was alloted 
ssh gpu010    #example

Then start Singularity again:
singularity exec --cleanenv --nv -B /home/fo37nor/assets/heat:/opt/heat -B /cluster:/cluster /home/fo37nor/assets/heat/heat_with_tensorboard.sif bash

bash /opt/heat/start_train_tensorboard_512.sh

Then from your Windows machine open another local terminal and make the tunnel:
ssh -L 6006:gpu010:6006 fo37nor@login2.draco.uni-jena.de
Then open in browser:
http://localhost:6006

************************************************************************************************************************
For runnning with slurm



sbatch /home/fo37nor/assets/heat/run_heat_512.slurm

If submitted, check whether the job is running, also check the gpu node by

squeue -u fo37nor

From your Windows machine, open a terminal and run:

ssh -L 6006:gpunode:6006 fo37nor@login2.draco.uni-jena.de

Then in your browser open:

http://localhost:6006

**********************************************************************************************************************
infer locally


python infer.py --checkpoint_path ./checkpoints/my_finetune_256/checkpoint_best.pth  --dataset outdoor --image_size 256 --viz_base ./results/viz_heat_outdoor_256 --save_base ./results/npy_heat_outdoor_256    #for refined model


python infer.py --checkpoint_path ./checkpoints/ckpts_heat_outdoor_256/checkpoint.pth  --dataset outdoor --image_size 256 --viz_base ./results/viz_heat_outdoor_256 --save_base ./results/npy_heat_outdoor_256      #for original model

