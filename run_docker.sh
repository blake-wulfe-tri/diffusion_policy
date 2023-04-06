docker run --gpus all -it --rm \
       -v /home/blakewulfe/programming/research/diffusion_policy/:/home/diffusion_policy  \
       -v /ssd1/datasets/robotics/:/home/datasets \
       -p 8888:8888 \
       --shm-size=100G \
       --name diffusion_policy_01 \
       diffusion_policy
