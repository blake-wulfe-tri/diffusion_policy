docker run --gpus all -it --rm \
       -v /home/blakewulfe/programming/research/diffusion_policy/:/home/diffusion_policy  \
       -v /ssd1/datasets/robotics/:/home/datasets \
       --shm-size=64G \
       --name diffusion_policy_01 \
       diffusion_policy
