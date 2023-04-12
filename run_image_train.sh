python train.py \
       --config-dir=diffusion_policy/config/  \
       --config-name=train_diffusion_unet_image_workspace.yaml \
       training.seed=42 \
       training.device=cuda:0 \
       hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
       task=pick_up_ball

# Typical run.dir
#        
