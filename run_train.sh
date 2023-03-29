python train.py \
       --config-dir=diffusion_policy/config/  \
       --config-name=train_diffusion_unet_hybrid_workspace.yaml \
       training.seed=42 \
       training.device=cuda:0 \
       hydra.run.dir='data/outputs/2023.03.29/01.28.41_train_diffusion_unet_hybrid_pick_up_ball' \
       task=pick_up_ball

# Typical run.dir
#        hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
