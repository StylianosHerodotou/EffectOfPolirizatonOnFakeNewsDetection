import os

device= "cpu"
gpus_per_trial=0;
dir_to_base="/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/"
dir_to_large=os.path.join(dir_to_base, "Datasets/sag")
dir_to_ray_checkpoints=os.path.join(dir_to_base, "Ray_Tune_Checkpoints")
dir_to_ray_results=os.path.join(dir_to_base, "ray_results")



