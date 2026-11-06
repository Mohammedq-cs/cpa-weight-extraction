# slurm_config.py

# ---- SLURM settings ----
job_name_suffix = "CPA simulation"
output_file = "./scriptsOut/filter"
error_file = "./scriptsErr/filter"
time_limit = "1400"       # minutes or Slurm format D-HH:MM:SS
nodes = 1
ntasks = 1
memory = 60000            # in MB
cpus_per_task = 1
gpus = 1
nodelist =  ""
array_concurrency = 32    # throttle concurrent jobs
partition =  ""
constraint = "geforce_rtx_3090" 
# ---- Environment exports ----
LD_LIBRARY_PATH = "/usr/local/cuda-12.2/lib64/"
