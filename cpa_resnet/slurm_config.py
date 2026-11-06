# slurm_config.py

# ---- SLURM settings ----
job_name_suffix = "Exper"
output_file = "./scriptsOut/filter"
error_file = "./scriptsErr/filter"
time_limit = "1400"       # minutes or Slurm format D-HH:MM:SS
nodes = 1
ntasks = 1
memory = 130000           # in MB; per-filter peak is ~80GB on the memmap pipeline -- the
                          # higher reservation discourages Slurm from co-locating two
                          # filter jobs on one node
cpus_per_task = 1
gpus = 1
nodelist = ""             # set to your cluster's node(s), or leave empty
array_concurrency = 32    # throttle concurrent jobs
partition = ""            # set to your cluster's partition, or leave empty
constraint = "geforce_rtx_3090"
# ---- Environment exports ----
LD_LIBRARY_PATH = "/usr/local/cuda-12.2/lib64/"
