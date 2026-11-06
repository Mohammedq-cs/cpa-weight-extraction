import subprocess
import slurm_config as cfg

def generate_slurm_script(job_name, output_file, error_file, time_limit, nodes, ntasks, memory, cpus_per_task, gpus, nodelist, array_range, layer_name, numTraces, noise_level, exp_name, leakage_model, bsInpsPath, script_name="extractFilterSignAmbiguousMultiPatchesOptimized"):
    """
    Generate a SLURM job script as a string.

    Parameters:
    - job_name: Name of the job.
    - output_file: Path for the standard output file.
    - error_file: Path for the standard error file.
    - time_limit: Maximum runtime (in minutes).
    - nodes: Number of nodes.
    - ntasks: Number of tasks.
    - memory: Memory allocation (in MB).
    - cpus_per_task: Number of CPU cores per task.
    - gpus: Number of GPUs.
    - nodelist: Specific nodes to run the job on.
    - partition: Partition to submit the job to.
    - array_range: Range for the job array (e.g., "0-99").
    - layer_name: Name of the layer.

    Returns:
    - A string containing the SLURM job script.
    """

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}_%a.out
#SBATCH --error={error_file}_%a.err
#SBATCH --time={time_limit}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --nodelist={nodelist}
#SBATCH --array={array_range}
#SBATCH --partition={cfg.partition}
#SBATCH --constraint={cfg.constraint}

# Load necessary modules or activate environments if required
# module load python/3.8

# Execute the Python script with the appropriate arguments

export LD_LIBRARY_PATH={cfg.LD_LIBRARY_PATH}

python {script_name}.py \\
    --baseLineInputDir "./baseLineInps/{bsInpsPath}/{numTraces}" \\
    --simulatedInputsDir "./simInps/N{noise_level}/{exp_name}/{numTraces}" \\
    --corrStatSavePath "./N{noise_level}/{exp_name}Corr/{numTraces}"\\
    --layerName "{layer_name}" \\
    --filterIdx ${{SLURM_ARRAY_TASK_ID}} \\
    --outputDir "./N{noise_level}/{exp_name}output/{numTraces}" \\
    --NoiseLevel {noise_level} \\
    --leakage_model {leakage_model}
    """
    
    return script
#update the path later
def submit_batch(start, end, concurrent_jobs, name, numTraces, noise_level, leakage_model, exp_name, bsInpsPath, script_name="extractFilterSignAmbiguousMultiPatchesOptimized"):
    job_script = generate_slurm_script(
        job_name=f"{name}{cfg.job_name_suffix}",
        output_file=cfg.output_file,
        error_file=cfg.error_file,
        time_limit=cfg.time_limit,
        nodes=cfg.nodes,
        ntasks=cfg.ntasks,
        memory=cfg.memory,
        cpus_per_task=cfg.cpus_per_task,
        gpus=cfg.gpus,
        nodelist=cfg.nodelist,
        array_range=f"{start}-{end}%{concurrent_jobs}",
        layer_name=f"{name}",
        numTraces = f"{numTraces}",
        noise_level = f"{noise_level}",
        leakage_model = f"{leakage_model}",
        exp_name = f"{exp_name}",
        bsInpsPath = bsInpsPath,
        script_name = script_name
    )

    slurm_script_filename = "extractFilter.sbatch"
    with open(slurm_script_filename, "w") as slurm_script_file:
        slurm_script_file.write(job_script)

    print(f"Submitting jobs {start} to {end} now")
    result = subprocess.run(['sbatch', '--wait', slurm_script_filename], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Job array submitted successfully:\n{result.stdout}")
    else:
        print(f"Error submitting job array:\n{result.stderr}")

