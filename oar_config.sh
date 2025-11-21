NAME="mistral24_text_reconstruction"
PROJECT_NAME="test1"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test1"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=2                  # Walltime in hours (2h is plenty for 5 examples)
L_NGPUS=1                  # Number of GPUs
P_MINCUDACAPABILITY=6      # Minimum compute capability (6 for 1080Ti, 7 for V100/A100)
P_MINGPUMEMORY=24000       # Minimum GPU memory in MB (24 GB is enough with 4-bit quantization)

# Submit the job
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "export HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN; \
     module load conda; \
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     echo 'Starting Mistral text reconstruction...'; \
     python3 full_gen_mistral24.py \
        --input_file out_jsonl/dev.jsonl \
        --output_dir ./reconstructed_outputs \
        --limit 5; \
     echo 'Mistral text reconstruction completed.'
    " \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

