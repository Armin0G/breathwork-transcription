#!/bin/bash
#SBATCH --partition=GPUshortx86
#SBATCH --nodelist=esi-svhpc107
#SBATCH --gpus=1
#SBATCH --job-name=tsc_bw
#SBATCH --error=error_tsc_%j.log
#SBATCH --time=02:00:00

source /cs/opt/env/python/x86_64/miniforge/bin/activate
conda activate tsc_dia

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /cs/home/goffina/breathwork-transcription
python pipeline/run_pipeline.py --input "/cs/projects/HEBznlReset/ZNLRESET-DATA/VIDEO_Ag/1W2ML9/BW/video_recording_2026-03-30T13_30_23/" --no-diarization

echo "Job finished at $(date)"

