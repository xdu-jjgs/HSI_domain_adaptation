call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

@echo off
setlocal enabledelayedexpansion
SET CUDA_VISIBLE_DEVICES=0
set CONFIGS_FOLDER=configs/houston/dsn/hypers
set RUNS_FOLDER=runs/houston_hyper/

REM Loop through each YAML file in the config folder
for %%f in (%CONFIGS_FOLDER%\*.yaml) do (
    REM Extract the filename without extension
    set FILENAME=%%~nf

    REM Define the output path dynamically
    set OUTPUT_PATH=!RUNS_FOLDER!\!FILENAME!-train

    REM Execute the training script with the appropriate parameters
    python train\dsn\train_nodecoder_noamp_ddc_score2_open_ada_ema.py %%f ^
        --path !OUTPUT_PATH! ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9000 ^
        --seed 1

    REM Optional: Echo the command for verification
    echo Running training for config file %%f
)