call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%
SET CUDA_VISIBLE_DEVICES=1

python inference/inference_ddc.py ^
        runs/houston/nommd-train/1/config.yaml ^
        runs/houston/nommd-train/1/best.pth ^
        --path runs/houston/nommd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000