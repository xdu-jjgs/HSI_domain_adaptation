call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python inference/inference.py ^
        runs/hyrank/ddc-train/config.yaml ^
        runs/hyrank/ddc-train/best.pth ^
        --path runs/hyrank/ddc-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --opt-level O2
