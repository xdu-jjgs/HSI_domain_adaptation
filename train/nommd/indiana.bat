call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python train/nommd/train.py configs/indiana/nommd/nommd.yaml ^
        --path ./runs/indiana/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem nommd
python train/nommd/train.py configs/indiana/nommd/nommd_1800_average.yaml ^
        --path ./runs/indiana_sample/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2