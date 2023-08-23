call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd-low_cut
python train/nommd/train.py configs/houston/fft/low_cut/nommd_01.yaml ^
        --path ./runs/houston/fft/low_cut/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem nommd-high_cut
python train/nommd/train.py configs/houston/fft/high_cut/nommd_01.yaml ^
        --path ./runs/houston/fft/high_cut/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem nommd-both_cut
python train/nommd/train.py configs/houston/fft/both_cut/nommd_01.yaml ^
        --path ./runs/houston/fft/both_cut/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2