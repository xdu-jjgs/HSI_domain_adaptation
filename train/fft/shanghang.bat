call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd-low_cut
python train/nommd/train.py configs/shanghang/fft/low_cut/nommd_01.yaml ^
        --path ./runs/shanghang/fft/low_cut_nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem nommd-high_cut
python train/nommd/train.py configs/shanghang/fft/high_cut/nommd_01.yaml ^
        --path ./runs/shanghang/fft/high_cut_nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem nommd-both_cut
python train/nommd/train.py configs/shanghang/fft/both_cut/nommd_01.yaml ^
        --path ./runs/shanghang/fft/both_cut_nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dan-high_cut
python train/ddc/train.py configs/shanghang/fft/high_cut/dan_01.yaml ^
        --path ./runs/shanghang/fft/high_cut_dan ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dann-high_cut
python train/dann/train.py configs/shanghang/fft/high_cut/dann_01.yaml ^
        --path ./runs/shanghang/fft/high_cut_dann ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem mcd-high_cut
python train/mcd/train.py configs/shanghang/fft/high_cut/mcd_01.yaml ^
        --path ./runs/shanghang/fft/high_cut_mcd ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2
