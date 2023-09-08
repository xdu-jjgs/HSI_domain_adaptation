call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd-low_cut
python train/nommd/train.py configs/houston/fft/low_cut/nommd_01.yaml ^
        --path ./runs/houston/fft/low_cut_nommd ^
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
        --path ./runs/houston/fft/high_cut_nommd ^
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
        --path ./runs/houston/fft/both_cut_nommd ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem ddc-low_cut
python train/ddc/train.py configs/houston/fft/low_cut/ddc_01.yaml ^
        --path ./runs/houston/fft/low_cut_ddc ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem ddc-high_cut
python train/ddc/train.py configs/houston/fft/high_cut/ddc_01.yaml ^
        --path ./runs/houston/fft/high_cut_ddc ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem ddc-both_cut
python train/ddc/train.py configs/houston/fft/both_cut/ddc_01.yaml ^
        --path ./runs/houston/fft/both_cut_ddc ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dann-low_cut
python train/dann/train.py configs/houston/fft/low_cut/dann_01.yaml ^
        --path ./runs/houston/fft/low_cut_dann ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dann-high_cut
python train/dann/train.py configs/houston/fft/high_cut/dann_01.yaml ^
        --path ./runs/houston/fft/high_cut_dann ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dann-both_cut
python train/dann/train.py configs/houston/fft/both_cut/dann_01.yaml ^
        --path ./runs/houston/fft/both_cut_dann ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem mcd-low_cut
python train/mcd/train.py configs/houston/fft/low_cut/mcd_01.yaml ^
        --path ./runs/houston/fft/low_cut_mcd ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem mcd-high_cut
python train/mcd/train.py configs/houston/fft/high_cut/mcd_01.yaml ^
        --path ./runs/houston/fft/high_cut_mcd ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem mcd-both_cut
python train/mcd/train.py configs/houston/fft/both_cut/mcd_01.yaml ^
        --path ./runs/houston/fft/both_cut_mcd ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2