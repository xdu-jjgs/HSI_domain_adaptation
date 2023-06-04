call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python inference/inference.py ^
        runs/houston/nommd-train/config.yaml ^
        runs/houston/nommd-train/best.pth ^
        --path runs/houston/nommd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem ddc
python inference/inference.py ^
        runs/houston/ddc-train/config.yaml ^
        runs/houston/ddc-train/best.pth ^
        --path runs/houston/ddc-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem dan
python inference/inference.py ^
        runs/houston/dan-train/config.yaml ^
        runs/houston/dan-train/best.pth ^
        --path runs/houston/dan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem jan
python inference/inference.py ^
        runs/houston/jan-train/config.yaml ^
        runs/houston/jan-train/best.pth ^
        --path runs/houston/jan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem dsan
python inference/inference.py ^
        runs/houston/dsan-train/config.yaml ^
        runs/houston/dsan-train/best.pth ^
        --path runs/houston/dsan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem dann
python inference/inference_dann.py ^
        runs/houston/dann-train/config.yaml ^
        runs/houston/dann-train/best.pth ^
        --path runs/houston/dann-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem mcd
python inference/inference_mcd.py ^
        runs/houston/mcd-train/config.yaml ^
        runs/houston/mcd-train/best.pth ^
        --path runs/houston/mcd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem dst
python inference/inference_dst.py ^
        runs/houston/dst_1_1_1_07_2-train/config.yaml ^
        runs/houston/dst_1_1_1_07_2-train/best.pth ^
        --path runs/houston/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2
