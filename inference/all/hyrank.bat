call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python inference/inference.py ^
        runs/hyrank/nommd-train/config.yaml ^
        runs/hyrank/nommd-train/best.pth ^
        --path runs/hyrank/nommd-train/1 ^
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
        runs/hyrank/ddc-train/config.yaml ^
        runs/hyrank/ddc-train/best.pth ^
        --path runs/hyrank/ddc-train/1 ^
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
        runs/hyrank/dan-train/config.yaml ^
        runs/hyrank/dan-train/best.pth ^
        --path runs/hyrank/dan-train/1 ^
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
        runs/hyrank/jan-train/config.yaml ^
        runs/hyrank/jan-train/best.pth ^
        --path runs/hyrank/jan-train/1 ^
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
        runs/hyrank/dsan-train/config.yaml ^
        runs/hyrank/dsan-train/best.pth ^
        --path runs/hyrank/dsan-train/1 ^
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
        runs/hyrank/dann-train/config.yaml ^
        runs/hyrank/dann-train/best.pth ^
        --path runs/hyrank/dann-train/1 ^
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
        runs/hyrank/mcd-train/config.yaml ^
        runs/hyrank/mcd-train/best.pth ^
        --path runs/hyrank/mcd-train/1 ^
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
        runs/hyrank/dst_1_1_1_07_2-train/config.yaml ^
        runs/hyrank/dst_1_1_1_07_2-train/best.pth ^
        --path runs/hyrank/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2
