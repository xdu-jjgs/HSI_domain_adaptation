call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python inference/inference.py ^
        runs/shanghang/nommd-train/config.yaml ^
        runs/shanghang/nommd-train/best.pth ^
        --path runs/shanghang/nommd-train/1 ^
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
        runs/shanghang/ddc-train/config.yaml ^
        runs/shanghang/ddc-train/best.pth ^
        --path runs/shanghang/ddc-train/1 ^
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
        runs/shanghang/dan-train/config.yaml ^
        runs/shanghang/dan-train/best.pth ^
        --path runs/shanghang/dan-train/1 ^
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
        runs/shanghang/jan-train/config.yaml ^
        runs/shanghang/jan-train/best.pth ^
        --path runs/shanghang/jan-train/1 ^
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
        runs/shanghang/dsan-train/config.yaml ^
        runs/shanghang/dsan-train/best.pth ^
        --path runs/shanghang/dsan-train/1 ^
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
        runs/shanghang/dann-train/config.yaml ^
        runs/shanghang/dann-train/best.pth ^
        --path runs/shanghang/dann-train/1 ^
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
        runs/shanghang/mcd-train/config.yaml ^
        runs/shanghang/mcd-train/best.pth ^
        --path runs/shanghang/mcd-train/1 ^
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
        runs/shanghang/dst_1_1_1_07_2-train/config.yaml ^
        runs/shanghang/dst_1_1_1_07_2-train/best.pth ^
        --path runs/shanghang/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000 ^
        --opt-level O2

rem tstnet
python inference/inference_tstnet.py ^
       runs/shanghang/dann-train/config.yaml ^
       E:/zts/IEEE_TNNLS_TSTnet/results/shanghang.npy ^
       --path runs/shanghang/tstnet/1 ^
       --nodes 1 ^
       --gpus 1 ^
       --rank-node 0 ^
       --backend gloo ^
       --master-ip localhost ^
       --master-port 8890 ^
       --sample-number 1000 ^
       --opt-level O0