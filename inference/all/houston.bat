call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
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


rem ddc
python inference/inference_ddc.py ^
        runs/houston/ddc-train/1/config.yaml ^
        runs/houston/ddc-train/1/best.pth ^
        --path runs/houston/ddc-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000


rem dan
python inference/inference_ddc.py ^
        runs/houston/dan-train/1/config.yaml ^
        runs/houston/dan-train/1/best.pth ^
        --path runs/houston/dan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000


rem jan
python inference/inference_ddc.py ^
        runs/houston/jan-train/1/config.yaml ^
        runs/houston/jan-train/1/best.pth ^
        --path runs/houston/jan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000


rem dsan
python inference/inference_ddc.py ^
        runs/houston/dsan-train/1/config.yaml ^
        runs/houston/dsan-train/1/best.pth ^
        --path runs/houston/dsan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000


rem dann
python inference/inference_dann.py ^
        runs/houston/dann-train/2/config.yaml ^
        runs/houston/dann-train/2/best.pth ^
        --path runs/houston/dann-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000


rem mcd
python inference/inference_mcd.py ^
        runs/houston/mcd-train/1/config.yaml ^
        runs/houston/mcd-train/1/best.pth ^
        --path runs/houston/mcd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000

rem dst
python inference/inference_dst.py ^
        runs/houston/dst_1_1_1_07_2-train/1/config.yaml ^
        runs/houston/dst_1_1_1_07_2-train/1/best.pth ^
        --path runs/houston/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --sample-number 1000

rem dsn
python inference/inference_dsn.py ^
        runs/houston/dsn_resnet18-train/1/config.yaml ^
        runs/houston/dsn_resnet18-train/1/best.pth ^
        --path runs/houston/dsn_resnet18-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --sample-number 1000

rem s4dl
python inference/inference_dsn_rev_ddc.py ^
        runs/houston/dsn_rev_nodecoder_ddc_nospec_revnet38_grad_filter_ada_5_100_250_3_tmp-train/1/config.yaml ^
        runs/houston/dsn_rev_nodecoder_ddc_nospec_revnet38_grad_filter_ada_5_100_250_3_tmp-train/1/best.pth ^
        --path runs/houston/dsn_rev_nodecoder_ddc_nospec_revnet38_grad_filter_ada_5_100_250_3_tmp-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --sample-number 1000

rem tstnet
python inference/inference_tstnet.py ^
       runs/houston/dann-train/1/config.yaml ^
       E:/zts/IEEE_TNNLS_TSTnet/results/houston.npy ^
       --path runs/houston/tstnet/1 ^
       --nodes 1 ^
       --gpus 1 ^
       --rank-node 0 ^
       --backend gloo ^
       --master-ip localhost ^
       --master-port 9006 ^
       --sample-number 1000