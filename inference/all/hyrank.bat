call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python inference/inference_ddc.py ^
        runs/hyrank/nommd-train/1/config.yaml ^
        runs/hyrank/nommd-train/1/best.pth ^
        --path runs/hyrank/nommd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem ddc
python inference/inference_ddc.py ^
        runs/hyrank/ddc-train/1/config.yaml ^
        runs/hyrank/ddc-train/1/best.pth ^
        --path runs/hyrank/ddc-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem dan
python inference/inference_ddc.py ^
        runs/hyrank/dan-train/1/config.yaml ^
        runs/hyrank/dan-train/1/best.pth ^
        --path runs/hyrank/dan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem jan
python inference/inference_ddc.py ^
        runs/hyrank/jan-train/1/config.yaml ^
        runs/hyrank/jan-train/1/best.pth ^
        --path runs/hyrank/jan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem dsan
python inference/inference_ddc.py ^
        runs/hyrank/dsan-train/1/config.yaml ^
        runs/hyrank/dsan-train/1/best.pth ^
        --path runs/hyrank/dsan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem dann
python inference/inference_dann.py ^
        runs/hyrank/dann-train/1/config.yaml ^
        runs/hyrank/dann-train/1/best.pth ^
        --path runs/hyrank/dann-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem mcd
python inference/inference_mcd.py ^
        runs/hyrank/mcd-train/1/config.yaml ^
        runs/hyrank/mcd-train/1/best.pth ^
        --path runs/hyrank/mcd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000


rem dst
python inference/inference_dst.py ^
        runs/hyrank/dst_1_1_1_07_2-train/1/config.yaml ^
        runs/hyrank/dst_1_1_1_07_2-train/1/best.pth ^
        --path runs/hyrank/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --sample-number 1000

rem dsn
python inference/inference_dsn.py ^
        runs/hyrank/dsn_log-train/1/config.yaml ^
        runs/hyrank/dsn_log-train/1/best.pth ^
        --path runs/hyrank/dsn_log-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --sample-number 1000

rem tstnet
python inference/inference_tstnet.py ^
       runs/hyrank/dann-train/1/config.yaml ^
       E:/zts/IEEE_TNNLS_TSTnet/results/hyrank.npy ^
       --path runs/hyrank/tstnet/1 ^
       --nodes 1 ^
       --gpus 1 ^
       --rank-node 0 ^
       --backend gloo ^
       --master-ip localhost ^
       --master-port 9006 ^
       --sample-number 1000

rem s4dl
python inference/inference_dsn_rev.py ^
        runs/hyrank/dsn_rev_nodecoder_nospec_revnet38_grad_filter_ada_10_100_250_9-train/1/config.yaml ^
        runs/hyrank/dsn_rev_nodecoder_nospec_revnet38_grad_filter_ada_10_100_250_9_test-train/1/best.pth ^
        --path runs/hyrank/dsn_rev_nodecoder_nospec_revnet38_grad_filter_ada_10_100_250_9_test-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9005 ^
        --sample-number 1000
