call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python inference/inference_ddc.py ^
        runs/shanghang/nommd-train/1/config.yaml ^
        runs/shanghang/nommd-train/1/best.pth ^
        --path runs/shanghang/nommd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem ddc
python inference/inference_ddc.py ^
        runs/shanghang/ddc-train/1/config.yaml ^
        runs/shanghang/ddc-train/1/best.pth ^
        --path runs/shanghang/ddc-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem dan
python inference/inference_ddc.py ^
        runs/shanghang/dan-train/1/config.yaml ^
        runs/shanghang/dan-train/1/best.pth ^
        --path runs/shanghang/dan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem jan
python inference/inference_ddc.py ^
        runs/shanghang/jan-train/1/config.yaml ^
        runs/shanghang/jan-train/1/best.pth ^
        --path runs/shanghang/jan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem dsan
python inference/inference_ddc.py ^
        runs/shanghang/dsan-train/1/config.yaml ^
        runs/shanghang/dsan-train/1/best.pth ^
        --path runs/shanghang/dsan-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem dann
python inference/inference_dann.py ^
        runs/shanghang/dann-train/1/config.yaml ^
        runs/shanghang/dann-train/1/best.pth ^
        --path runs/shanghang/dann-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem mcd
python inference/inference_mcd.py ^
        runs/shanghang/mcd-train/1/config.yaml ^
        runs/shanghang/mcd-train/1/best.pth ^
        --path runs/shanghang/mcd-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem dst
python inference/inference_dst.py ^
        runs/shanghang/dst_1_1_1_07_2-train/1/config.yaml ^
        runs/shanghang/dst_1_1_1_07_2-train/1/best.pth ^
        --path runs/shanghang/dst_1_1_1_07_2-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --sample-number 1000 

rem dsn
python inference/inference_dsn.py ^
        runs/shanghang/dsn_log-train/1/config.yaml ^
        runs/shanghang/dsn_log-train/1/best.pth ^
        --path runs/shanghang/dsn_log-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --sample-number 1000



rem s4dl
python inference/inference_dsn_rev_st.py ^
        runs/shanghang/dsn_rev_nodecoder_nospec_revnet38_st_grad_filter_ada_ema_0_100_250_18_2_1_tmp-train/1/config.yaml ^
        runs/shanghang/dsn_rev_nodecoder_nospec_revnet38_st_grad_filter_ada_ema_0_100_250_18_2_1_tmp-train/1/best.pth ^
        --path runs/shanghang/dsn_rev_nodecoder_nospec_revnet38_st_grad_filter_ada_ema_0_100_250_18_2_1_tmp-train/1 ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9005 ^
        --sample-number 1000

rem tstnet
python inference/inference_tstnet.py ^
       runs/shanghang/dann-train/1/config.yaml ^
       E:/zts/IEEE_TNNLS_TSTnet/results/shanghang.npy ^
       --path runs/shanghang/tstnet/1 ^
       --nodes 1 ^
       --gpus 1 ^
       --rank-node 0 ^
       --backend gloo ^
       --master-ip localhost ^
       --master-port 9006 ^
       --sample-number 1000