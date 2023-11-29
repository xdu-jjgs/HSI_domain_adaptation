call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsn
python train/dsn/train.py configs/hyrank/dsn/dsn.yaml ^
        --path ./runs/hyrank/dsn-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_gate
python train/dsn/train_gate.py configs/hyrank/dsn/dsn_gate.yaml ^
        --path ./runs/hyrank/dsn_gate-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_resnet34
python train/dsn/train.py configs/hyrank/dsn/dsn_inn_resnet34.yaml ^
        --path ./runs/hyrank/dsn_inn_resnet34-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet38
python train/dsn/train.py configs/hyrank/dsn/dsn_inn_revnet38.yaml ^
        --path ./runs/hyrank/dsn_inn_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet110
python train/dsn/train.py configs/hyrank/dsn/dsn_inn_revnet110.yaml ^
        --path ./runs/hyrank/dsn_inn_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet38
python train/dsn/train_gate.py configs/hyrank/dsn/dsn_inn_gate_revnet38.yaml ^
        --path ./runs/hyrank/dsn_inn_gate_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet110
python train/dsn/train_gate.py configs/hyrank/dsn/dsn_inn_gate_revnet110.yaml ^
        --path ./runs/hyrank/dsn_inn_gate_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_st
python train/dsn/train_st.py configs/hyrank/dsn/dsn_st.yaml ^
        --path ./runs/hyrank/dsn_st-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1%

rem dsn_inn_nodecoder_revnet38_st
python train/dsn/train_nodecoder_noamp_st.py configs/hyrank/dsn/dsn_inn_nodecoder_revnet38_st.yaml ^
        --path ./runs/hyrank/dsn_inn_nodecoder_revnet38_st-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1%

rem dsn_inn_nodecoder_revnet38_dst
python train/dsn/train_nodecoder_noamp_dst.py configs/hyrank/dsn/dsn_inn_nodecoder_revnet38_dst.yaml ^
        --path ./runs/hyrank/dsn_inn_nodecoder_revnet38_dst-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1%

rem dsn_inn_nodecoder_revnet38_supinfonce
python train/dsn/train_nodecoder_noamp_supinfonce.py configs/hyrank/dsn/dsn_inn_nodecoder_revnet38_supinfonce.yaml ^
        --path ./runs/hyrank/dsn_inn_nodecoder_revnet38_supinfonce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9002 ^
        --seed %~1%
