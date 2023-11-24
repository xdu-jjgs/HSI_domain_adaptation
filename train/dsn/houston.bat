call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsn
python train/dsn/train.py configs/houston/dsn/dsn.yaml ^
        --path ./runs/houston/dsn-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_gate
python train/dsn/train_gate.py configs/houston/dsn/dsn_gate.yaml ^
        --path ./runs/houston/dsn_gate-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_resnet34
python train/dsn/train.py configs/houston/dsn/dsn_inn_resnet34.yaml ^
        --path ./runs/houston/dsn_inn_resnet34-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet38
python train/dsn/train.py configs/houston/dsn/dsn_inn_revnet38.yaml ^
        --path ./runs/houston/dsn_inn_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet110
python train/dsn/train.py configs/houston/dsn/dsn_inn_revnet110.yaml ^
        --path ./runs/houston/dsn_inn_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet38
python train/dsn/train_gate.py configs/houston/dsn/dsn_inn_gate_revnet38.yaml ^
        --path ./runs/houston/dsn_inn_gate_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet110
python train/dsn/train_gate.py configs/houston/dsn/dsn_inn_gate_revnet110.yaml ^
        --path ./runs/houston/dsn_inn_gate_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_nodecoder
python train/dsn/train_nodecoder.py configs/houston/dsn/dsn_nodecoder.yaml ^
        --path ./runs/houston/dsn_nodecoder-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_nodecoder_revnet38_dst
python train/dsn/train_nodecoder_noamp_supinfonce.py configs/hyrank/dsn/dsn_inn_nodecoder_revnet38_supinfonce.yaml ^
        --path ./runs/hyrank/dsn_inn_nodecoder_revnet38_supinfonce-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1%
