call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsn
python train/dsn/train.py configs/shanghang/dsn/dsn.yaml ^
        --path ./runs/shanghang/dsn-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_gate
python train/dsn/train_gate.py configs/shanghang/dsn/dsn_gate.yaml ^
        --path ./runs/shanghang/dsn_gate-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_resnet34
python train/dsn/train.py configs/shanghang/dsn/dsn_inn_resnet34.yaml ^
        --path ./runs/shanghang/dsn_inn_resnet34-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet38
python train/dsn/train.py configs/shanghang/dsn/dsn_inn_revnet38.yaml ^
        --path ./runs/shanghang/dsn_inn_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_revnet110
python train/dsn/train.py configs/shanghang/dsn/dsn_inn_revnet110.yaml ^
        --path ./runs/shanghang/dsn_inn_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet38
python train/dsn/train_gate.py configs/shanghang/dsn/dsn_inn_gate_revnet38.yaml ^
        --path ./runs/shanghang/dsn_inn_gate_revnet38-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1

rem dsn_inn_gate_revnet110
python train/dsn/train_gate.py configs/shanghang/dsn/dsn_inn_gate_revnet110.yaml ^
        --path ./runs/shanghang/dsn_inn_gate_revnet110-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %~1% ^
        --opt-level O1
