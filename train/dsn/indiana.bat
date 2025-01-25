call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsn
python train/dsn/train.py configs/indiana/dsn/dsn.yaml ^
        --path ./runs/indiana/dsn-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% 

rem dsn_gate
python train/dsn/train_gate.py configs/indiana/dsn/dsn_gate.yaml ^
        --path ./runs/indiana/dsn_gate-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% 