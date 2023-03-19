call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem self_training
python train/self_training/train.py configs/hyrank/self_training_1_07.yaml ^
        --path ./runs/hyrank/self_training_1_07-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2


rem self_training
python train/self_training/train.py configs/hyrank/self_training_1_07_1800_average.yaml ^
        --path ./runs/hyrank_sample/self_training_1_07-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O2

python train/self_training/train.py configs/hyrank/self_training_1_08.yaml ^
        --path ./runs/hyrank/self_training_1_08-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2


rem self_training
python train/self_training/train.py configs/hyrank/self_training_1_08_1800_average.yaml ^
        --path ./runs/hyrank_sample/self_training_1_08-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O2

