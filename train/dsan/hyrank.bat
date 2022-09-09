call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem dsan
python train/ddc/train.py configs/hyrank/dsan.yaml ^
        --path ./runs/hyrank/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/dsan_2.yaml ^
        --path ./runs/hyrank/dsan_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/dsan_3.yaml ^
        --path ./runs/hyrank/dsan_3-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/dsan_5.yaml ^
        --path ./runs/hyrank/dsan_5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/dsan_05.yaml ^
        --path ./runs/hyrank/dsan_05-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

