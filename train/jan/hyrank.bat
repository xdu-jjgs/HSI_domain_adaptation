call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem jan
python train/ddc/train.py configs/hyrank/jan.yaml ^
        --path ./runs/hyrank/jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/jan_2.yaml ^
        --path ./runs/hyrank/jan_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/jan_3.yaml ^
        --path ./runs/hyrank/jan_3-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/jan_5.yaml ^
        --path ./runs/hyrank/jan_5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/hyrank/jan_05.yaml ^
        --path ./runs/hyrank/jan_05-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

