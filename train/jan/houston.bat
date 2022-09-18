call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem jan
python train/ddc/train.py configs/houston/jan.yaml ^
        --path ./runs/houston/jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/houston/jan_2.yaml ^
        --path ./runs/houston/jan_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/houston/jan_3.yaml ^
        --path ./runs/houston/jan_3-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/houston/jan_5.yaml ^
        --path ./runs/houston/jan_5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/houston/jan_05.yaml ^
        --path ./runs/houston/jan_05-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O1