call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem deepcoral
python train/ddc/train.py configs/shanghang/deepcoral.yaml ^
        --path ./runs/shanghang/deepcoral-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/shanghang/deepcoral_2.yaml ^
        --path ./runs/shanghang/deepcoral_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/shanghang/deepcoral_3.yaml ^
        --path ./runs/shanghang/deepcoral_3-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/shanghang/deepcoral_5.yaml ^
        --path ./runs/shanghang/deepcoral_5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1

python train/ddc/train.py configs/shanghang/deepcoral_05.yaml ^
        --path ./runs/shanghang/deepcoral_05-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1