call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
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
        --opt-level O2

rem deepcoral
python train/ddc/train.py configs/shanghang/deepcoral_540_average.yaml ^
        --path ./runs/shanghang_sample/deepcoral-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O2