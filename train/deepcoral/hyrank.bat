call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem deepcoral
python train/ddc/train.py configs/hyrank/deepcoral/deepcoral.yaml ^
        --path ./runs/hyrank/deepcoral-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed %~1% ^
        --opt-level O2

rem deepcoral
python train/ddc/train.py configs/hyrank/deepcoral/deepcoral_1800_average.yaml ^
        --path ./runs/hyrank_sample/deepcoral-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed %~1% ^
        --opt-level O2
