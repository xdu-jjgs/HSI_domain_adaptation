call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsan
python train/ddc/train.py configs/houston/dsan/dsan.yaml ^
        --path ./runs/houston/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed %~1% ^
        --opt-level O2

rem dsan
python train/ddc/train.py configs/houston/dsan/dsan_1260_average.yaml ^
        --path ./runs/houston_sample/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed %~1% ^
        --opt-level O2
