call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsan
python train/ddc/train.py configs/shanghang/dsan.yaml ^
        --path ./runs/shanghang/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O2

rem dsan
python train/ddc/train.py configs/shanghang/dsan_540_average.yaml ^
        --path ./runs/shanghang_sample/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O2