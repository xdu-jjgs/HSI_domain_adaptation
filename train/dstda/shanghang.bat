call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dstda
python train/dstda/train.py configs/shanghang/dstda/dstda_1_1_1_1_07_2.yaml ^
        --path ./runs/shanghang/dstda_1_1_1_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem dstda
python train/dstda/train.py configs/shanghang/dstda/dstda_1_1_1_1_07_2_540_average.yaml ^
        --path ./runs/shanghang_sample/dstda_1_1_1_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O2

