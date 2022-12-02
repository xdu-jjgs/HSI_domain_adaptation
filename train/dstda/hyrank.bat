call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dstda
python train/dstda/train.py configs/hyrank/dstda_1_07_2.yaml ^
        --path ./runs/hyrank/dstda_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2


remdstda
python train/dstda/train.py configs/hyrank/dstda_1_07_2_1800_average.yaml ^
        --path ./runs/hyrank_sample/dstda_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O2
