call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dstda_mapping
python train/dstda/train.py configs/hyrank/dstda/dstda_mapping_1_1_05_1_07_2.yaml ^
        --path ./runs/hyrank/dstda_mapping_1_1_05_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem dstda_mapping
python train/dstda/train.py configs/hyrank/dstda/dstda_mapping_1_1_1_1_07_2.yaml ^
        --path ./runs/hyrank/dstda_mapping_1_1_1_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem dstda_mapping
python train/dstda/train.py configs/hyrank/dstda/dstda_mapping_1_1_2_1_07_2.yaml ^
        --path ./runs/hyrank/dstda_mapping_1_1_2_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0