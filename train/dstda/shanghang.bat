call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dadst
python train/dadst/train.py configs/shanghang/dadst/dadst_1_1_1_1_07_2.yaml ^
        --path ./runs/shanghang/dadst_1_1_1_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem dadst_mapping
python train/dadst/train.py configs/shanghang/dadst/dadst_mapping_1_1_1_1_07_2.yaml ^
        --path ./runs/shanghang/dadst_mapping_1_1_1_1_07_2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1