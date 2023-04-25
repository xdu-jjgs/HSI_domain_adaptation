call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem mmoe-ddc
python train/mmoe/train_ddc.py configs/hyrank/mmoe/mmoe_fe_ddc.yaml ^
        --path ./runs/hyrank/mmoe/mmoe_fe_ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem mmoe-dan
python train/mmoe/train_ddc.py configs/hyrank/mmoe/mmoe_fe_dan.yaml ^
        --path ./runs/hyrank/mmoe/mmoe_fe_dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1
