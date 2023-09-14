call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dd_fe_mmoe
python train/dd_fe_mmoe/train.py configs/houston/dd_fe_mmoe/dd_fe_mmoe.yaml ^
        --path ./runs/houston/dd_fe_mmoe-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1
