call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dd_mmoe
python train/dd_mmoe/train.py configs/shanghang/dd_mmoe/dd_mmoe.yaml ^
        --path ./runs/shanghang/dd_mmoe-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1