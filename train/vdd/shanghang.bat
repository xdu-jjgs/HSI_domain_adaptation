call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem vdd
python train/vdd/train_os.py configs/shanghang/vdd/vdd_os.yaml ^
        --path ./runs/shanghang/vdd_os-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1

rem vdd
python train/vdd/train_ts.py configs/shanghang/vdd/vdd_ts.yaml ^
        --path ./runs/shanghang/vdd_ts-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1
