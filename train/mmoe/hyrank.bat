call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/ddc/train.py configs/hyrank/mmoe/mmoe_fe_ddc.yaml ^
        --path ./runs/hyrank/mmoe/mmoe_fe_ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2
