call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem hma
python train/hma/train_ddc.py configs/shanghang/hma/hma_5.yaml ^
        --path ./runs/shanghang/hma-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O1
