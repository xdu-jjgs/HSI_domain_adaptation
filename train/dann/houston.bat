call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem dann
python train/dann/train.py configs/houston/dann.yaml ^
        --path ./runs/houston/dann-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1