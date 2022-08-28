call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
rem dan
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%
python train/nommd/train.py configs/houston/nommd.yaml ^
        --path ./runs/houston/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O2