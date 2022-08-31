call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
rem ddc
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%
python train/dsan/train.py configs/houston/dsan.yaml ^
        --path ./runs/houston/dsan-train ^
        --nodes 1 ^
        --gpus 2 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8892 ^
        --seed 30 ^
        --opt-level O2