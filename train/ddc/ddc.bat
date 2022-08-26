call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
rem ddc
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%
python train/ddc/train.py configs/houston/ddc.yaml ^
        --path ./runs/houston/ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O1