call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
rem dan
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%
set CUDA_VISIBLE_DEVICES=1
python train/ddc/train.py configs/houston/dan.yaml ^
        --path ./runs/houston/dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2