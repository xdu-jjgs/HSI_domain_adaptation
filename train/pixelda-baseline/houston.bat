call "C:\Users\zzy\anaconda3\Scripts\activate.bat" base
cd E:\zzy\GAN\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/pixelda-baseline/train.py configs/houston/pixelda-baseline.yaml ^
        --path ./runs/houston/pixelda-baseline-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2