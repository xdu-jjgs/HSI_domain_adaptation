call "C:\Users\zzy\anaconda3\Scripts\activate.bat" base
cd E:\zzy\GAN\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/pixelda-weightG-caps/train.py configs/houston/pixelda-weightG-caps.yaml ^
        --path ./runs/houston/pixelda-weightG-caps-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2