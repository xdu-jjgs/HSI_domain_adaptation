call "C:\Users\zzy\anaconda3\Scripts\activate.bat" base
cd E:\zzy\GAN\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/pixelda-weightG-classbalance/train.py configs/houston/pixelda-weightG-classbalance.yaml ^
        --path ./runs/houston/pixelda-weightG-classbalance-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 1999 ^
        --seed 30 ^
        --opt-level O2