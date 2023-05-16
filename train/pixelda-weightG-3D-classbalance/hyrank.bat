call "C:\Users\zzy\anaconda3\Scripts\activate.bat" base
cd E:\zzy\GAN\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/pixelda-weightG-3D-classbalance/train.py configs/hyrank/pixelda-weightG-3D-classbalance.yaml ^
        --path ./runs/hyrank/pixelda-weightG-3D-classbalance-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 1999 ^
        --seed 30 ^
        --opt-level O2