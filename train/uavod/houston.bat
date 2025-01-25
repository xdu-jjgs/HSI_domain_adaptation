call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem uavod
for /l %%i in (1,1,10) do (
python train/uavod/train.py configs/houston/uavod/uavod.yaml ^
        --path ./runs/houston/uavod-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9004 ^
        --seed %%i
)

