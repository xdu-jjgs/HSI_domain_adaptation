call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem uavod
for /l %%i in (1,1,10) do (
python train/uavod/train.py configs/hyrank/uavod/uavod.yaml ^
        --path ./runs/hyrank/uavod-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9005 ^
        --seed %%i
)
