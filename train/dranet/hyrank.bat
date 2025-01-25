call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dranet
for /l %%i in (1,1,10) do (
python train/dranet/train.py configs/hyrank/dranet/dranet.yaml ^
        --path ./runs/hyrank/dranet-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9001 ^
        --seed %%i
)