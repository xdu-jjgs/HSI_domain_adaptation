call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dranet
for /l %%i in (1,1,5) do (
python train/dranet/train.py configs/shanghang/dranet/dranet.yaml ^
        --path ./runs/shanghang/dranet-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --seed %%i
)