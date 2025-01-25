call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem sad
for /l %%i in (1,1,10) do (
python train/sad/train.py configs/shanghang/sad/sad.yaml ^
        --path ./runs/shanghang/sad-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9003 ^
        --seed %%i
)