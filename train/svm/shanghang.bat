call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

for /l %%i in (1,1,5) do (
    python train/svm/train.py configs/shanghang/nommd/nommd.yaml ^
        --path ./runs/shanghang/svm-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %%i

)
