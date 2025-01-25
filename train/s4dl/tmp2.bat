call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%


for /r %%f in (configs/hyrank/s4dl/loss_hypers\*.yaml) do (
    for /l %%i in (1,1,1) do (
        python train/s4dl/train_dann.py %%f ^
            --path ./runs/hyrank_hyper2/%%~nxf ^
            --nodes 1 ^
            --gpus 1 ^
            --rank-node 0 ^
            --backend gloo ^
            --master-ip localhost ^
            --master-port 9002 ^
            --seed %%i
    )
)



