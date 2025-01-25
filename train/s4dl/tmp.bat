call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dsn
for /r %%f in (configs/houston/s4dl/loss_hypers\*.yaml) do (
    for /l %%i in (1,1,1) do (
        python train/s4dl/train_ts.py %%f ^
            --path ./runs/houston_hyper2/%%~nxf ^
            --nodes 1 ^
            --gpus 1 ^
            --rank-node 0 ^
            --backend gloo ^
            --master-ip localhost ^
            --master-port 9001 ^
            --seed %%i
    )
)
