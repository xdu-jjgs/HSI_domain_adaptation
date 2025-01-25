call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%
SET CUDA_VISIBLE_DEVICES=1


for /r %%f in (configs/shanghang/s4dl/loss_hypers\*.yaml) do (
    for /l %%i in (1,1,1) do (
        python train/s4dl/train_dann.py %%f ^
            --path ./runs/shanghang_hyper2/%%~nxf ^
            --nodes 1 ^
            --gpus 1 ^
            --rank-node 0 ^
            --backend gloo ^
            --master-ip localhost ^
            --master-port 9003 ^
            --seed %%i
    )
)
