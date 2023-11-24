call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem dann
python train/dann/train.py configs/indiana/dann/dann_lr1e5.yaml ^
        --path ./runs/indiana/dann_lr1e5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 9006 ^
        --seed %~1%

