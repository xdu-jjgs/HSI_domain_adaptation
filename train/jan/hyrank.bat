call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem jan
python train/ddc/train.py configs/hyrank/jan/jan.yaml ^
        --path ./runs/hyrank/jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O2

rem jan
python train/ddc/train.py configs/hyrank/jan/jan_1800_average.yaml ^
        --path ./runs/hyrank_sample/jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed %~1% ^
        --opt-level O2