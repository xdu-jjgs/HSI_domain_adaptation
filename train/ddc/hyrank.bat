call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python train/ddc/train.py configs/hyrank/ddc.yaml ^
        --path ./runs/hyrank/ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O2

rem ddc
python train/ddc/train.py configs/hyrank/ddc_1800_average.yaml ^
        --path ./runs/hyrank_sample/ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2