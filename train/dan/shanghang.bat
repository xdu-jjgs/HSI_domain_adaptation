call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem dan
python train/ddc/train.py configs/shanghang/dan_5.yaml ^
        --path ./runs/shanghang/dan_5-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O1