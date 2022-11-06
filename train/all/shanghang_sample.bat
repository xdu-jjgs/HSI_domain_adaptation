call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python train/nommd/train.py configs/shanghang/nommd_540_average.yaml ^
        --path ./runs/shanghang_sample/nommd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O2
rem ddc
python train/ddc/train.py configs/shanghang/ddc_540_average.yaml ^
        --path ./runs/shanghang_sample/ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2
rem dan
python train/ddc/train.py configs/shanghang/dan_540_average.yaml ^
        --path ./runs/shanghang_sample/dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2

rem jan
python train/ddc/train.py configs/shanghang/jan_540_average.yaml ^
        --path ./runs/shanghang_sample/jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O2

rem dsan
python train/ddc/train.py configs/shanghang/dsan_540_average.yaml ^
        --path ./runs/shanghang_sample/dsan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O2

rem dann
python train/dann/train.py configs/shanghang/dann_540_average.yaml ^
        --path ./runs/shanghang_sample/dann-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 30 ^
        --opt-level O2

rem mcd
python train/mcd/train.py configs/shanghang/mcd_540_average.yaml ^
        --path ./runs/shanghang_sample/mcd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2

rem self_training
python train/self_training/train.py configs/shanghang/self_training_1_05_540_average.yaml ^
        --path ./runs/shanghang_sample/self_training_1_05-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2

rem deepcoral
python train/ddc/train.py configs/shanghang/deepcoral_540_average.yaml ^
        --path ./runs/shanghang_sample/deepcoral-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8826 ^
        --seed 30 ^
        --opt-level O2
