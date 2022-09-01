call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%
set CUDA_VISIBLE_DEVICES=1

rem deepcoral
python train/ddc/train.py configs/hyrank/deepcoral.yaml ^
        --path ./runs/hyrank/deepcoral1-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2