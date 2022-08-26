rem preprocess
cd E:\zts\HSI_domain_adaption
python train.py configs/houston/ddc.yaml ^
        --path ./runs/houston/ddc-train ^
        --nodes 1 ^
        --gpus 2 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8888 ^
        --seed 30 ^
        --opt-level O1