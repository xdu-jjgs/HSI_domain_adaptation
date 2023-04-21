call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem nommd
python train/dadqn/train_nommd.py configs/hyrank/dqn_nommd_bs16.yaml ^
        --path ./runs/hyrank/dqn_nommd_bs16-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed 1 ^
        --opt-level O1