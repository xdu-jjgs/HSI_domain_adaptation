call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem de_fe_mmoe
python train/de_fe_mmoe/de_fe_mmoe_dann_var.py configs/houston/de_fe_mmoe/de_fe_resnet_mmoe_dann_var.yaml ^
        --path ./runs/houston/de_fe_mmoe_dann_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1
