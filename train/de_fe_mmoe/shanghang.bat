call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem de_fe_mmoe_dann
python train/de_fe_mmoe/de_fe_mmoe_dann_var.py configs/shanghang/de_fe_mmoe/de_fe_resnet_mmoe_dann_var.yaml ^
        --path ./runs/shanghang/de_fe_resnet_mmoe_dann_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1

rem de_fe_mmoe_dadst
python train/de_fe_mmoe/de_fe_mmoe_dadst_var.py configs/shanghang/de_fe_mmoe/de_fe_resnet_mmoe_dadst_var.yaml ^
        --path ./runs/shanghang/de_fe_resnet_mmoe_dadst_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1

rem de_fe_mmoe_dadst
python train/de_fe_mmoe/de_fe_mmoe_dadst_var.py configs/shanghang/de_fe_mmoe/de_fe_resnet_mmoe_dadst_gate_conv_var.yaml ^
        --path ./runs/shanghang/de_fe_resnet_mmoe_dadst_gate_conv_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed %~1% ^
        --opt-level O1
