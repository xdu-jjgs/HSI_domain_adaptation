call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem mmoe-ddc
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_fe2_ddc.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_fe2_dan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-jan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_fe2_jan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dann
python train/fe_mmoe/train_dann.py configs/houston/fe_mmoe/fe_mmoe_fe2_dann.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_dann-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_mmoe/train_self_training.py configs/houston/fe_mmoe/fe_mmoe_fe2_self_training_08.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_self_training_08-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-mcd
python train/fe_mmoe/train_mcd.py configs/houston/fe_mmoe/fe_mmoe_fe2_mcd.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_fe2_mcd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0
        
rem mmoe-ddc
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att2_ddc.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att2_dan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-jan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att2_jan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dann
python train/fe_mmoe/train_dann.py configs/houston/fe_mmoe/fe_mmoe_att2_dann.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_dann-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_mmoe/train_self_training.py configs/houston/fe_mmoe/fe_mmoe_att2_self_training_08.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_self_training_08-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-mcd
python train/fe_mmoe/train_mcd.py configs/houston/fe_mmoe/fe_mmoe_att2_mcd.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_mcd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-ddc
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att4_ddc.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att4_dan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_dan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-jan
python train/fe_mmoe/train_ddc.py configs/houston/fe_mmoe/fe_mmoe_att4_jan.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_jan-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dann
python train/fe_mmoe/train_dann.py configs/houston/fe_mmoe/fe_mmoe_att4_dann.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_dann-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_mmoe/train_self_training.py configs/houston/fe_mmoe/fe_mmoe_att4_self_training_08.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_self_training_08-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-mcd
python train/fe_mmoe/train_mcd.py configs/houston/fe_mmoe/fe_mmoe_att4_mcd.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att4_mcd-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dan
python train/fe_mmoe/train_ddc_var.py configs/houston/fe_mmoe/fe_mmoe_att2_dan_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_dan_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dann
python train/fe_mmoe/train_dann_var.py configs/houston/fe_mmoe/fe_mmoe_att2_dann_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_dann_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_mmoe/train_self_training_var.py configs/houston/fe_mmoe/fe_mmoe_att2_self_training_08_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_self_training_08_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-mcd
python train/fe_mmoe/train_mcd_var.py configs/houston/fe_mmoe/fe_mmoe_att2_mcd_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_mcd_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-dann
python train/fe_mmoe/train_dann_var2.py configs/houston/fe_mmoe/fe_mmoe_att2_dann_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_dann_var2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_mmoe/train_self_training_var2.py configs/houston/fe_mmoe/fe_mmoe_att2_self_training_08_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_self_training_08_var2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-mcd
python train/fe_mmoe/train_mcd_var2.py configs/houston/fe_mmoe/fe_mmoe_att2_mcd_var.yaml ^
        --path ./runs/houston/fe_mmoe/fe_mmoe_att2_mcd_var2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0
