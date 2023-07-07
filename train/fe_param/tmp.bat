call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem mmoe-dan
python train/fe_param/train_ddc_var.py configs/houston/fe_param/fe_param_att2_dan_var.yaml ^
        --path ./runs/houston/fe_param/fe_param_att2_dan_var-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0

rem mmoe-self_training
python train/fe_param/train_self_training_var2.py configs/houston/fe_param/fe_param_att2_self_training_08_var.yaml ^
        --path ./runs/houston/fe_param/fe_param_att2_self_training_08_var2-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8890 ^
        --seed %~1% ^
        --opt-level O0