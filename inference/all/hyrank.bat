call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem ddc
python inference/inference.py ^
        runs/hyrank/ddc-train/1/model_best.pth
        --path runs/hyrank/ddc-train/1 ^
