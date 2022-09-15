call "C:\Users\zzy\anaconda3\Scripts\activate.bat" base
cd E:\zzy\GAN\HSI_domain_adaptation
set PYTHONPATH=%cd%

rem preprocess
python preprocess/preprocess.py configs/preprocess/houston.yaml ^
      --path dataset/houston_preprocessed
