call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem preprocess
python preprocess/preprocess.py configs/preprocess/houston.yaml ^
      --path E:/zts/dataset/houston_preprocessed
