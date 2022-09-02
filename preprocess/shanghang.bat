call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem preprocess
python preprocess/preprocess.py configs/preprocess/shanghang.yaml ^
      --path E:/zts/dataset/shanghaihangzhou_preprocessed
