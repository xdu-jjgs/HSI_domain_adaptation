call "E:\zts\software\Anaconda\Scripts\activate.bat" zts
cd E:\zts\HSI_domain_adaption
set PYTHONPATH=%cd%

rem preprocess
python preprocess/preprocess.py configs/preprocess/hyrank.yaml ^
      --path E:/zzy/GAN/data/HyRANK
