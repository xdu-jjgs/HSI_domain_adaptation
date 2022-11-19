cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\hyrank.bat" %%i
    call "train\dann\hyrank.bat" %%i
    call "train\ddc\hyrank.bat" %%i
    call "train\deepcoral\hyrank.bat" %%i
    call "train\dsan\hyrank.bat" %%i
    call "train\jan\hyrank.bat" %%i
    call "train\mcd\hyrank.bat" %%i
    call "train\nommd\hyrank.bat" %%i
    call "train\self_training\hyrank.bat" %%i
)

