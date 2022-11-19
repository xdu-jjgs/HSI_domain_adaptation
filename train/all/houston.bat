cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\houston.bat" %%i
    call "train\dann\houston.bat" %%i
    call "train\ddc\houston.bat" %%i
    call "train\deepcoral\houston.bat" %%i
    call "train\dsan\houston.bat" %%i
    call "train\jan\houston.bat" %%i
    call "train\mcd\houston.bat" %%i
    call "train\nommd\houston.bat" %%i
    call "train\self_training\houston.bat" %%i
)