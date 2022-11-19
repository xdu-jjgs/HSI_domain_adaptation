cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\shanghang.bat" %%i
    call "train\dann\shanghang.bat" %%i
    call "train\ddc\shanghang.bat" %%i
    call "train\deepcoral\shanghang.bat" %%i
    call "train\dsan\shanghang.bat" %%i
    call "train\jan\shanghang.bat" %%i
    call "train\mcd\shanghang.bat" %%i
    call "train\nommd\shanghang.bat" %%i
    call "train\self_training\shanghang.bat" %%i
)