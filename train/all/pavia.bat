cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\pavia.bat" %%i
    call "train\dann\pavia.bat" %%i
    call "train\ddc\pavia.bat" %%i
    call "train\deepcoral\pavia.bat" %%i
    call "train\dsan\pavia.bat" %%i
    call "train\jan\pavia.bat" %%i
    call "train\mcd\pavia.bat" %%i
    call "train\nommd\pavia.bat" %%i
    call "train\self_training\pavia.bat" %%i
    call "train\dsn\pavia.bat" %%i
    call "train\dst\pavia.bat" %%i
    call "train\dadst\pavia.bat" %%i
    call "train\hma\pavia.bat" %%i

)

