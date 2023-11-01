cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\indiana.bat" %%i
    call "train\dann\indiana.bat" %%i
    call "train\ddc\indiana.bat" %%i
    call "train\deepcoral\indiana.bat" %%i
    call "train\dsan\indiana.bat" %%i
    call "train\jan\indiana.bat" %%i
    call "train\mcd\indiana.bat" %%i
    call "train\nommd\indiana.bat" %%i
    call "train\self_training\indiana.bat" %%i
    call "train\dsn\indiana.bat" %%i
    call "train\dst\indiana.bat" %%i
    call "train\dadst\indiana.bat" %%i
    call "train\hma\indiana.bat" %%i

)

