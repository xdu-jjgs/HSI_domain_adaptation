cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dst\houston.bat" %%i
    call "train\dst\hyrank.bat" %%i
    call "train\dst\shanghang.bat" %%i
    call "train\self_training\houston.bat" %%i
    call "train\self_training\hyrank.bat" %%i
    call "train\self_training\shanghang.bat" %%i
)