cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dstda\houston.bat" %%i
    call "train\dstda\hyrank.bat" %%i
    call "train\dstda\shanghang.bat" %%i
)