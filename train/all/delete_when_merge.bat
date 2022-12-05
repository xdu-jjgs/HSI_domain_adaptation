cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dstda\tmp.bat" %%i
)
