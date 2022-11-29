cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\self_training\delete_when_merge.bat" %%i
)