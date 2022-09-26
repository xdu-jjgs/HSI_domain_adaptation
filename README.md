# HSI域自适应

## 目录

- [数据集描述](#a-namedatasetsa-)
- [数据预处理](#a-namepreprocessa-)
- [支持的模型](#a-namemodelsa-)
- [用法](#a-nameusagea-)
    - [训练](#a-nameusage-traina-)
    - [测试](#a-nameusage-testa-)
- [试验记录](#a-nameresulta-)
- [许可证](#a-namelicensea-)

## <a name="datasets"></a> 数据集描述

### <a name="datasets-houston"></a> Houston数据集

| 类别    | 名称                        | Houston13      | Houston18      |
|-------|---------------------------|----------------|----------------|
| 1     | Grass healthy             | 345            | 1353           |
| 2     | Grass stressed            | 365            | 4888           |
| 3     | Trees                     | 365            | 2766           |
| 4     | Water                     | 285            | 22             |
| 5     | Residential buildings     | 319            | 5347           |
| 6     | Non-residential buildings | 408            | 32459          |
| 7     | Road                      | 443            | 6365           |
| total | total                     | 2530           | 53200          |
| shape | N * H * C                 | 210 * 954 * 48 | 210 * 954 * 48 |

### <a name="datasets-hyrank"></a> HyRANK数据集

| 类别    | 名称                       | Dioni            | Loukia          |
|-------|--------------------------|------------------|-----------------|
| 1     | Dense urban fabric       | 1262             | 288             |
| 2     | Mineral extraction sites | 204              | 67              |
| 3     | Non irrigated land       | 614              | 542             |
| 4     | Fruit trees              | 150              | 79              |
| 5     | Olive Groves             | 1768             | 1401            |
| 6     | Coniferous Forest        | 361              | 900             |
| 7     | Dense Vegetation         | 5035             | 3793            |
| 8     | Sparce Vegetation        | 6374             | 2803            |
| 9     | Sparce Areas             | 1754             | 404             |
| 10    | Rocks and Sand           | 492              | 487             |
| 11    | Water                    | 1612             | 1393            |
| 12    | Coastal Water            | 398              | 451             |
| total | total                    | 20024            | 12208           |  
| shape | N * H * C                | 250 * 1376 * 176 | 249 * 945 * 176 |  

### <a name="datasets-shanghang"></a> ShanghaiHangzhou数据集

| 类别    | 名称            | Shanghai         | Hangzhou        |
|-------|---------------|------------------|-----------------|
| 1     | Water         | 18043            | 123123          |
| 2     | Land/Building | 77450            | 161689          |
| 3     | Plant         | 40207            | 83188           |
| total | total         | 135700           | 368000          |  
| shape | N * H * C     | 1600 * 260 * 198 | 590 * 230 * 198 |

## <a name="preprocess"></a> 数据预处理

包括Z-Score归一化、图像裁剪、筛选类别和调整标签等

1. Houston数据集

```shell
python preprocess/preprocess.py configs/preprocess/houston.yaml ^
      --path E:/zts/dataset/houston_preprocessed
```

2. HyRANK数据集

```shell
python preprocess/preprocess.py configs/preprocess/hyrank.yaml ^
      --path E:/zts/dataset/hyrank_preprocessed
```

3. ShanghaiHangzhou数据集

```shell
python preprocess/preprocess.py configs/preprocess/shanghang.yaml ^
      --path E:/zts/dataset/shanghaihangzhou_preprocessed
```

## <a name="models"></a> 支持的模型

- [x] DDC
- [x] DAN
- [ ] DeepCORAL
- [x] DSAN
- [x] DANN
- [ ] ADAA
- [ ] CDAN
- [x] MCD
- [ ] ParetoDA
- [ ] TSTNet

## <a name="usage"></a> 用法

### <a name="usage-train"></a> 训练

1. 运行 train/[model]/[dataset].bat文件
2. 或者运行如下命令

 ```shell
python train/ddc/train.py configs/houston/ddc.yaml ^
        --path ./runs/houston/ddc-train ^
        --nodes 1 ^
        --gpus 1 ^
        --rank-node 0 ^
        --backend gloo ^
        --master-ip localhost ^
        --master-port 8886 ^
        --seed 30 ^
        --opt-level O2
```

### <a name="usage-test"></a> 测试

验证集等于测试集，无需再另行测试

## <a name="result"></a> 试验记录

| Dataset          | Model  | loss                           | loss-ratio | kernel | batch-size | OA-best | OA-worst |
|------------------|--------|--------------------------------|------------|--------|------------|---------|----------|
| Houston          | MCD    | softmax+ce, discrepancy        | -          | l      | 64        | 0.633   | 0.608    |
| Houston          | DANN   | softmax+ce                     | -          | l      | 64        | 0.633   | 0.608    |
| Houston          | PixelDA| softmax+ce                     | -          | l      | 64        | 0.633   | 0.608    |
| HyRANK           | MCD    | softmax+ce, discrepancy        | -          | l      | 64        | 0.633   | 0.608    |
| HyRANK           | DANN   | softmax+ce                     | -          | l      | 64        | 0.633   | 0.608    |
| HyRANK           | PixelDA| softmax+ce                     | -          | l      | 64        | 0.633   | 0.608    |


## <a name="license"></a> 许可证

This project is released under the MIT(LICENSE) license.