# HSI域自适应

## 目录

- [数据集描述](#datasets)
- [支持的模型](#modelsgh)
- [用法](#usage)
  - [训练](#usage-train)
  - [测试](#usage-test)
- [结果](#result)
- [许可证](#license)

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


## <a name="models"></a> 支持的模型

- [x] DDC
- [x] DAN
- [x] DeepCORAL
- [x] JAN
- [x] DSAN
- [x] DANN
- [ ] ADDA
- [ ] CDAN
- [x] MCD
- [ ] ParetoDA
- [x] Self-training
- [x] DST
- [x] TSTNet

## <a name="usage"></a> 用法

### <a name="usage-train"></a> 训练

1. 运行 train/[model]/[dataset].bat文件
2. 或者运行如下命令

 ```shell
python train/ddc/train.py configs/houston/dan_1800_average.yaml ^
        --path ./runs/houston/dan-train ^
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

## <a name="result"></a> 结果

| Dataset          | Model         | OA-best     | backbone | sample-num | sample-order          | loss                             | loss-ratio | kernel | batch-size |
|------------------|---------------|-------------|----------|------------|-----------------------|----------------------------------|------------|--------|------------|
| Houston          | DNN           | 0.686±0.035 | fe       | -          | -                     | softmax+ce                       | 1          | -      | 64         |
| Houston          | DDC           | 0.705±0.027 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:1        | g1     | 64         |
| Houston          | DAN           | 0.694±0.048 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:1        | g5     | 64         |
| Houston          | JAN           | 0.694±0.033 | fe       | -          | -                     | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| Houston          | DSAN          | 0.664±0.108 | fe       | -          | -                     | softmax+ce, local mmd loss       | 1:1        | g5     | 64         |
| Houston          | DANN          | 0.620±0.060 | fe       | -          | -                     | softmax+ce                       | 1          | -      | 64         |
| Houston          | MCD           | 0.632±0.033 | fe       | -          | -                     | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| Houston          | Self-training | 0.652±0.003 | fe       | -          | softmax+ce, cbst loss | 1:1                              | -          | 100    |
| Houston          | DST           | 0.597±0.018 | fe       | -          | -                     | softmax+ce, wcec loss, cbst loss | 1:1:1      | -      | 100        |
| Houston          | DSTDA         | 0.593±0.013 | fe       | -          | -                     | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1      | -      | 100        |
| Houston          | TSTNet        | 0.762       | fe       | -          | -                     | softmax+ce, mmd loss, got loss   | 1:1:0.1    | g5     | 100        |
| Houston          | DNN           | 0.671±0.042 | fe       | 1260       | average               | softmax+ce                       | 1          | -      | 64         |
| Houston          | DDC           | 0.676±0.053 | fe       | 1260       | average               | softmax+ce, mmd loss             | 1:1        | g1     | 64         |
| Houston          | DAN           | 0.686±0.058 | fe       | 1260       | average               | softmax+ce, mmd loss             | 1:1        | g5     | 64         |
| Houston          | JAN           | 0.677±0.058 | fe       | 1260       | average               | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| Houston          | DSAN          | 0.643±0.050 | fe       | 1260       | average               | softmax+ce, local mmd loss       | 1:1        | g5     | 64         |
| Houston          | DANN          | 0.590±0.060 | fe       | 1260       | average               | softmax+ce                       | 1          | -      | 64         |
| Houston          | MCD           | 0.618±0.027 | fe       | 1260       | average               | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| Houston          | Self-training | 0.631±0.011 | fe       | 1260       | average               | softmax+ce, cbst loss            | 1:1        | -      | 64         |
| Houston          | DST           | 0.576±0.015 | fe       | 1260       | average               | softmax+ce, wcec loss, cbst loss | 1:1:1      | -      | 64         |
| Houston          | DSTDA         | 0.575±0.015 | fe       | 1260       | average               | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1      | -      | 64         |
| HyRANK           | DNN           | 0.507±0.023 | fe       | -          | -                     | softmax+ce                       | 1          | l      | 64         |  
| HyRANK           | DDC           | 0.523±0.030 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:1        | g1     | 64         |    
| HyRANK           | DAN           | 0.504±0.039 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:3        | g5     | 64         | 
| HyRANK           | JAN           | 0.516±0.026 | fe       | -          | -                     | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| HyRANK           | DANN          | 0.582±0.038 | fe       | -          | -                     | softmax+ce                       | 1          | -      | 64         |
| HyRANK           | MCD           | 0.561±0.026 | fe       | -          | -                     | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| HyRANK           | Self-training | 0.514±0.009 | fe       | -          | -                     | softmax+ce, cbst loss            | 1:1        | -      | 64         |
| HyRANK           | DST           | 0.558±0.021 | fe       | -          | -                     | softmax+ce, wcec loss, cbst loss | 1:1:1      | -      | 64         |
| HyRANK           | DSTDA         | 0.558±0.015 | fe       | -          | -                     | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1      | -      | 64         |
| HyRANK           | DSTDA         | 0.572±0.023 | fe       | -          | -                     | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:2:1      | -      | 64         |
| HyRANK           | TSTNet        | 0.633       | fe       | -          | -                     | softmax+ce, mmd loss, got loss   | 1:1:0.1    | l      | 100        |
| HyRANK           | DNN           | 0.492±0.029 | fe       | 1800       | average               | softmax+ce                       | 1          | l      | 64         |  
| HyRANK           | DDC           | 0.491±0.028 | fe       | 1800       | average               | softmax+ce, mmd loss             | 1:1        | g1     | 64         |    
| HyRANK           | DAN           | 0.496±0.021 | fe       | 1800       | average               | softmax+ce, mmd loss             | 1:3        | g5     | 64         | 
| HyRANK           | JAN           | 0.485±0.022 | fe       | 1800       | average               | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| HyRANK           | DANN          | 0.473±0.036 | fe       | 1800       | average               | softmax+ce                       | 1          | -      | 64         |
| HyRANK           | MCD           | 0.552±0.027 | fe       | 1800       | average               | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| HyRANK           | Self-training | 0.514±0.006 | fe       | 1800       | average               | softmax+ce, cbst loss            | 1:1        | -      | 64         |
| HyRANK           | DST           | 0.478±0.034 | fe       | 1800       | average               | softmax+ce, wcec loss, cbst loss | 1:1:1        | -      | 64         |
| HyRANK           | DSTDA         | 0.478±0.028 | fe       | 1800       | average               | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1        | -      | 64         |
| ShanghaiHangzhou | DNN           | 0.909±0.002 | fe       | -          | -                     | softmax+ce                       | 1          | -      | 64         |  
| ShanghaiHangzhou | DDC           | 0.887±0.008 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:1        | g1     | 64         |    
| ShanghaiHangzhou | DAN           | 0.904±0.011 | fe       | -          | -                     | softmax+ce, mmd loss             | 1:1        | g5     | 64         | 
| ShanghaiHangzhou | JAN           | 0.903±0.011 | fe       | -          | -                     | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| ShanghaiHangzhou | DSAN          | 0.907±0.005 | fe       | -          | -                     | softmax+ce, local mmd loss       | 1:1        | g5     | 64         |   
| ShanghaiHangzhou | DANN          | 0.905±0.016 | fe       | -          | -                     | softmax+ce                       | 1          | -      | 64         |
| ShanghaiHangzhou | MCD           | 0.717±0.105 | fe       | -          | -                     | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| ShanghaiHangzhou | Self-training | 0.915±0.000 | fe       | -          | -                     | softmax+ce, cbst loss            | 1:1        | -      | 64         |
| ShanghaiHangzhou | DST           | 0.933±0.012 | fe       | -          | -                     | softmax+ce, wcec loss, cbst loss | 1:1:1      | -      | 64         |
| ShanghaiHangzhou | DSTDA         | 0.927±0.007 | fe       | -          | -                     | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1      | -      | 64         |
| ShanghaiHangzhou | TSTNet        | 0.801       | fe       | -          | -                     | softmax+ce, mmd loss, got loss   | 1:1:0.1    | l      | 100        |
| ShanghaiHangzhou | DNN           | 0.911±0.020 | fe       | 540        | average               | softmax+ce                       | 1          | -      | 64         |  
| ShanghaiHangzhou | DDC           | 0.928±0.004 | fe       | 540        | average               | softmax+ce, mmd loss             | 1:1        | g1     | 64         |    
| ShanghaiHangzhou | DAN           | 0.913±0.011 | fe       | 540        | average               | softmax+ce, mmd loss             | 1:1        | g5     | 64         | 
| ShanghaiHangzhou | JAN           | 0.905±0.014 | fe       | 540        | average               | softmax+ce, joint mmd loss       | 1:1        | g5     | 64         |
| ShanghaiHangzhou | DSAN          | 0.901±0.013 | fe       | 540        | average               | softmax+ce, local mmd loss       | 1:1        | g5     | 64         |   
| ShanghaiHangzhou | DANN          | 0.910±0.010 | fe       | 540        | average               | softmax+ce                       | 1          | -      | 64         |
| ShanghaiHangzhou | MCD           | 0.930±0.004 | fe       | 540        | average               | softmax+ce, l1 loss              | 1:1        | -      | 64         |
| ShanghaiHangzhou | Self-training | 0.925±0.000 | fe       | 540        | average               | softmax+ce, cbst loss            | 1:1        | -      | 64         |
| ShanghaiHangzhou | DST           | 0.927±0.015 | fe       | 540        | average               | softmax+ce, wcec loss, cbst loss | 1:1:1      | -      | 64         |
| ShanghaiHangzhou | DSTDA         | 0.933±0.004 | fe       | 540        | average               | softmax+ce, wcec loss, adv loss, cbst loss | 1:1:1:1      | -      | 64         |

## <a name="license"></a> 结果

This project is released under the MIT(LICENSE) license.