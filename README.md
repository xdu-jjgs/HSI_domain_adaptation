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
- [ ] ADAA
- [ ] CDAN
- [x] MCD
- [ ] ParetoDA
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

| Dataset          | Model  | backbone | sample-num | sample-order | loss                           | loss-ratio | kernel | batch-size | OA-best |
|------------------|--------|----------|------------|--------------|--------------------------------|------------|--------|------------|---------|
| Houston          | DNN    | fe       | -          | -            | softmax+ce                     | 1          | -      | 64         | 0.720   |
| Houston          | DDC    | fe       | -          | -            | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.706   |
| Houston          | DAN    | fe       | -          | -            | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.672   |
| Houston          | JAN    | fe       | -          | -            | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.622   |
| Houston          | DSAN   | fe       | -          | -            | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.444   |
| Houston          | DANN   | fe       | -          | -            | softmax+ce                     | 1          | -      | 64         | 0.573   |
| Houston          | MCD    | fe       | -          | -            | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.573   | 
| Houston          | TSTNet | fe       | -          | -            | softmax+ce, mmd loss, got loss | 1:1:0.1    | g5     | 100        | 0.762   |
| Houston          | DNN    | fe       | 1260       | average      | softmax+ce                     | 1          | -      | 64         | 0.685   |
| Houston          | DDC    | fe       | 1260       | average      | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.690   |
| Houston          | DAN    | fe       | 1260       | average      | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.647   |
| Houston          | JAN    | fe       | 1260       | average      | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.440   |
| Houston          | DSAN   | fe       | 1260       | average      | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.104   |
| Houston          | DANN   | fe       | 1260       | average      | softmax+ce                     | 1          | -      | 64         | 0.608   |
| Houston          | MCD    | fe       | 1260       | average      | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.614   | 
|
| HyRANK           | DNN    | fe       | -          | -            | softmax+ce                     | 1          | l      | 64         | 0.538   |  
| HyRANK           | DDC    | fe       | -          | -            | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.544   |    
| HyRANK           | DAN    | fe       | -          | -            | softmax+ce, mmd loss           | 1:3        | g5     | 64         | 0.532   | 
| HyRANK           | JAN    | fe       | -          | -            | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.520   |
| HyRANK           | DANN   | fe       | -          | -            | softmax+ce                     | 1          | -      | 64         | 0.615   |
| HyRANK           | MCD    | fe       | -          | -            | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.544   |
| HyRANK           | TSTNet | fe       | -          | -            | softmax+ce, mmd loss, got loss | 1:1:0.1    | l      | 100        | 0.633   |
|
| HyRANK           | DNN    | fe       | 1800       | average      | softmax+ce                     | 1          | l      | 64         | 0.470   |  
| HyRANK           | DDC    | fe       | 1800       | average      | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.469   |    
| HyRANK           | DAN    | fe       | 1800       | average      | softmax+ce, mmd loss           | 1:3        | g5     | 64         | 0.436   | 
| HyRANK           | JAN    | fe       | 1800       | average      | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.490   |
| HyRANK           | DANN   | fe       | 1800       | average      | softmax+ce                     | 1          | -      | 64         | 0.422   |
| HyRANK           | MCD    | fe       | 1800       | average      | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.436   |
|
| ShanghaiHangzhou | DNN    | fe       | -          | -            | softmax+ce                     | 1          | -      | 64         | 0.929   |  
| ShanghaiHangzhou | DDC    | fe       | -          | -            | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.927   |    
| ShanghaiHangzhou | DAN    | fe       | -          | -            | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.916   | 
| ShanghaiHangzhou | JAN    | fe       | -          | -            | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.921   |
| ShanghaiHangzhou | DSAN   | fe       | -          | -            | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.958   |   
| ShanghaiHangzhou | DANN   | fe       | -          | -            | softmax+ce                     | 1          | -      | 64         | 0.903   |
| ShanghaiHangzhou | MCD    | fe       | -          | -            | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.901   |
| ShanghaiHangzhou | TSTNet | fe       | -          | -            | softmax+ce, mmd loss, got loss | 1:1:0.1    | l      | 100        | 0.801   |
|
| ShanghaiHangzhou | DNN    | fe       | 540        | average      | softmax+ce                     | 1          | -      | 64         | 0.904   |  
| ShanghaiHangzhou | DDC    | fe       | 540        | average      | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.885   |    
| ShanghaiHangzhou | DAN    | fe       | 540        | average      | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.393   | 
| ShanghaiHangzhou | JAN    | fe       | 540        | average      | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.505   |
| ShanghaiHangzhou | DSAN   | fe       | 540        | average      | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.523   |   
| ShanghaiHangzhou | DANN   | fe       | 540        | average      | softmax+ce                     | 1          | -      | 64         | 0.921   |
| ShanghaiHangzhou | MCD    | fe       | 540        | average      | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.921   |


## <a name="license"></a> 结果

This project is released under the MIT(LICENSE) license.