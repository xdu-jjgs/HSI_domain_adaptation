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
- [ ] DeepCORAL
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
实验要重新跑。。。

| Dataset          | Model  | loss                           | loss-ratio | kernel | batch-size | OA-best | OA-worst |
|------------------|--------|--------------------------------|------------|--------|------------|---------|----------|
| Houston          | DNN    | softmax+ce                     | 1          | -      | 64         | 0.739   | 0.738    |
| Houston          | DDC    | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.719   | 0.693    |
| Houston          | DDC    | softmax+ce, mmd loss           | 1:2        | g1     | 64         | 0.733   | 0.675    |
| Houston          | DDC    | softmax+ce, mmd loss           | 1:3        | g1     | 64         | 0.704   | 0.674    |
| Houston          | DAN    | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.725   | 0.655    |
| Houston          | DAN    | softmax+ce, mmd loss           | 1:2        | g5     | 64         | 0.731   | 0.595    |
| Houston          | DAN    | softmax+ce, mmd loss           | 1:3        | g5     | 64         | 0.683   | 0.670    |
| Houston          | JAN    | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.742   | -        | 
| Houston          | JAN    | softmax+ce, joint mmd loss     | 1:0.5      | g5     | 64         | 0.700   | -        | 
| Houston          | JAN    | softmax+ce, joint mmd loss     | 1:2        | g5     | 64         | 0.676   | -        | 
| Houston          | JAN    | softmax+ce, joint mmd loss     | 1:3        | g5     | 64         | 0.700   | -        | 
| Houston          | JAN    | softmax+ce, joint mmd loss     | 1:5        | g5     | 64         | 0.638   | -        | 
| Houston          | DSAN   | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.738   | -        | 
| Houston          | DSAN   | softmax+ce, local mmd loss     | 1:0.5      | g5     | 64         | 0.699   | -        | 
| Houston          | DSAN   | softmax+ce, local mmd loss     | 1:2        | g5     | 64         | 0.633   | -        | 
| Houston          | DSAN   | softmax+ce, local mmd loss     | 1:3        | g5     | 64         | 0.619   | -        | 
| Houston          | DSAN   | softmax+ce, local mmd loss     | 1:5        | g5     | 64         | 0.615   | -        | 
| Houston          | DANN   | softmax+ce                     | 1          | -      | 64         | 0.581   | -        |
| Houston          | MCD    | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.722   | -        |       
| Houston          | TSTNet | softmax+ce, mmd loss, got loss | 1:1:0.1    | g5     | 100        | 0.762   | -        |       
| HyRANK           | DNN    | softmax+ce                     | 1          | l      | 64         | 0.506   | 0.504    |   
| HyRANK           | DDC    | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.507   | 0.503    |   
| HyRANK           | DDC    | softmax+ce, mmd loss           | 1:2        | g1     | 64         | 0.501   | 0.482    |   
| HyRANK           | DDC    | softmax+ce, mmd loss           | 1:3        | g1     | 64         | 0.513   | 0.496    |   
| HyRANK           | DAN    | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.524   | 0.517    |   
| HyRANK           | DAN    | softmax+ce, mmd loss           | 1:2        | g5     | 64         | 0.507   | 0.487    |   
| HyRANK           | DAN    | softmax+ce, mmd loss           | 1:3        | g5     | 64         | 0.507   | 0.487    |  
| HyRANK           | JAN    | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.528   | -        | 
| HyRANK           | JAN    | softmax+ce, joint mmd loss     | 1:0.5      | g5     | 64         | 0.499   | -        | 
| HyRANK           | JAN    | softmax+ce, joint mmd loss     | 1:2        | g5     | 64         | 0.511   | -        | 
| HyRANK           | JAN    | softmax+ce, joint mmd loss     | 1:3        | g5     | 64         | 0.507   | -        | 
| HyRANK           | JAN    | softmax+ce, joint mmd loss     | 1:5        | g5     | 64         | 0.490   | -        | 
| HyRANK           | DANN   | softmax+ce                     | 1          | -      | 64         | 0.625   | -        |
| HyRANK           | MCD    | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.549   | -        | 
| HyRANK           | TSTNet | softmax+ce, mmd loss, got loss | 1:1:0.1    | l      | 100        | 0.633   | 0.608    |
| ShanghaiHangzhou | DNN    | softmax+ce                     | 1          | -      | 64         | 0.921   | -        |   
| ShanghaiHangzhou | DDC    | softmax+ce, mmd loss           | 1:1        | g1     | 64         | 0.929   | -        |   
| ShanghaiHangzhou | DDC    | softmax+ce, mmd loss           | 1:2        | g1     | 64         | 0.942   | -        |   
| ShanghaiHangzhou | DDC    | softmax+ce, mmd loss           | 1:3        | g1     | 64         | 0.924   | -        |   
| ShanghaiHangzhou | DAN    | softmax+ce, mmd loss           | 1:1        | g5     | 64         | 0.928   | -        |   
| ShanghaiHangzhou | DAN    | softmax+ce, mmd loss           | 1:2        | g5     | 64         | 0.910   | -        |   
| ShanghaiHangzhou | DAN    | softmax+ce, mmd loss           | 1:3        | g5     | 64         | 0.910   | -        |  
| ShanghaiHangzhou | JAN    | softmax+ce, joint mmd loss     | 1:1        | g5     | 64         | 0.920   | -        | 
| ShanghaiHangzhou | JAN    | softmax+ce, joint mmd loss     | 1:0.5      | g5     | 64         | 0.922   | -        | 
| ShanghaiHangzhou | JAN    | softmax+ce, joint mmd loss     | 1:2        | g5     | 64         | 0.931   | -        | 
| ShanghaiHangzhou | JAN    | softmax+ce, joint mmd loss     | 1:3        | g5     | 64         | 0.923   | -        | 
| ShanghaiHangzhou | JAN    | softmax+ce, joint mmd loss     | 1:5        | g5     | 64         | 0.937   | -        | 
| ShanghaiHangzhou | DSAN   | softmax+ce, local mmd loss     | 1:1        | g5     | 64         | 0.908   | -        |    
| ShanghaiHangzhou | DSAN   | softmax+ce, local mmd loss     | 1:0.5      | g5     | 64         | 0.931   | -        |   
| ShanghaiHangzhou | DSAN   | softmax+ce, local mmd loss     | 1:2        | g5     | 64         | 0.933   | -        |   
| ShanghaiHangzhou | DSAN   | softmax+ce, local mmd loss     | 1:3        | g5     | 64         | 0.916   | -        |   
| ShanghaiHangzhou | DSAN   | softmax+ce, local mmd loss     | 1:5        | g5     | 64         | 0.925   | -        |
| ShanghaiHangzhou | DANN   | softmax+ce                     | 1          | -      | 64         | 0.918   | -        |
| ShanghaiHangzhou | MCD    | softmax+ce, l1 loss            | 1:1        | -      | 64         | 0.662   | -        | 
| ShanghaiHangzhou | TSTNet | softmax+ce, mmd loss, got loss | 1:1:0.1    | l      | 100        | 0.801   | -        | 

## <a name="license"></a> 结果

This project is released under the MIT(LICENSE) license.