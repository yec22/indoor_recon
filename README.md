# NeuRIS

## 数据预处理指令
```
python ./pre_process.py --data_type private
```

## 训练指令
```
python ./exp_runner.py --mode train --conf ./confs/neuris.conf --gpu 0
```

## 训练完成后提取几何的指令
```
python ./exp_runner.py --mode validate_mesh --conf ./confs/neuris.conf --is_continue --gpu 2 --mc_reso 512
```

生成结果在exps/indoor/neus/N11_new/exp_N11_new/meshes文件夹下

## 后处理的指令（按照view-frustrum去除多余部分）

```
python ./post_process.py --scene_name N11_new
```

生成结果在exps/indoor/neus/N11_new文件夹下