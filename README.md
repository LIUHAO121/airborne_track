Script for training detection model

### data
The data and annotation files are as follows
I have provided the generated annotation file,
Organize the training data into the following format to start training

```
|-- data
    |-- airmot
        |-- airmot_jpg
            |-- train
                |-- part1
                |-- part1_annotation.json
                |-- part2
                |-- part2_annotation.json
                |-- part3
                `-- part3_annotation.json
        |-- airmot_png
            |-- train
                |-- part1
                |-- part1_annotation.json
                |-- part2
                |-- part2_annotation.json
                |-- part3
                `-- part3_annotation.json
```

Prepare data in two formats, because both are used in training,
They are placed in directory airmot_png and directory airmot_jpg respectively，The annotation files are the same

```
|-- part3
    |-- part3004474050bdc46c2805ae42048c24c2f
    |-- part3004c26c2de5a4d2a85248be48844cb48
    |-- part30075dbdd030f49ae9572cbe397d53971
    |-- part3007f5536cd714212805d711b2419d2a6
    ...
```

```
|-- part3004474050bdc46c2805ae42048c24c2f
    |-- 15550613259964594611d94e7230db3404389e8cba84f78e204.png
    |-- 15550613260705697051d94e7230db3404389e8cba84f78e204.png
    |-- 15550613261978695081d94e7230db3404389e8cba84f78e204.png
    |-- 15550613262703240661d94e7230db3404389e8cba84f78e204.png
    |-- 15550613263771810831d94e7230db3404389e8cba84f78e204.png
    ...
```

## detection model train
### jpg train 30 epoch 

Modify the 311th line of the “train/CenterNet/src/lib/datasets/dataset/airborne.py” to the jpg path 

“”“
self.data_dir = os.path.join(opt.data_dir, 'airmot/airmot_jpg/train')
”“”

```
cd train/CenterNet/src
python main.py --task air_borne --dataset air_borne --num_epochs 30 --batch_size 8 --arch res_18 --input_h 2048 --input_w 2048 --lr 5e-4 --gpus 0,1,2,3 --num_workers 32  --lr_step 16,25 --train_part 1,2,3
```
### png train 5 epoch 

Modify the 311th line of the “train/CenterNet/src/lib/datasets/dataset/airborne.py” to the jpg path 

“”“
self.data_dir = os.path.join(opt.data_dir, 'airmot/airmot_png/train')
”“”

```
python main.py --task air_borne --dataset air_borne --num_epochs 5 --batch_size 16 --arch res_18 --input_h 2048 --input_w 2048 --lr 5e-6 --gpus 0,1,3,4,5,6,7 --num_workers 32  --lr_step 16,25 --train_part 1,2,3 --load_model ../exp/air_borne/default/model_last.pth
```

## reid model train
We did not train the reid model of AOT, but directly use the reid model of pedestrians. The reid model of pedestrians still has certain distinguishing ability in AOT.