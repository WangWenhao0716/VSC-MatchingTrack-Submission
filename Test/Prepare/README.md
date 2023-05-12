# The training process for the auxiliary models.

## Get ```deblack_20230107.pt```

Please first enter the ```deblack_20230107``` folder by ```cd deblack_20230107```, and then follow the below instructions.

1. Download the pre-trained models:
```
mkdir weights && cd weights
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
```

2. The training dataset is auto-generated (by adding black padding to images from DISC21), you can directly download from [here](https://drive.google.com/file/d/1kN2j5HXJNIkMWvH-163yUfUrFlWK38mL/view?usp=share_link), and unzip by:
```
tar -xvf isc_v5.tar
```

3. Assume we have $4$ A100 GPUs, and the images are stored in ```/raid/VSC/yolov5/isc_v5```; we can train the model by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --img-size 640 --batch-size 64 \
--epochs 50 --data ./data/isc_v5.yaml --cfg ./models/yolov5x.yaml --weights weights/yolov5x.pt
```

A demo generated by the train script is

![image](https://github.com/WangWenhao0716/VSC-DescriptorTrack-Submission/blob/main/Test/Prepare/deblack_20230107/train_batch0.jpg)

4. After training, we use the best checkpoint as the ```deblack_20230107.pt``` by:
```
cp ./runs/train/exp/weights/best.pt deblack_20230107.pt
```

## Get ```best_20230101.pt```


Please first enter the ```best_20230101``` folder by ```cd best_20230101```, and then follow the below instructions.

1. Download the pre-trained models:
```
mkdir weights && cd weights
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
```

2. The training dataset is auto-generated (by adding random overlay to images from DISC21), you can directly download from [here](https://drive.google.com/file/d/1-2mniqP36BKqKwy-Frk6EqIHOZnN_7v0/view?usp=share_link), and unzip by:
```
tar -xvf isc_v3.tar
```

3. Assume we have $4$ A100 GPUs, and the images are stored in ```/raid/VSC/yolov5/isc_v3```; we can train the model by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --img-size 640 --batch-size 64 \
--epochs 50 --data ./data/isc_v3.yaml --cfg ./models/yolov5x.yaml --weights weights/yolov5x.pt
```

A demo generated by the train script is

![image](https://github.com/WangWenhao0716/VSC-DescriptorTrack-Submission/blob/main/Test/Prepare/best_20230101/train_batch0.jpg)

4. After training, we use the best checkpoint as the ```best_20230101.pt``` by:
```
cp ./runs/train/exp/weights/best.pt best_20230101.pt
```

## Get ```rotate_detect.pth```

Please first enter the ```rotate_detect``` folder by ```cd rotate_detect```, and then follow the below instructions.

1. Download the pre-trained models:
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth
```

2. The training dataset is auto-generated (by rotating images from  DISC21 90, 180, 270 degrees), you can directly download from [here](https://drive.google.com/file/d/12N0pXF2dP1NNRvXnJZKzGtGajnQAwRtZ/view?usp=share_link), and unzip by:
```
tar -xvf rotate_images.tar
```

3. Assume we have $8$ A100 GPUs, and the images are stored in ```/raid/VSC/images/rotate_images```; we can train the model by

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_small_patch4_window7_224.yaml --data-path /raid/VSC/images/rotate_images --batch-size 128 \
--pretrained swin_small_patch4_window7_224_22k.pth
```

4. After training, we use the final checkpoint as the ```rotate_detect.pth``` by: 

```
cp output/swin_small_patch4_window7_224/default/ckpt_epoch_299.pth rotate_detect.pth
```