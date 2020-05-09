# Airport_Apron
```shell script
python train.py ctdet --exp_id coco_dla --batch_size 1 --lr 1.25e-4  --gpus 0 --num_epochs 1000 --lr_step 500,800
python test.py ctdet --exp_id coco_dla --keep_res --load_model ../exp/ctdet/coco_dla/model_last.pth
python demo.py ctdet --exp_id coco_dls --keep_res --load_model ../exp/ctdet/coco_dla/model_last.pth
python demo.py ctdet --demo ../data/VOC2007/JPEGImages/2.jpg --load_model ../data/model_last.pth
```
