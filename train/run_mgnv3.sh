python train_mgnv2.py --gpu 0,1,2,3,4,5,6,7 --code mgnv3_resnet101_ibn_a_32_tf_sample_kesci_trainval_v2 --epochs 80 --batch-size 256 --load_img_to_cash 0 --least_image_per_class 2 --use_tf_sample 1  --net MGNv3 --height 384 --width 128 --optim SGD --weight-decay 1e-3 --lr 0.1 --workers 16 --data KESCI --margin 0.5 --use_random_pad 1 --part trainval

#python train_mgnv2.py --gpu 0,1,2,3 --code mgnv3_resnet101_ibn_a_32_tf_sample_kesci_random_crop_and_pad_erase --epochs 80 --batch-size 128 --load_img_to_cash 0 --least_image_per_class 2 --use_tf_sample 1  --net MGNv3 --height 384 --width 128 --optim SGD --weight-decay 1e-3 --lr 0.05 --workers 16 --data KESCI --margin 0.5 --use_random_pad 1


