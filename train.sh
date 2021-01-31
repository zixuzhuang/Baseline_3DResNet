gpu=0
net=3D_ResNet10
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 0
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 1
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 2
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 3
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 4

net=3D_ResNet18
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 0
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 1
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 2
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 3
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 4

net=3D_ResNet34
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 0
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 1
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 2
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 3
CUDA_VISIBLE_DEVICES=$gpu python ./train.py -net=$net -fold 4