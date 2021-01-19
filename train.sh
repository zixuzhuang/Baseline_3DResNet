gpu=0
bs=4
fold=0

# log_path="./data/ViT/logs/$(date +'%Y-%m-%d')/"
# log=$log_path"/F$fold-$(date +'%H-%M').log"

# if [ ! -d $log_path ]; then
#   mkdir -p $log_path
# fi

CUDA_VISIBLE_DEVICES=$gpu python ./train.py  -bs $bs -fold $fold
#  | tee $log
