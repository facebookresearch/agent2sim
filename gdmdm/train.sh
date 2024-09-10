# bash projects/gdmdm/train.sh home-2024-02-14--17 bunny 0
load_logname=$1
logname=$2
batchsize=$3
dev=$4
# add_args=${*: 4:$#-1}

# rm -rf projects/gdmdm/exps/$load_logname-$logname
# bash scripts/train.sh projects/gdmdm/train.py $dev --num_epochs 4001 \
#   --load_logname $load_logname \
#   --logname $logname \
#   --train_batch_size $batchsize
#   # $add_args

rm -rf projects/gdmdm/exps/$load_logname-$logname
bash ../lab4d/scripts/train.sh train.py $dev --num_epochs 4001 \
  --load_logname $load_logname \
  --logname $logname \
  --train_batch_size $batchsize \
  # --fill_to_size $batchsize
  # --swap_cam_root
  # $add_args