DATA=~/imagenet

NAME=NP_4

CONFIG=configs/configs_np_4_eval.yml

END=trained_models/${NAME}/checkpoint_epoch15.pth.tar

# python -u eval.py $DATA -c $CONFIG --resume $END --attack onepixel --batch-size 1
# python -u eval.py $DATA -c $CONFIG --resume $END --attack nattack --batch-size 1
# python -u eval.py $DATA -c $CONFIG --resume $END --attack spsa --batch-size 10

# python -u eval.py $DATA -c $CONFIG --resume $END --attack ead --batch-size 10

# python -u eval.py $DATA -c $CONFIG --resume $END --attack cw --batch-size 10

python -u eval.py $DATA -c $CONFIG --resume $END --attack pgd --batch-size 10

# python -u eval.py $DATA -c $CONFIG --resume $END --attack no
