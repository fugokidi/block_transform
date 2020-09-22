DATA=~/imagenet

CONFIG=configs/configs_pretrained_eval.yml

# python -u eval.py $DATA -c $CONFIG --pretrained --attack onepixel --batch-size 1
# python -u eval.py $DATA -c $CONFIG --pretrained --attack nattack --batch-size 1
# python -u eval.py $DATA -c $CONFIG --pretrained --attack spsa --batch-size 1
python -u eval.py $DATA -c $CONFIG --pretrained --attack pgd --batch-size 32
# python -u eval.py $DATA -c $CONFIG --pretrained --attack cw --batch-size 10
# python -u eval.py $DATA -c $CONFIG --pretrained --attack ead --batch-size 10

