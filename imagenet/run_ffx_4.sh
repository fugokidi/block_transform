DATA160=~/imagenet-sz/160
DATA352=~/imagenet-sz/352
DATA=~/imagenet

NAME=FFX_4

CONFIG1=configs/configs_ffx_4_phase1.yml
CONFIG2=configs/configs_ffx_4_phase2.yml
CONFIG3=configs/configs_ffx_4_phase3.yml


OUT1=train_ffx_4_phase1.out
OUT2=train_ffx_4_phase2.out
OUT3=train_ffx_4_phase3.out

EVAL1=eval_ffx_4_phase1.out
EVAL2=eval_ffx_4_phase2.out
EVAL3=eval_ffx_4_phase3.out

END1=trained_models/${NAME}/checkpoint_epoch6.pth.tar
END2=trained_models/${NAME}/checkpoint_epoch12.pth.tar
END3=trained_models/${NAME}/checkpoint_epoch15.pth.tar

# training for phase 1
python -u main.py $DATA160 -c $CONFIG1 --pretrained | tee $OUT1

# evaluation for phase 1
# python -u main.py $DATA160 -c $CONFIG1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
python -u main.py $DATA352 -c $CONFIG2 --resume $END1 | tee $OUT2

# evaluation for phase 2
# python -u main.py $DATA352 -c $CONFIG2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
python -u main.py $DATA -c $CONFIG3 --resume $END2 | tee $OUT3

# evaluation for phase 3
python -u main.py $DATA -c $CONFIG3 --resume $END3 --evaluate | tee $EVAL3

# evalutate pretrained model
# python -u main.py $DATA -c $CONFIG3 --evaluate --pretrained

# train fixed size
# python -u main.py $DATA -c $CONFIG4 --pretrained | tee $OUT4
# python -u main.py $DATA -c $CONFIG4 --resume $END4 --evaluate | tee $EVAL4
