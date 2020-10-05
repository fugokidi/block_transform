# Block-wise Image Transformation with Secret Key for Adversarially Robust Defense

This repository is the official implementation of [Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](http://arxiv.org/abs/2010.00801). 

Training scripts are mostly adapted from [fast adversarial training](https://github.com/locuslab/fast_adversarial)


## Requirements
* Automatic mixed precision is used and therefore, [apex](https://nvidia.github.io/apex/) library is required.
* For format preserving encryption, we utilize [pyffx](https://github.com/emulbreh/pyffx).

```setup
pip install -r requirements.txt
```
* Make sure normalization is enabled in `keydefense.py` (i.e., uncomment normalization)


## CIFAR-10

```
python train.py --work-path <path-to-config-folder>
python eval.py --work-path <path-to-config-folder> --attack <attack-name>
```

## ImageNet
Run the training/evaluation script, as an example,
```
bash run_shuffle_4.sh
bash run_shuffle_eval.sh
```
