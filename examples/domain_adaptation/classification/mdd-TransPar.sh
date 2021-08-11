#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2W --keep_ratio 0.48
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2W --keep_ratio 0.5523
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2D --keep_ratio 0.5429
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2D --keep_ratio 0.4866
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2A --keep_ratio 0.4875
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2A --keep_ratio 0.4914

# Office-Home
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Cl --keep_ratio 0.5139
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Pr --keep_ratio 0.5554
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Ar2Rw --keep_ratio 0.6308
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Ar --keep_ratio 0.5152
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Pr --keep_ratio 0.5471
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Cl2Rw --keep_ratio 0.5402
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Ar --keep_ratio 0.5454
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Cl --keep_ratio 0.5262
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Pr2Rw --keep_ratio 0.6237
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Ar --keep_ratio 0.6031
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Cl --keep_ratio 0.5348
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdd/OfficeHome_Rw2Pr --keep_ratio 0.6214

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python mdd-TransPar.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --epochs 30 \
    --bottleneck-dim 1024 --seed 0 --center-crop --per-class-eval -b 36 --log logs/mdd/VisDA2017 --keep_ratio 0.4664
