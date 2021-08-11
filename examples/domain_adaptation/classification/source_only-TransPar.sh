#!/usr/bin/env bash
# Office31
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_A2W --keep_ratio 0.48
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_D2W --keep_ratio 0.5523
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_W2D --keep_ratio 0.5429
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_A2D --keep_ratio 0.4866
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_D2A --keep_ratio 0.4875
CUDA_VISIBLE_DEVICES=0 python source_only-TransPar.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 5 --seed 0 --log logs/src_only/Office31_W2A --keep_ratio 0.4914