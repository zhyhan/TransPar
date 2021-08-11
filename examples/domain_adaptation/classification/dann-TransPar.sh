#!/usr/bin/env bash
#Office31
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W --keep_ratio 0.48
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_D2W --keep_ratio 0.5523
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_W2D --keep_ratio 0.5429
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2D --keep_ratio 0.4866
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_D2A --keep_ratio 0.4875
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_W2A --keep_ratio 0.4914

# Office-Home
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Ar2Cl --keep_ratio 0.5139
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Ar2Pr --keep_ratio 0.5554
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Ar2Rw --keep_ratio 0.6308
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Cl2Ar --keep_ratio 0.5152
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Cl2Pr --keep_ratio 0.5471
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Cl2Rw --keep_ratio 0.5402
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Pr2Ar --keep_ratio 0.5454
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Pr2Cl --keep_ratio 0.5262
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Pr2Rw --keep_ratio 0.6237
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Rw2Ar --keep_ratio 0.6031
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Rw2Cl --keep_ratio 0.5348
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --seed 0 --log logs/dann/OfficeHome_Rw2Pr --keep_ratio 0.6214

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 \
    --epochs 30 --seed 0 --per-class-eval --center-crop --log logs/dann/VisDA2017 --keep_ratio 0.4664

# # DomainNet
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s c -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_c2i
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s c -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_c2p
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s c -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_c2r
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s c -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_c2s
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s i -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_i2c
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s i -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_i2p
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s i -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_i2r
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s i -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_i2s
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s p -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_p2c
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s p -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_p2i
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s p -t r -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_p2r
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s p -t s -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_p2s
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s r -t c -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_r2c
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s r -t i -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_r2i
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s r -t p -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_r2p
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s r -t s -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_r2s
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s s -t c -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_s2c
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s s -t i -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_s2i
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s s -t p -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_s2p
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s s -t r -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_s2r
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s c -t q -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_c2q
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s i -t q -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_i2q 
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s p -t q -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_p2q
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s r -t q -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_r2q
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s q -t c -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_q2c
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s q -t i -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_q2i
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s q -t p -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_q2p
CUDA_VISIBLE_DEVICES=0 python dann_transPar.py data/domainnet -d DomainNet -s q -t r -a resnet101 --epochs 20 -i 2500 -p 500 --seed 0 --log logs/dann/DomainNet_q2r
