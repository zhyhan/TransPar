CUDA_VISIBLE_DEVICES=0 python TransPar-regda.py data/RHD data/H3D_crop -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs/TransPar-regda/rhd2h3d --phase test

CUDA_VISIBLE_DEVICES=0 python source_only.py data/RHD data/H3D_crop -s RenderedHandPose -t Hand3DStudio --log logs/baseline/rhd2h3d --debug --seed 0 --phase test


CUDA_VISIBLE_DEVICES=0 python regda.py data/RHD data/H3D_crop -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs/regda/rhd2h3d --phase test
# CUDA_VISIBLE_DEVICES=0 python regda.py data/FreiHand data/RHD \
#     -s FreiHand -t RenderedHandPose --seed 0 --debug --log logs/regda/freihand2rhd


# CUDA_VISIBLE_DEVICES=0 python regda.py data/surreal_processed data/Human36M \
#     -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 10 --log logs/regda/surreal2human36m #--finetune
# CUDA_VISIBLE_DEVICES=0 python regda.py data/surreal_processed data/lsp \
#     -s SURREAL -t LSP --seed 0 --debug --rotation 30 --log logs/regda/surreal2lsp #--finetune 

