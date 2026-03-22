CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=/Code/UGP/ckpts/kitti.pth.tar --distance=10 # 20, 30, 40
CUDA_VISIBLE_DEVICES=0 python eval.py --method=lgr
CUDA_VISIBLE_DEVICES=0 python eval.py --method=ransac --num_corr=50000


