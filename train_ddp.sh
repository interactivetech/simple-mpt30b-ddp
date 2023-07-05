CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nproc_per_node=8 test_int8_training.py  \
  --model_cache_dir <PATH> \
  --cache_dir <PATH> \
  --model_save_dir <PATH> \
  --dataset_path <PATH>