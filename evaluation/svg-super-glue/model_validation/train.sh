python3 train.py \
  --model_name answerdotai/ModernBERT-base \
  --model_revision main \
  --task_name sgp-bench \
  --learning_rate 8e-5 \
  --weight_decay 1e-6 \
  --batch_size 16 \
  --num_train_epochs 1 \
  --lr_scheduler_type linear