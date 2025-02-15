export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs/json_proj
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/json_align.sh 2>&1 | tee logs/json_proj/tee_logs.txt
sleep 1m

mkdir -p logs/code_proj
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/code_align.sh 2>&1 | tee logs/code_proj/tee_logs.txt
sleep 1m

mkdir -p logs/table_proj
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/table_align.sh 2>&1 | tee logs/table_proj/tee_logs.txt