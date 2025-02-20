export PYTHONPATH="."
export BASE_URL=""
export API_KEY=""

num_sim_tasks=400
num_real_tasks=600

python evaluate/evaluate_track1.py \
    --agent dummyagent.dummyagent \
    --task_set amazon \
    --num_sim_tasks $num_sim_tasks \
    --eval_num_sim_tasks $num_sim_tasks \
    --num_real_tasks $num_real_tasks \
    --eval_num_real_tasks $num_real_tasks \
    --multi_threading \
    --workers 32 \
    --prefix "dummyagent" \
    --suffix "real-final"