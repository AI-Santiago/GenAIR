gpu_id=0
dataset="beauty"
seed_list=(44 45 46)
export LLM_EMB_FILE=${LLM_EMB_FILE:-./data/${dataset}/handled/item_pca384_concat768.pkl}
export CO_OCCURRENCE_FILE=${CO_OCCURRENCE_FILE:-./data/${dataset}/handled/co_occurrence.npz}
llm_emb_file=${LLM_EMB_FILE}
ts_user=9
ts_item=4
tau=4
alpha=0.0001

model_name="genair_sasrec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 50 \
                --seed ${seed} \
                --check_path "sasrec" \
                --patience 20 \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --log 
done

model_name="genair_bert4rec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 50 \
                --seed ${seed} \
                --check_path "bert4rec" \
                --patience 20 \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --log 
done

model_name="genair_gru4rec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 50 \
                --seed ${seed} \
                --check_path "gru4rec" \
                --patience 20 \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --log 
done