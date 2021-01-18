
datasets_syn=(
    "TRIANGLE"
    "TRIANGLE_EX"
    "LCC"
    "LCC_EX"
)

mds=(
    "MDS"
    "MDS_EX"
)

datasets_real=(
    "MUTAG"
    "NCI1"
    "PROTEINS"
)


mkdir -p logs
mkdir -p dataset

for i in "${datasets_syn[@]}"; do
    python3 dataset_gen.py $i
    python3 main.py --dataset $i --random 100 --filename logs/log_${i}_random
    python3 main.py --dataset $i --filename logs/log_${i}
    python3 main.py --dataset $i --random 100 --filename logs/log_GCN_${i}_random --neighbor_pooling_type average --num_mlp_layers 1
    python3 main.py --dataset $i --filename logs/log_GCN_${i} --neighbor_pooling_type average --num_mlp_layers 1
done

for i in "${mds[@]}"; do
    python3 dataset_gen.py $i
    python3 main.py --dataset $i --hidden_dim 1024 --num_layers 10 --lr 0.1 --opt sgd --epochs 50000 --random 100 --filename logs/log_${i}_random
    python3 main.py --dataset $i --hidden_dim 1024 --num_layers 10 --lr 0.1 --opt sgd --epochs 50000 --filename logs/log_${i}
    python3 main.py --dataset $i --hidden_dim 1024 --num_layers 10 --lr 0.1 --opt sgd --epochs 50000 --random 100 --filename logs/log_GCN_${i}_random --neighbor_pooling_type average --num_mlp_layers 1
    python3 main.py --dataset $i --hidden_dim 1024 --num_layers 10 --lr 0.1 --opt sgd --epochs 50000 --filename logs/log_GCN_${i} --neighbor_pooling_type average --num_mlp_layers 1
done


for i in "${datasets_real[@]}"; do
    for j in `seq 0 9`; do
        python3 main.py --dataset $i --fold_idx $j --random 100 --filename logs/log_${i}_random
        python3 main.py --dataset $i --fold_idx $j --filename logs/log_${i}
        python3 main.py --dataset $i --fold_idx $j --random 100 --filename logs/log_GCN_${i}_random --neighbor_pooling_type average --num_mlp_layers 1
        python3 main.py --dataset $i --fold_idx $j --filename logs/log_GCN_${i} --neighbor_pooling_type average --num_mlp_layers 1
    done
done

