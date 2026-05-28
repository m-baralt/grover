
mkdir -p model # create directory to save fine-tuned model weights if it doesn't exist

accelerate launch main.py finetune --data_path external/grover/solubility_data/solubility_data.csv \
    --save_dir model \
    --checkpoint_path external/grover/checkpoints/grover_large.pt \
    --dataset_type regression --split_type scaffold_balanced \
    --split_sizes 0.8 0.1 0.1 \
    --metric rmse \
    --ensemble_size 1 \
    --num_folds 2 \
    --ffn_hidden_size 200 \
    --batch_size 32 \
    --epochs 5 \
    --init_lr 0.00015 \
    --self_attention \
    --save_smiles_splits \
    --fine_tune_coff 1.0
    --use_wandb \
    --wandb_entity mariabar 2>&1 | tee run.log