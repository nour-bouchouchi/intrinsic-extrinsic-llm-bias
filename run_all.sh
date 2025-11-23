#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GENDER_BIAS_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

LIST_CONCEPTS_ALL=(
    "professions"
    "months"
    "colors"
    "languages"
    "sports"
    "diseases"
)


mkdir -p "$PROJECT_ROOT/results"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> \"<list_layers (space-separated)>\""
    exit 1
fi

MODEL_NAME="$1"
read -r -a LIST_LAYERS <<< "$2"



python3 src/experiments/generation.py --model_name "$MODEL_NAME" --list_concepts "${LIST_CONCEPTS_ALL[@]}"
python3 src/experiments/generation_classif.py --model_name "$MODEL_NAME" --method "llama70b"  --list_concepts "${LIST_CONCEPTS_ALL[@]}"
python3 src/experiments/dot_prod.py --model_name "$MODEL_NAME" --list_layers "${LIST_LAYERS[@]}" --list_concepts "${LIST_CONCEPTS_ALL[@]}" 
python3 src/experiments/correlation.py --model_name "$MODEL_NAME" --list_layers "${LIST_LAYERS[@]}" --list_concepts "${LIST_CONCEPTS_ALL[@]}" --method "spearman"


# --- SFT ---
python3 src/training/sft_training.py --model_name "$MODEL_NAME"
python3 src/experiments/generation.py --model_name "$MODEL_NAME" --use_model_ft_sft True --list_concepts "${LIST_CONCEPTS_ALL[@]}"
python3 src/experiments/generation_classif.py --model_name "$MODEL_NAME" --method "llama70b" --use_model_ft_sft True --list_concepts "${LIST_CONCEPTS_ALL[@]}"
python3 src/experiments/dot_prod.py --model_name "$MODEL_NAME" --list_layers "${LIST_LAYERS[@]}" --use_model_ft_sft True --list_concepts "${LIST_CONCEPTS_ALL[@]}" #
python3 src/experiments/correlation.py --model_name "$MODEL_NAME" --list_layers "${LIST_LAYERS[@]}" --use_model_ft_sft True --list_concepts "${LIST_CONCEPTS_ALL[@]}" --method "spearman"
python3 src/experiments/correlation.py --model_name "$MODEL_NAME" --list_layers "${LIST_LAYERS[@]}" --use_model_ft_sft True --corr_previous_predictions True --list_concepts "${LIST_CONCEPTS_ALL[@]}" --method "spearman"


# SFT + system prompt jailbreak
python3 src/experiments/generation.py --model_name "$MODEL_NAME" --use_model_ft_sft True --system_prompt_key "jailbreak" --instruction_in_prompt True --list_concepts "${LIST_CONCEPTS_ALL[@]}"
python3 src/experiments/generation_classif.py --model_name "$MODEL_NAME" --method "llama70b" --use_model_ft_sft True --system_prompt_key "jailbreak" --instruction_in_prompt True  --list_concepts "${LIST_CONCEPTS_ALL[@]}"


# --- Benchmarks ---
python3 eval_benchmarks.py --model_name "$MODEL_NAME" --use_model_ft_sft False --list_datasets mmlu
python3 eval_benchmarks.py --model_name "$MODEL_NAME" --use_model_ft_sft True --list_datasets mmlu
python3 eval_benchmarks.py --model_name "$MODEL_NAME" --use_model_ft_sft False --list_datasets ifeval
python3 eval_benchmarks.py --model_name "$MODEL_NAME" --use_model_ft_sft True --list_datasets ifeval
