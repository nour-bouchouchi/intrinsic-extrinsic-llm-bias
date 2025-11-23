import torch
import argparse
import json
import os
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from utils.cli import str2bool
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

if hasattr(torch, "compile"):
    torch.compile = lambda model, *args, **kwargs: model

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"

def merge_peft_model(model_name, peft_type):
    """
    Merge a PEFT model with its base model.
    Args:
        model_name: Name of the base model.
        peft_type: Type of PEFT model ('dpo' or 'sft').
    Returns:
        save_path: Path to the merged model.
    """
    peft_path = f"models_{peft_type}/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(peft_path)
    peft_config = PeftConfig.from_pretrained(peft_path)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    merged_model = peft_model.merge_and_unload()
    
    save_path = f"./tmp_merged_{model_name}_{peft_type}"
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return save_path
    


def main(args):
    torch.set_float32_matmul_precision("high")
    torch._dynamo.disable()

    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    
    if args.use_model_ft_dpo:
        model_path = merge_peft_model(args.model_name, "dpo")
    elif args.use_model_ft_sft:
        model_path = merge_peft_model(args.model_name, "sft")
    else:
        model_path = args.model_name

    hf_model = HFLM(
        pretrained=model_path,
        dtype="bfloat16",
        device=device,
        batch_size="auto:4",
    )

    print(f"Évaluation sur : {args.list_datasets}, (en {args.num_fewshot} shot(s))")

    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=args.list_datasets,
        num_fewshot=args.num_fewshot,
        batch_size=None, 
    )
    
    suffix = f"{'_lora_dpo' if args.use_model_ft_dpo else ''}{'_lora_sft' if args.use_model_ft_sft else ''}"
    folder = ROOT / "results" / "eval" / f"{args.model_name}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"results_eval_{'_'.join(args.list_datasets)}.json"

    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nRésultats sauvegardés dans : {filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--num_fewshot', type=int, default=0, 
                        help='Nombre d\'exemples few-shot pour l\'évaluation')
    parser.add_argument(
        '--list_datasets',
        nargs='+',
        default=['ifeval', 'bbh', 'mmlu'],
        help="Liste des datasets à utiliser"
    )
    args = parser.parse_args()
    main(args)