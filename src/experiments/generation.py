import torch
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from models.hf import get_model                    
from models.cache import format_with_system_prompt 
from utils.cli import str2bool, get_suffix_folder, get_sentence_prompt                  
from constants import PERSON, CONCEPTS, SYSTEM_PROMPTS
from models.cache import add_hooks, get_direction_ablation_input_pre_hook, get_format_block_names

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



def generate_only(model, tokenizer, neutral_person, entity, concept="professions", system_prompt=None, instruction_in_prompt=False, 
                  ablation_layer=None, ablation_direction=None, model_name=None, ft=False):
    """
    Generate text for a given neutral person and entity.
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        neutral_person: The neutral persona to use.
        entity: The entity to include in the prompt.
        concept: The concept to analyze (e.g., 'professions').
        system_prompt: Optional system prompt to use.
        instruction_in_prompt: Boolean indicating if instruction is in the prompt.
        ablation_layer: Layer number for direction ablation (if any).
        ablation_direction: Direction vector for ablation (if any).
        model_name: The name of the model.
        ft: Boolean indicating if the model is fine-tuned.
    Returns:
        generated_text: The generated text from the model.  
    """

    prompt = get_sentence_prompt(concept, neutral_person, entity) 

    formatted_prompt = format_with_system_prompt(tokenizer, prompt, system_prompt, instruction_in_prompt=instruction_in_prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    if ablation_layer is not None and ablation_direction is not None:
        block_name = get_format_block_names(model_name, ablation_layer, ft=True)
        module = dict(model.named_modules())[block_name]

        with add_hooks(
            module_forward_pre_hooks=[
                (module, get_direction_ablation_input_pre_hook(ablation_direction))
            ]
        ):
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
    else :
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                do_sample=True, 
                temperature=0.7, 
                pad_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("Generated text:", generated_text)
    return generated_text


def generate_all(model, tokenizer, folder, overwrite, concept="professions", system_prompt=None, instruction_in_prompt=False, ablation_layer=None, ablation_direction=None, 
                 model_name=None, ft=False):
    """
    Generate texts for all entities in a concept and save to a JSON file.
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        folder: Folder to save the generated texts.
        overwrite: Boolean indicating whether to overwrite existing files.
        concept: The concept to analyze (e.g., 'professions').
        system_prompt: Optional system prompt to use.
        instruction_in_prompt: Boolean indicating if instruction is in the prompt.
        ablation_layer: Layer number for direction ablation (if any).
        ablation_direction: Direction vector for ablation (if any).
        model_name: The name of the model.
        ft: Boolean indicating if the model is fine-tuned.
    """    
    if not os.path.exists(f"{folder}/{concept}.json") or overwrite:
        generation_data = {}
        for entity in tqdm(CONCEPTS[concept]):
            generation_data[entity] = {}
            for neutral_person in PERSON["neutral"]:
                generation_data[entity][neutral_person] = []
                for _ in range(10):
                    text = generate_only(model, tokenizer, neutral_person, entity, concept=concept, system_prompt=system_prompt, instruction_in_prompt=instruction_in_prompt, 
                                         ablation_layer=ablation_layer, ablation_direction=ablation_direction, model_name=model_name, ft=ft)
                    generation_data[entity][neutral_person].append(text)

        with open(f"{folder}/{concept}.json", "w", encoding="utf-8") as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)


def generation(model, tokenizer, folder, overwrite, list_concepts, system_prompt=None, instruction_in_prompt=False, ft=False): 
    for concept in list_concepts:
        print(f"Generating {concept} examples...")
        generate_all(model, tokenizer, folder, overwrite, concept=concept, system_prompt=system_prompt, instruction_in_prompt=instruction_in_prompt, 
                     model_name=args.model_name, ft=ft)
        print(f"Generation for {concept} completed.")
    
    
    

def main(args):
    print("====== Text Generation ======")
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    
    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_key, None) if hasattr(args, 'system_prompt_key') else None
     
     
    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft, args.lora_scale)


    suffix_folder = get_suffix_folder(args.instruction_in_prompt, args.system_prompt_key, args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft, args.lora_scale)
   
    folder = ROOT / "results" / "generation" / suffix_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    generation(model, tokenizer, str(folder), args.overwrite, args.list_concepts, system_prompt=system_prompt, instruction_in_prompt=args.instruction_in_prompt, ft=args.use_model_ft_dpo or args.use_model_ft_sft)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--lora_scale', type=float, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--system_prompt_key', type=str, default="none",
                        choices=["none", "jailbreak"],
                        help='Type de system prompt à utiliser')
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    args = parser.parse_args()
    main(args)