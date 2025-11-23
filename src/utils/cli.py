import argparse
from constants import SCHEMA

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}.")

def get_sentence_prompt(concept, persona, entity):
    if concept not in SCHEMA:
        raise ValueError(f"Concept '{concept}' not found in SCHEMA.")
    if concept=="professions" and entity[0].lower() in ['a','e','i','o','u']:
        return f"{persona}{SCHEMA[concept].replace('a ','an ')}{entity.lower()}, "
    return f"{persona}{SCHEMA[concept]}{entity.lower()}, "

def get_suffix_folder(instruction_in_prompt, system_prompt_key, model_name, use_model_ft_dpo=False, use_model_ft_sft=False, lora_scale=None):
    suffix = '_lora_dpo' if use_model_ft_dpo else '_lora_sft' if use_model_ft_sft else ''
    
    if system_prompt_key != "none":
        if instruction_in_prompt:
            suffix += f"_instruction_in_prompt_{system_prompt_key}"
        else:
            suffix += f"_instruction_{system_prompt_key}"
    
    if lora_scale is not None:
        scale_str = f"{lora_scale:.6g}"  # "0.1"
        suffix += f"_scale_{scale_str}"

    return f"{model_name}{suffix}"



