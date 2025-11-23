import os 
import argparse
import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  

from models.hf import get_model                    
from utils.cli import str2bool, get_suffix_folder, get_sentence_prompt
from constants import PERSON, CONCEPTS
from utils.directions import compute_lambdas, compute_gender_direction

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch._dynamo
import torch.nn.functional as F

torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

if hasattr(torch, "compile"):
    torch.compile = lambda model, *args, **kwargs: model

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"




def compute_dot_product(model, tokenizer, layer, concept, v_gender, model_name, ft, system_prompt=None, instruction_in_prompt=False, eps=1e-12): 
    """
    Compute the dot product between the gender direction vector and the concept activations.
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        layer: The layer number to extract activations from.
        concept: The concept to analyze (e.g., 'professions').
        v_gender: The gender direction vector.
        model_name: The name of the model.
        ft: Boolean indicating if the model is fine-tuned.
        system_prompt: Optional system prompt to use.
        instruction_in_prompt: Boolean indicating if instruction is in the prompt.
        eps: Small value to avoid division by zero.
    Returns:
        dico_dot_products_mean: Mean dot products for each concept.
        dico_dot_products_std: Standard deviation of dot products for each concept.
        all_neutral_activations: List of all neutral activations.
        activations_records: List of activation records with entity, persona, and activation.
    """
    lambdas_concept = {"neutral": {}}
    
    activations_records = [] 
    for entity in CONCEPTS[concept]:
        sentences = [get_sentence_prompt(concept, persona, entity) for persona in PERSON["neutral"]]
        activations = [compute_lambdas(s, model, tokenizer, layer, model_name, ft, system_prompt, instruction_in_prompt) for s in sentences]
        lambdas_concept["neutral"][entity] = activations
        
        for i, act in enumerate(activations):
            activations_records.append({
                "entity": entity,
                "persona": PERSON["neutral"][i],
                "activation": act[0].detach().float().cpu().numpy().ravel()
            })

    all_neutral_activations = [v for sub in lambdas_concept["neutral"].values() for v in sub]

    
    dot_products = {"neutral": {}}
    
    for c in CONCEPTS[concept]:  
        dot_products["neutral"][c] = [torch.dot(v[0].float(), v_gender).item() for v in lambdas_concept["neutral"][c]]

    dico_dot_products_mean = {"neutral": {}}
    dico_dot_products_std = {"neutral": {}}

    for c in CONCEPTS[concept]:
        dico_dot_products_mean["neutral"][c] = np.mean(dot_products["neutral"][c])
        dico_dot_products_std["neutral"][c] = np.std(dot_products["neutral"][c])
        
    
    return (dico_dot_products_mean, 
            dico_dot_products_std, 
            all_neutral_activations, 
            activations_records)


def save_dot_products(dico_mean, dico_std, concept, folder, layer, measure="dot_products"):  
    """
    Save the dot product results to a CSV file.
    Args:
        dico_mean: Dictionary of mean dot products.
        dico_std: Dictionary of standard deviation of dot products.
        concept: The concept being analyzed.
        folder: The folder to save the CSV file in.
        layer: The layer number.
        measure: The measure type (default is "dot_products").  
    """
    mean_values = dico_mean['neutral']
    std_values = dico_std['neutral']

    sorted_items = sorted(mean_values.items(), key=lambda x: x[1]) 

    concepts = [c for c, _ in sorted_items]
    means = [mean_values[c] for c in concepts]
    stds = [std_values[c] for c in concepts]

    df_dot = pd.DataFrame({
        'Concept': concepts,
        'Mean': means,
        'Std': stds
    })

    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"{measure}_{concept}_L{layer}.csv")
    df_dot.to_csv(csv_path, index=False)


def compute_bias_scores(mean_values, activations=None):
    """
    Compute summary statistics of latent bias for a concept.
    Args:
        mean_values (dict): Mean dot products or cosine similarities for each concept.
        activations (list, optional): List of neutral activations for the concept.
    Returns:
        dict: A dictionary containing mean absolute dot product, standard deviation, and normalized standard deviation.
    """
    values = np.array(list(mean_values.values()))
    mean_abs = np.mean(np.abs(values))
    std = np.std(values)

    if activations is not None :
        norms = [torch.norm(act[0].float()).item() for act in activations]
        mean_norm = np.mean(norms)
        std_normalized = std / mean_norm if mean_norm != 0 else 0
    else:
        std_normalized = None

    return {
        "mean_abs_dot": mean_abs,
        "std_dot": std,
        "std_norm_dot": std_normalized, 
    }



def main(args):
    print("====== Dot Product Analysis ======")
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    

    ft = args.use_model_ft_dpo or args.use_model_ft_sft
    
    suffix_folder = get_suffix_folder(None, "none", args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)
    
    folder = ROOT / "results" / "dot_prod" / suffix_folder
    folder_dot_prod = folder / "dot_products"
    folder_activations = folder / "activations"
    os.makedirs(folder_dot_prod, exist_ok=True)
    os.makedirs(folder_activations, exist_ok=True)

    folder = str(folder)
    folder_dot_prod = str(folder_dot_prod)
    folder_activations = str(folder_activations)
    
    model, tokenizer = get_model(args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)
    
    bias_scores_all_layers_dot_prod = []
    for i,layer in enumerate(args.list_layers):
        print(f"Processing layer {layer}...")
        v_gender, _ = compute_gender_direction(model, tokenizer, layer, args.model_name, ft, None, args.instruction_in_prompt)
        print("Gender direction computed.")
        for concept in args.list_concepts:
            print(f"Processing concept: {concept}...")
            (
                dico_dot_products_mean,
                dico_dot_products_std,
                neutral_activations, 
                activations_records
            ) = compute_dot_product(model, tokenizer, layer, concept, v_gender, args.model_name, ft, None, args.instruction_in_prompt)

            df_acts = pd.DataFrame([
                {
                    "entity": a["entity"],
                    "persona": a["persona"],
                    "activation": " ".join(map(str, a["activation"]))
                }
                for a in activations_records
            ])

            df_acts.to_csv(os.path.join(folder_activations, f"activations_{concept}_L{layer}.csv"), index=False)           

            save_dot_products(dico_dot_products_mean, dico_dot_products_std, concept, folder_dot_prod, layer, measure="dot_products")

            bias_score_dot_prod = compute_bias_scores(dico_dot_products_mean["neutral"],  activations=neutral_activations)
            bias_score_dot_prod["layer"] = layer
            bias_score_dot_prod["concept"] = concept
            bias_scores_all_layers_dot_prod.append(bias_score_dot_prod)
            
            


    df_scores_dot_prod = pd.DataFrame(bias_scores_all_layers_dot_prod)
    df_scores_dot_prod.to_csv(os.path.join(folder, "bias_scores_dot_products.csv"), index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--list_layers', nargs='+', default=[5], type=int)
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    args = parser.parse_args()
    main(args)