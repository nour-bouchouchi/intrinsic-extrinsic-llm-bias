import torch
from sklearn.covariance import ledoit_wolf
from typing import List, Tuple, Any

from models.cache import run_with_cache_hf
from constants import gender_phrases  # pairs (M/F)

@torch.no_grad()
def compute_lambdas(
    texts: List[str] | str,
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    model_name: str,
    ft: bool,
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
) -> torch.Tensor:
    cache = run_with_cache_hf(model, tokenizer, texts, layer, model_name, ft, system_prompt, instruction_in_prompt)
    return cache["hook_resid_post"][:, -1, :].cpu()

def estimate_single_dir_from_embeddings(embeddings: torch.Tensor, unit_vector: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    # Adapted from Park 2024 : The Linear Representation Hypothesis and the Geometry of Large Language Models
    embeddings = embeddings.to(torch.float32)
    
    category_mean = embeddings.mean(dim=0)
    
    cov = ledoit_wolf(embeddings.cpu().numpy())
    cov = torch.tensor(cov[0], device = embeddings.device)
    
    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean
    
    eps = 1e-12
    lda_dir = lda_dir / (torch.norm(lda_dir) + eps)

    if not unit_vector: # rescale to original magnitude
        lda_dir = (category_mean @ lda_dir) * lda_dir
    # else : 
    #     category_mean = category_mean / (torch.norm(category_mean) + eps)

    return lda_dir, category_mean

@torch.no_grad()
def compute_gender_direction(
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    model_name: str,
    ft: bool,
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
    unit_vector: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings_diffs = []
    for masc, fem in gender_phrases:
        lambda_m = compute_lambdas([masc], model, tokenizer, layer, model_name, ft, system_prompt, instruction_in_prompt)[0]
        lambda_f =  compute_lambdas([fem], model, tokenizer, layer, model_name, ft, system_prompt, instruction_in_prompt)[0]
        embeddings_diffs.append((lambda_f - lambda_m).unsqueeze(0))
    embeddings_diffs = torch.cat(embeddings_diffs, dim=0)
    v_gender, v_gender_mean = estimate_single_dir_from_embeddings(embeddings_diffs, unit_vector=unit_vector)  # direction de genre
    return v_gender, v_gender_mean
