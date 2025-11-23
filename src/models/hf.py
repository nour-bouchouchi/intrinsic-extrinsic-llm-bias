import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft.tuners.lora import LoraLayer

import torch._dynamo
torch._dynamo.config.suppress_errors = True
def _fake_compile(model, *args, **kwargs):
    return model
torch.compile = _fake_compile

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))


def _get_active_adapters(model):
    """
    Returns the list of active adapters in the model. If no adapter is active, returns an empty list.
    Args:
        model: The PEFT model.
    Returns:
        List of active adapter names.
    """
    active = getattr(model, "active_adapter", None)
    if active is None:
        if hasattr(model, "peft_config") and len(model.peft_config):
            model.set_adapter(next(iter(model.peft_config)))
            active = model.active_adapter
        else:
            return []
    if isinstance(active, str):
        return [active]
    return list(active)


def _set_lora_scale(model, scale: float):
    """
    Set the LoRA scaling factor for the model.
    Args:
        model: The PEFT model.
        scale: Scaling factor for LoRA adapters.
    Returns:
        None
    Notes:
    scale = 0   : no LoRA effect (adapter disabled)
    scale = 1   : no change (adapter enabled)
    scale > 1   : amplify effect (adapter enabled)
    """
    # 0 : no LoRA effect (adapter disabled)
    if scale == 0:
        try:
            model.disable_adapter()
        except Exception:
            pass
        return

    adapters = _get_active_adapters(model)
    if not adapters:
        return  

    for m in model.modules():
        # We look for LoraLayer modules 
        has_lora = hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "lora_alpha") and hasattr(m, "r")
        if not has_lora:
            continue

        for name in adapters:
            # lora_alpha/r can be scalars OR dicts indexed by adapter
            la = m.lora_alpha[name] if isinstance(m.lora_alpha, dict) else m.lora_alpha
            r  = m.r[name]          if isinstance(m.r, dict)          else m.r
            if r is None or r == 0:
                continue
            base = la / r  # facteur appris (alpha/r)

            # scaling can be scalar or dict indexed by adapter
            if hasattr(m, "scaling"):
                if isinstance(m.scaling, dict):
                    m.scaling[name] = base * scale
                else:
                    m.scaling = base * scale


def get_model(model_name: str, use_model_ft_dpo: bool, use_model_ft_sft: bool, lora_scale: float | None = None):
    """
    Load a HuggingFace model and tokenizer, optionally with PEFT fine-tuning (DPO or SFT).
    Args:
        model_name: Name of the model.
        use_model_ft_dpo: Boolean indicating if DPO fine-tuning should be used.
        use_model_ft_sft: Boolean indicating if SFT fine-tuning should be used.
        lora_scale: Optional scaling factor for LoRA adapters.
    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    is_ft = use_model_ft_dpo or use_model_ft_sft

    if is_ft:
        ft_folder = "models_dpo" if use_model_ft_dpo else "models_sft"
        path = str(ROOT / ft_folder / model_name)

        tokenizer = AutoTokenizer.from_pretrained(path)
        peft_config = PeftConfig.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, path)
        
        if lora_scale is not None:
            _set_lora_scale(model, lora_scale)

        try:
            model = torch._dynamo.disable(model)
        except Exception:
            pass

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def merge_peft_model(model_name: str, peft_type: str) -> str:
    """
    Merge a PEFT model with its base model.
    Args:
        model_name: Name of the base model.
        peft_type: Type of PEFT model ('dpo' or 'sft').
    Returns:
        save_path: Path to the merged model.    
    """
    peft_path = str(ROOT / f"models_{peft_type}" / model_name)
    tokenizer = AutoTokenizer.from_pretrained(peft_path)
    peft_config = PeftConfig.from_pretrained(peft_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    merged_model = peft_model.merge_and_unload()

    save_path = ROOT / "results" / "merged" / f"{model_name.replace('/','_')}_{peft_type}"
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    return str(save_path)
