from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import contextlib
import functools
from torch import Tensor


def get_format_block_names(model_name: str, layer: int, ft: bool = False) -> str:
    """
    Get the formatted block name for a given layer in the model.
    Args:
        model_name: Name of the model.
        layer: Layer number.
        ft: Boolean indicating if the model is fine-tuned.
    Returns:
        block_name: Formatted block name as a string.s
    """
    if ft:
        return f"_orig_mod.base_model.model.model.layers.{layer}"
    else:
        return f"model.layers.{layer}"


def format_with_system_prompt(
    tokenizer: Any,
    text: Union[str, List[str]],
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
):
    """
    Format the input text with a system prompt if provided.
    Args:
        tokenizer: The tokenizer for the model.
        text: The input text or list of texts to format.
        system_prompt: Optional system prompt to use.
        instruction_in_prompt: Boolean indicating if instruction is in the prompt.
    Returns:
        formatted_text: The formatted text or list of texts.
    """
    if system_prompt is None:
        return text

    if instruction_in_prompt:
        if isinstance(text, list):
            return [f"{system_prompt}\n {t}" for t in text]
        return f"{system_prompt}\n {text}"

    if hasattr(tokenizer, 'apply_chat_template'):
        def _one(t: str) -> str:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": t}
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return f"System: {system_prompt}\nUser: {t}\nAssistant:"
        if isinstance(text, list):
            return [_one(t) for t in text]
        return _one(text)
    else:
        if isinstance(text, list):
            return [f"System: {system_prompt}\n{t}" for t in text]
        return f"System: {system_prompt}\n{text}"

def run_with_cache_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str] | str,
    layer: int,
    model_name: str,
    ft: bool,
    system_prompt: str | None = None,
    instruction_in_prompt: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run the model with a forward hook to capture activations at a specified layer.
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        texts: The input text or list of texts to process.
        layer: The layer number to extract activations from.
        model_name: The name of the model.
        ft: Boolean indicating if the model is fine-tuned.
        system_prompt: Optional system prompt to use.
        instruction_in_prompt: Boolean indicating if instruction is in the prompt.
    Returns:
        activations: Dictionary containing the captured activations.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations["hook_resid_post"] = output.detach()

    block_name = get_format_block_names(model_name, layer, ft)
    named = dict(model.named_modules())
    if block_name not in named:
        raise KeyError(f"Module not found: {block_name}")
    handle = named[block_name].register_forward_hook(hook_fn)

    formatted_texts = format_with_system_prompt(
        tokenizer, texts, system_prompt, instruction_in_prompt
    )
    inputs = tokenizer(
        formatted_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()
    return activations



@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]] = [],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]] = [],
    **kwargs
):
    """
    Context manager for temporarily adding hooks to modules.
    Args:
        module_forward_pre_hooks: List of tuples (module, hook) for forward pre-hooks.
        module_forward_hooks: List of tuples (module, hook) for forward hooks.
        **kwargs: Additional keyword arguments to pass to the hooks.
    Yields:
        None
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    """
    Get a forward pre-hook function for direction ablation on input activations.
    Args:
        direction: The direction vector to ablate.
    Returns:
        hook_fn: The hook function for ablation.
    """
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)

        # Projection
        ablated = activation - (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (ablated, *input[1:])
        else:
            return ablated

    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    """
    Get a forward hook function for direction ablation on output activations.
    Args:
        direction: The direction vector to ablate.
    Returns:
        hook_fn: The hook function for ablation.
    """
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)

        ablated = activation - (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (ablated, *output[1:])
        else:
            return ablated

    return hook_fn
