import torch
import argparse
import os
import shutil
import matplotlib.pyplot as plt
import json
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  

from utils.training import set_global_seed, print_gpu_memory
from utils.cli import str2bool

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def format_sft_genderalign():
    """
    Format the GenderAlign dataset for SFT training.
    Returns:
        sft_data: List of dictionaries with 'input' and 'output' keys.
    """
    sft_data = []
    dataset = load_dataset("json", data_files=str(ROOT / "datasets" / "GenderAlign.json"))["train"]
    
    for example in dataset:
        try:
            # Prompt extraction
            prompt = example["chosen"].split("Assistant:")[0].replace("Human:", "").strip()

            # Expected answer = version "chosen"
            output = example["chosen"].split("Assistant:", 1)[1].strip()
            
            if len(prompt) == 0 or len(output) == 0:
                continue

            sft_data.append({
                "input": prompt,
                "output": output
            })
        except Exception as e:
            print(f"Erreur de parsing : {e}")
    
    return sft_data


def format_sft_winobias():
    """
    Format the Winobias dataset for SFT training.
    Returns:
        sft_data: List of dictionaries with 'input' and 'output' keys.
    """
    def extract_doc_id(document_id: str):
        parts = document_id.split("/")
        split = parts[1].split("_")[0]  
        num = document_id.split("//")[-1]
        return f"{split}{num}"

    def load_winobias_sentences(subset_name: str):
        dataset_dict = load_dataset("uclanlp/wino_bias", subset_name)
        dataset = concatenate_datasets([dataset_dict["validation"], dataset_dict["test"]])
        sentences = {}
        for entry in dataset:
            num_doc_id = extract_doc_id(entry["document_id"])     
            sentence = " ".join(entry["tokens"])
            sentences[num_doc_id] = sentence
        return sentences

    def make_blank_prompt(pro_sentence, anti_sentence):
        tokens_pro = pro_sentence.split()
        tokens_anti = anti_sentence.split()
        if len(tokens_pro) != len(tokens_anti):
            return None

        diffs = [(i, w1, w2) for i, (w1, w2) in enumerate(zip(tokens_pro, tokens_anti)) if w1 != w2]

        allowed_pairs = {
            ('he', 'she'), ('she', 'he'),
            ('his', 'her'), ('her', 'his'),
            ('him', 'her'), ('her', 'him'),
            ('herself', 'himself'), ('himself', 'herself')
        }

        if len(diffs) == 1:
            idx, word1, word2 = diffs[0]
            prompt_tokens = tokens_pro.copy()
            prompt_tokens[idx] = f"[{word1}/{word2}]"
        elif len(diffs) == 2:
            words = tuple((w1.lower(), w2.lower()) for _, w1, w2 in diffs)
            if all(pair in allowed_pairs for pair in words):
                prompt_tokens = tokens_pro.copy()
                for idx, w1, w2 in diffs:
                    prompt_tokens[idx] = f"[{w1}/{w2}]"
            else:
                return None
        else:
            return None

        sentence_with_blank = " ".join(prompt_tokens)
        instruction = "Avoid gender assumptions when processing information about people and roles. Challenge common stereotypes.\n\nComplete this sentence: "
        return f"{instruction}{sentence_with_blank}"

    sft_dataset = []

    for type_label in ["type1", "type2"]:
        pro_data = load_winobias_sentences(f"{type_label}_pro")
        anti_data = load_winobias_sentences(f"{type_label}_anti")

        for doc_id in pro_data:
            if doc_id in anti_data:
                sent_pro = pro_data[doc_id]
                sent_anti = anti_data[doc_id]
                prompt = make_blank_prompt(sent_pro, sent_anti)
                if prompt:
                    sft_dataset.append({
                        "input": prompt,
                        "output": sent_anti.strip()
                    })

    return sft_dataset


def format_sft_bbq():
    """
    Format the BBQ dataset for SFT training.
    Returns:
        sft_data: List of dictionaries with 'input' and 'output' keys.
    """
    def map_answer_info(example):
        dico_answer_group = {
            'unknown': "unknown", 'F': 'F', 'M': 'M', 'boy': 'M', 'girl': 'F', 
            'man': 'M', 'nonTrans': 'nonTrans', 'nonTrans_F': 'nonTrans_F', 
            'nonTrans_M': 'nonTrans_M', 'trans': 'trans', 'trans_F': 'trans_F', 
            'trans_M': 'trans_M', 'woman': "F",
        }

        answer_info = example["answer_info"]
        return {
            "ans0_group": dico_answer_group.get(answer_info["ans0"][1], "unknown"),
            "ans1_group": dico_answer_group.get(answer_info["ans1"][1], "unknown"),
            "ans2_group": dico_answer_group.get(answer_info["ans2"][1], "unknown"),
        }
    
    def create_full_answer(answer_group, answer, question):
        if "who" in question.lower():
            if answer_group == 'M' or answer_group == 'F':
                return f"Based on the context, {answer.lower()} is the person being referred to."
            elif answer_group == 'unknown':
                return "The context doesn't provide enough information to determine who is being referred to."
            else: 
                return f"Based on the information provided, the answer is: {answer.lower()}."
        else:
            return f"Based on the information provided, the answer is: {answer.lower()}."

    ds = load_dataset("json", data_files=str(ROOT / "datasets" / "bbq" / "data" / "Gender_identity.jsonl"), split="train")
    sft_data = []

    for ex in ds:
        groups = ex["additional_metadata"]["stereotyped_groups"]
        if groups not in [['F'], ['M']]:
            continue

        answer_groups = map_answer_info(ex)
        stereotype_group = groups[0]
        polarity = ex["question_polarity"]
        context_type = ex["context_condition"]
        label = ex["label"]

        correct_answer = ex[f"ans{label}"]
        correct_group = answer_groups[f"ans{label}_group"]

        # Add coherent instruction
        base_prompt = f"{ex['context'].strip()}\n\nQuestion: {ex['question'].strip()}"
        prompt = f"Be fair and unbiased. Avoid making assumptions based on stereotypes.\n\n{base_prompt}"

        if context_type == "ambig":
            for i in range(3):
                if answer_groups[f"ans{i}_group"] == "unknown":
                    output = create_full_answer(answer_groups[f"ans{i}_group"], ex[f"ans{i}"], ex["question"])
                    sft_data.append({
                        "input": prompt,
                        "output": output
                    })
                    break

        elif context_type == "disambig":
            if (polarity == "neg" and correct_group != stereotype_group and correct_group != "unknown") or \
               (polarity == "nonneg" and correct_group == stereotype_group):
                
                output = create_full_answer(correct_group, correct_answer, ex["question"])
                sft_data.append({
                    "input": prompt,
                    "output": output
                })

    return sft_data


def format_sft_crowspairs():
    """
    Format the Crowspairs dataset for SFT training.
    Returns:
        sft_data: List of dictionaries with 'input' and 'output' keys.
    """

    df = pd.read_csv(str(ROOT / "datasets" / "crows-pairs" / "data" / "crows_pairs_anonymized.csv"))

    sft_data = []

    df = df[df["bias_type"] == "gender"]

    for _, row in df.iterrows():
        direction = row["stereo_antistereo"]

        if direction == "antistereo":
            chosen = row["sent_less"]    
            rejected = row["sent_more"]
        else: 
            continue  
        
        prompt = f"Avoid reinforcing stereotypes.\n\nRephrase this: {rejected}"

        sft_data.append({
            "input": prompt,
            "output": chosen
        })

    return sft_data


def format_sft_stereoset():
    """
    Format the StereoSet dataset for SFT training.
    Returns:
        sft_data: List of dictionaries with 'input' and 'output' keys.
    """
    sft_data = []

    for inter in [True, False]:
        if inter:
            ds = load_dataset("McGill-NLP/stereoset", "intersentence")["validation"]
        else:
            ds = load_dataset("McGill-NLP/stereoset", "intrasentence")["validation"]

        for example in ds:
            if example["bias_type"].lower() != "gender":
                continue

            sentences = example["sentences"]["sentence"]
            labels = example["sentences"]["gold_label"]

            try:
                idx_stereo = labels.index(1)
                idx_anti = labels.index(0)
            except ValueError:
                continue  # skip if labels are missing

            if inter:
                # Intersentence : generate continuation
                prompt = f"Maintain objectivity and avoid making assumptions based on stereotypes.\n\nContinue this story: {example['context']}"
            else:
                # Intrasentence : complete a sentence
                prompt = f"Maintain objectivity and avoid making assumptions based on stereotypes.\n\nComplete this sentence: {example['context']}"
            answer = sentences[idx_anti]

            sft_data.append({
                "input": prompt.strip(),
                "output": answer.strip()
            })

    return sft_data


def load_all_dataset(selected_datasets, seed=42, eval_ratio=0.1):
    """
    Load and combine selected datasets for SFT training and evaluation.
    Args:
        selected_datasets (list): List of dataset names to load.
        seed (int): Random seed for reproducibility.
        eval_ratio (float): Proportion of the dataset to use for evaluation.
    Returns:
        Tuple[Dataset, Dataset]: Training and evaluation datasets.
    """
    dataset_functions = {
        "genderalign": format_sft_genderalign,
        "winobias": format_sft_winobias,
        "bbq": format_sft_bbq,
        "crowspairs": format_sft_crowspairs,
        "stereoset": format_sft_stereoset
    }

    train_splits = []
    eval_splits = []

    for dataset_name in selected_datasets:
        if dataset_name not in dataset_functions:
            raise ValueError(f"Dataset inconnu: {dataset_name}")

        data = dataset_functions[dataset_name]()
        hf_dataset = Dataset.from_list(data)
        split = hf_dataset.train_test_split(test_size=eval_ratio, seed=seed)

        train_splits.append(split["train"])
        eval_splits.append(split["test"])

    if len(train_splits) > 1:
        train_dataset = concatenate_datasets(train_splits)
        eval_dataset = concatenate_datasets(eval_splits)
    else :
        train_dataset = train_splits[0]
        eval_dataset = eval_splits[0]

    return train_dataset, eval_dataset


def plot_metrics(log_file, output_dir="."):
    with open(log_file) as f:
        logs = json.load(f)

    log_history = logs.get("log_history", [])

    # Get loss values and steps
    steps_loss = [x["step"] for x in log_history if "step" in x and "loss" in x]
    loss_values = [x["loss"] for x in log_history if "loss" in x]

    steps_eval_loss = [x["step"] for x in log_history if "step" in x and "eval_loss" in x]
    eval_loss_values = [x["eval_loss"] for x in log_history if "eval_loss" in x]

    # Plot the curves
    if steps_loss and loss_values:
        plt.figure(figsize=(8, 5))
        plt.plot(steps_loss, loss_values, label="Training Loss", marker='o')
        if steps_eval_loss and eval_loss_values:
            plt.plot(steps_eval_loss, eval_loss_values, label="Eval Loss", marker='x')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training & Evaluation Loss over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()
    else:
        print("Aucune donnée de loss trouvée dans le fichier de log.")


    



def preprocess(example, tokenizer):
    """
    Preprocess a single example for causal language modeling with labels.
    Args:
        example: A dictionary with 'input' and 'output' keys.
        tokenizer: The tokenizer to use.
    Returns:
        A dictionary with tokenized 'input_ids', 'attention_mask', and 'labels'.
    """
    input_text = example["input"].strip()
    output_text = example["output"].strip()
    
    input_with_space = input_text + " "
    full_text = input_text + " " + output_text
    
    # SAME method and SAME tokenizer settings as in training/eval scripts
    input_tokenized = tokenizer(input_with_space, truncation=True, padding=False, add_special_tokens=True)
    full_tokenized = tokenizer(full_text, truncation=True, padding=False, add_special_tokens=True)
    
    input_length = len(input_tokenized["input_ids"])
    labels = full_tokenized["input_ids"].copy()
    
    if input_length > 0 and input_length < len(labels):
        labels[:input_length] = [-100] * input_length
    
    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels
    }
    

    
def apply_lora_to_model(model):
    """
    Apply LoRA adapters to the model for fine-tuning.
    Args:
        model: The base model.
    Returns:
        model: The model with LoRA adapters applied.
    """
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=128, 
        lora_alpha=256, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    return model

class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded
        }


def train_lora(model_name, train_dataset, eval_dataset, output_dir, overwrite, use_quantization=False):
    """
    Train a LoRA fine-tuned model using the provided datasets.
    Args:
        model_name: Name of the base model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        output_dir: Directory to save the model and logs.
        overwrite: Whether to overwrite the output directory.
        use_quantization: Whether to use quantization.
    """
    seed = 42 
    set_global_seed(seed)
    
    if os.path.exists(output_dir) and overwrite:
        print(f"Le dossier {output_dir} existe déjà. Suppression pour réentraînement.")
        shutil.rmtree(output_dir)



    print("Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Preprocess datasets...")
    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: preprocess(x, tokenizer), remove_columns=eval_dataset.column_names)
    
    print("Load model...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    if "gemma-2" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
    
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  
        )
        model_kwargs["quantization_config"] = bnb_config
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.gradient_checkpointing_enable()

    model = apply_lora_to_model(model)
    model.config.use_cache = False


    print("Configuration de l'entraînement...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200, 
        save_total_limit=2,
        bf16=True,
        seed=seed,
        remove_unused_columns=False,
        report_to="none",
        overwrite_output_dir=overwrite,
        warmup_ratio=0.1,
        weight_decay=0.005,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit" if use_quantization else "adamw_torch",
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True, 
        gradient_checkpointing=True if not use_quantization else False,  
        max_grad_norm=1.0,  
        ddp_find_unused_parameters=False, 
    )

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer)

    print("Init Trainer...")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train()

    print("Training complete.")

    print("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}.")


    log_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(log_path):
        plot_metrics(log_path, output_dir)
    else:
        print("No log file found, no metrics plotted.")


def main(args):
    print("List datasets :", args.list_dataset)
    print(type(args.list_dataset))
    train_dataset, eval_dataset = load_all_dataset(args.list_dataset, seed=42, eval_ratio=0.1)
    
    train_lora(
        model_name = args.model_name,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        output_dir = str(ROOT / "models_sft" / args.model_name),
        overwrite = args.overwrite,
        use_quantization = args.quantize
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--quantize', type=str2bool, default=False, 
                       help='Utiliser la quantification 4-bit pour économiser la mémoire')
    parser.add_argument(
        '--list_dataset',
        nargs='+',
        default=['genderalign', 'crowspairs', 'winobias', 'stereoset', 'bbq'], # 'bbq'
        help="Liste des datasets à utiliser"
    )
    args = parser.parse_args()
    main(args)