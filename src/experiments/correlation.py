import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path
from utils.cli import str2bool, get_suffix_folder
from constants import SYSTEM_PROMPTS

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GENDER_BIAS_ROOT", DEFAULT_ROOT))



def compute_bias_score_from_predictions(csv_path, include_neutral=True):
    """
    Compute bias scores from model predictions in a CSV file.
    Args:
        csv_path: Path to the CSV file containing predictions.
        include_neutral: Boolean indicating whether to include neutral predictions in the bias score calculation.
    Returns:
        df_bias_score: DataFrame containing bias scores for each concept.
    """
    df = pd.read_csv(csv_path)
    grouped = df.groupby("concept")
    rows = []

    for concept_name, group in grouped:
        n_F = (group["predicted_gender"] == "F").sum()
        n_M = (group["predicted_gender"] == "M").sum()
        n_N = (group["predicted_gender"] == "neutral").sum()

        denom = n_F + n_M if not include_neutral else n_F + n_M + n_N
        score = 0 if denom == 0 else (n_F - n_M) / denom

        rows.append({
            "Concept": concept_name,
            "n_F": n_F,
            "n_M": n_M,
            "n_Neutral": n_N,
            "Bias_Score": score
        })

    df_bias_score = pd.DataFrame(rows)
    return df_bias_score

def get_correlation_pval(method, x, y):
    """
    Get the p-value for a correlation test.
    Args:
        method: The correlation method to use (e.g., "spearman").
        x: The first variable.
        y: The second variable.
    Returns:
        corr: The correlation coefficient.
        pval: The p-value for the correlation test.
    """
    if method == "spearman":
        return spearmanr(x, y)
    else:
        raise ValueError("Invalid method for correlation.")
    
def correlate_dot_gen(dot_folder, df_bias_score, concept, layer, method="spearman", similarity_type="dot_products"):
    """
    Correlate dot product scores with generation bias scores.
    Args:
        dot_folder: Folder containing dot product CSV files.
        df_bias_score: DataFrame containing bias scores for each concept.
        concept: The concept to analyze.
        layer: The layer number.
        method: The correlation method to use (default is "spearman").
        similarity_type: The type of similarity measure (default is "dot_products").
    Returns:
        corr: The correlation coefficient.
        pval: The p-value for the correlation test.
    """
    dot_file = os.path.join(dot_folder, f"{similarity_type}_{concept}_L{layer}.csv")

    df_dot = pd.read_csv(dot_file)

    df_merged = pd.merge(df_dot, df_bias_score, on="Concept", how="inner")


    x = df_merged["Mean"]
    y = df_merged["Bias_Score"]

    corr, pval = get_correlation_pval(method, x, y)

    print(f"Corrélation dot/gen ({method}): r = {corr:.3f}, p = {pval:.3g}")

    return corr, pval




def plot_correlation_by_layer(df_plot, concept, save_folder, method):
    """
    Plot correlation coefficients and p-values by layer for a given concept.
    Args:
        df_plot: DataFrame containing correlation results.
        concept: The concept to analyze.
        save_folder: Folder to save the plots.
        method: The correlation method used.
    """
    title_kind = {
        "dot_vs_gen": "dot product vs génération",
    }

    for sim in df_plot["SimilarityType"].unique():
        df_sim = df_plot[df_plot["SimilarityType"] == sim]

        # === PLOT 1: Correlations ===
        plt.figure(figsize=(8, 6))
        df_sub = df_sim[df_sim["Type"] == "dot_vs_gen"]
        if not df_sub.empty:
            plt.plot(df_sub["Layer"], df_sub["Correlation"],
                        marker="^", linestyle="-", label=title_kind["dot_vs_gen"])

        plt.xlabel("Layer")
        plt.ylabel("Corrélation")
        plt.title(f"Corrélation ({method.capitalize()}) par couche – {concept} – {sim}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.ylim(-1, 1)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)
        plt.savefig(os.path.join(save_folder, f"correlation_by_layer_{concept}_{sim}.png"))

        # === PLOT 2: p-values ===
        plt.figure(figsize=(8, 6))
        df_sub = df_sim[df_sim["Type"] == "dot_vs_gen"]
        if not df_sub.empty:
            plt.plot(df_sub["Layer"], df_sub["PValue"],
                        marker="^", linestyle="-", label=title_kind["dot_vs_gen"])

        plt.xlabel("Layer")
        plt.ylabel("p-value")
        plt.title(f"p-values ({method.capitalize()}) par couche – {concept} – {sim}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.axhline(0.05, color='red', linestyle='--', linewidth=1.2, label='p = 0.05')
        plt.savefig(os.path.join(save_folder, f"pvalues_by_layer_{concept}_{sim}.png"))
        plt.close()

    
def main(args):
    print(f"====== Correlation Analysis ({args.method}) ======")
    if args.use_model_ft_dpo and args.use_model_ft_sft:
        raise ValueError("Cannot use both DPO and LoRA fine-tuning at the same time.")
    
    suffix = f"{'_lora_dpo' if args.use_model_ft_dpo else ''}{'_lora_sft' if args.use_model_ft_sft else ''}"
    if SYSTEM_PROMPTS.get(args.system_prompt_key):
        suffix += f"_{'instruction_in_prompt' if args.instruction_in_prompt else 'instruction'}_{args.system_prompt_key}"
    suffix_output = "" if args.corr_previous_predictions else suffix

    suffix_folder = get_suffix_folder(args.instruction_in_prompt, args.system_prompt_key, args.model_name, args.use_model_ft_dpo, args.use_model_ft_sft)
    
    base_folder = ROOT / "results" / "correlation" / suffix_folder / ("prev_pred" if args.corr_previous_predictions else "")
    save_folder = base_folder / args.method
    save_folder.mkdir(parents=True, exist_ok=True)

    results_dot_ppl = []
    results_dot_logP_next = []
    results_dot_gen = []
    for similarity_type in args.similarity_types:
        for concept in args.list_concepts:
            for layer in args.list_layers:
                dot_folder = ROOT / "results" / "dot_prod" / f"{args.model_name}{suffix}" / similarity_type
                gen_csv = ROOT / "results" / "generation" / f"{args.model_name}{suffix_output}" / f"{concept}_llama70b_predictions.csv"

                try:
                    df_bias_score = compute_bias_score_from_predictions(
                        csv_path=str(gen_csv),
                        include_neutral=True,
                    )
                    corr_gen, pval_gen = correlate_dot_gen(
                        dot_folder=str(dot_folder),
                        df_bias_score=df_bias_score,
                        concept=concept,
                        layer=layer,
                        method=args.method,
                        similarity_type=similarity_type
                    )

                except Exception as e:
                    print(f"[WARN] dot_vs_gen failed for {concept}, L{layer}: {e}")
                    df_bias_score = None

                results_dot_gen.append({
                    "Concept": concept,
                    "SimilarityType": similarity_type,
                    "Layer": layer,
                    "Correlation": corr_gen,
                    "PValue": pval_gen,
                    "Type": "dot_vs_gen",
                })

    df_all = pd.DataFrame(results_dot_ppl + results_dot_logP_next + results_dot_gen)
    results_path = save_folder / f"correlation_summary_all_similarities{'_prev_pred' if args.corr_previous_predictions else ''}.csv"
    df_all.to_csv(results_path, index=False)
    print(f"Corrélations par layer sauvegardées dans {results_path}")

    for concept in args.list_concepts:
        df_plot = df_all[df_all["Concept"] == concept]
        plot_correlation_by_layer(df_plot, concept, str(save_folder), args.method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_model_ft_sft', type=str2bool, default=False)
    parser.add_argument('--use_model_ft_dpo', type=str2bool, default=False)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    parser.add_argument('--list_concepts', nargs='+',
                        default=['professions', 'colors', 'months'])
    parser.add_argument('--list_layers', nargs='+', default=[5], type=int)
    parser.add_argument('--corr_previous_predictions', type=str2bool, default=False,)
    parser.add_argument('--method', type=str, default='spearman')
    parser.add_argument('--similarity_types', nargs='+', default=['dot_products'])
    parser.add_argument('--system_prompt_key', type=str, default="none",
                        choices=["none"],
                        help='Type de system prompt à utiliser')
    parser.add_argument('--instruction_in_prompt', type=str2bool, default=False,
                   help='Put instruction directly in prompt rather than system prompt')
    args = parser.parse_args()
    main(args)