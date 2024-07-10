import hydra
import pandas as pd
import matplotlib.pyplot as plt

import os

from utils.jsons import load_json


# Function to parse filename
def parse_filename(filename):
    if 'gpt-3.5-turbo' in filename:
        model = 'gpt-3.5-turbo'
    elif 'gpt-4' in filename:
        model = 'gpt-4'
    elif 'Fine_tune_a_Mistral_7b' in filename:
        if 'sythetic_new' in filename:
            model = 'Hermes-2-Pro-Mistral-7B_DPO-156'
        else:
            model = 'Hermes-2-Pro-Mistral-7B_DPO-92'
    elif "Mistral" in filename:
        model = "Hermes-2-Pro-Mistral-7B"
    else:
        model = 'Unknown'
    
    if '0shot' in filename or 'zero-shot' in filename:
        prompt = '0S'
    elif '1shot' in filename:
        prompt = '1S'
    elif '2shot' in filename:
        prompt = '2S'
    elif '2CoP' in filename:
        prompt = 'PC'
    else:
        prompt = 'Unknown'
    return model, prompt


def load_and_process_data(results_dir):
    data = []
    if os.path.exists(results_dir):
        result_files = os.listdir(results_dir)
    else:
        raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")
    for file in result_files:
        filepath = os.path.join(results_dir, file)
        result = load_json(filepath)['results'][-1]
        inc_f2_score = result.get('Inclusion F2', None)[0]
        exc_f2_score = result.get('Exclusion F2', None)[0]
        dnf_inc_f2_score = result.get('Inclusion DNF F2', None)[0]
        dnf_exc_f2_score = result.get('Exclusion DNF F2', None)[0]
        if inc_f2_score and exc_f2_score:
            data.append({
                "file": file,
                "inclusion_f2_score": inc_f2_score,
                "exclusion_f2_score": exc_f2_score,
                "dnf_inclusion_f2_score": dnf_inc_f2_score,
                "dnf_exclusion_f2_score": dnf_exc_f2_score,
            })
    df = pd.DataFrame(data)
    df[['model', 'prompt']] = df['file'].apply(parse_filename).apply(pd.Series)
    return df


def plot_dataframe(df, score_column, model_order, prompt_order, filename):
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["prompt"] = pd.Categorical(df["prompt"], categories=prompt_order, ordered=True)
    df = df.sort_values(by=["model", "prompt"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.54,3), dpi=800)
    bar_width = 0.35
    models = df["model"].unique()
    prompts = df["prompt"].unique()
    n = 2
    model_index = {model: i for i, model in enumerate(models)}
    colors = plt.cm.get_cmap('tab20', len(prompts))

    for i, prompt in enumerate(prompts):
        prompt_data = df[df['prompt'] == prompt]
        bar_positions = [model_index[model] + (i / n) * bar_width for model in prompt_data['model']]
        ax.bar(bar_positions, prompt_data[score_column], bar_width / 2, alpha=0.95, label=f'{prompt}', color=colors(i))

    ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
    ax.set_xticklabels(['gpt-3.5-turbo', 'gpt-4', 'Hermes-2-Pro-\nMistral-7B',f"Hermes-2-Pro-\nMistral-7B_DPO-92",f'Hermes-2-Pro-\nMistral-7B_DPO-156'])
    ax.set_xlabel('Models', fontsize=7)
    ax.set_ylabel('F2 Score', fontsize=7)

    # rotates labels and aligns them horizontally to left 
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-90, ha="left", rotation_mode="anchor")
    ax.legend(fontsize=5)
    plt.xticks(fontsize=5) 
    plt.yticks(fontsize=5)
    plt.tight_layout()

    plt.savefig(filename)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    df = load_and_process_data(cfg.data.results_dir)

    model_order = [
        'gpt-3.5-turbo',
        'gpt-4',
        'Hermes-2-Pro-Mistral-7B',
        'Hermes-2-Pro-Mistral-7B_DPO-92',
        'Hermes-2-Pro-Mistral-7B_DPO-156'
    ]

    prompt_order = ['0S', 'PC', '1S', '2S']

    score_columns = ['inclusion_f2_score', 'exclusion_f2_score', 'dnf_inclusion_f2_score', 'dnf_exclusion_f2_score']

    figures_dir = cfg.figures.dir
    filenames = [f'{figures_dir}/gpt_plot_models_inclusion.png', f'{figures_dir}/gpt_plot_models_exclusion.png',
                 f'{figures_dir}/gpt_plot_models_dnf_inclusion.png', f'{figures_dir}/gpt_plot_models_dnf_exclusion.png']
    
    for score_column, filename in zip(score_columns, filenames):
        plot_dataframe(df, score_column, model_order, prompt_order, filename)


if __name__ == "__main__":
    main()
