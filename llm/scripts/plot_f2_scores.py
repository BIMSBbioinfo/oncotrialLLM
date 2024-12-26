import hydra
import pandas as pd
from loguru import logger
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
            model = 'Hermes-FT-synth'
        else:
            model = 'Hermes-FT'
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
    try:
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
        else:
            raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")
        for file in result_files:
            try:
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
            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")
            df = pd.DataFrame(data)
        df[['model', 'prompt']] = df['file'].apply(parse_filename).apply(pd.Series)
        return df
    except Exception as e:
        logger.error(f"Error loading and processing data: {e}")
        raise


def plot_combined_dataframe(df, score_columns, model_order, prompt_order, filename, figure_title):
    try:
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df["prompt"] = pd.Categorical(df["prompt"], categories=prompt_order, ordered=True)
        df = df.sort_values(by=["model", "prompt"]).reset_index(drop=True)

        from matplotlib import font_manager

        # Specify the font path directly (if necessary)
        font_path = '.fonts/arial/ARIAL.TTF'
        prop = font_manager.FontProperties(fname=font_path)

        fig, axes = plt.subplots(1, len(score_columns), figsize=(7, 3.54), dpi=800, constrained_layout=True)

        bar_width = 0.35
        models = df["model"].unique()
        prompts = df["prompt"].unique()
        n = 2
        model_index = {model: i for i, model in enumerate(models)}
        colors = plt.cm.get_cmap('tab20', len(prompts))

        for ax, score_column, letter in zip(axes, score_columns, ['a.', 'b.']):
            for i, prompt in enumerate(prompts):
                prompt_data = df[df['prompt'] == prompt]
                bar_positions = [model_index[model] + (i / n) * bar_width for model in prompt_data['model']]
                ax.bar(bar_positions, prompt_data[score_column], bar_width / 2, alpha=0.95, label=f'{prompt}', color=colors(i))

            ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
            ax.set_xticklabels(['gpt-3.5-turbo', 'gpt-4', 'Hermes-2-Pro-\nMistral-7B', f"Hermes-FT", f'Hermes-FT-synth'], fontsize=8, fontproperties=prop)
            ax.set_xlabel('Models', fontsize=9, fontproperties=prop)
            ax.set_ylabel('F2 score', fontsize=9, fontproperties=prop)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=-90, ha="left", rotation_mode="anchor")

            # Add the letter to the bottom-left corner, further away from the plot
            ax.text(
                -0.1,  # X coordinate (slightly outside the plot area)
                -0.65, # Y coordinate (further below the x-axis)
                letter, 
                fontsize=10, 
                fontweight='bold', 
                transform=ax.transAxes, 
                va='center', 
                ha='center'
            )

            ax.legend(fontsize=6)
            
        #plt.suptitle(figure_title, fontsize=10)
        plt.tight_layout()

        plt.savefig(f"{filename}.pdf")
        plt.savefig(f"{filename}.pdf",
                    format='pdf',
                    bbox_inches='tight',  # To remove any unnecessary whitespace
                    dpi=600,               # High resolution for rasterized elements
                    transparent=True)      # Ensures a clean transparent background if needed

    except Exception as e:
        logger.error(f"Error plotting combined data: {e}")
        raise


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    try:
        df = load_and_process_data(cfg.data.results_dir)

        model_order = [
            'gpt-3.5-turbo',
            'gpt-4',
            'Hermes-2-Pro-Mistral-7B',
            'Hermes-FT',
            'Hermes-FT-synth'
        ]

        prompt_order = ['0S', 'PC', '1S', '2S']

        # Generate Figure 1
        plot_combined_dataframe(
            df, 
            score_columns=['inclusion_f2_score', 'exclusion_f2_score'], 
            model_order=model_order, 
            prompt_order=prompt_order, 
            filename=f"{cfg.figures.dir}/Figure1", 
            figure_title="Figure 1: Inclusion and Exclusion F2 Scores"
        )

        # Generate Figure 2
        plot_combined_dataframe(
            df, 
            score_columns=['dnf_inclusion_f2_score', 'dnf_exclusion_f2_score'], 
            model_order=model_order, 
            prompt_order=prompt_order, 
            filename=f"{cfg.figures.dir}/Figure2", 
            figure_title="Figure 2: DNF Inclusion and Exclusion F2 Scores"
        )

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    main()
