import torch
import hydra
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import os

from utils.jsons import load_json


def configure_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def plot_data_lengths(tokenized_train_set, tokenized_test_set, dpi):
    plt.rc('xtick', labelsize=5) 
    plt.rc('ytick', labelsize=5)
    lengths = [len(x['input_ids']) for x in tokenized_train_set]
    lengths += [len(x['input_ids']) for x in tokenized_test_set]
    fig = plt.figure(figsize=(3.54,2.36), dpi=dpi) # specify figure size and resolution
    n, bins, patches = plt.hist(lengths, bins=90, linewidth=0.5, ec="black", color='C0')
    patches[len(patches)-1].set_fc('r') # color the last bar, it is the outlier
    plt.xlabel('Token count', fontsize=7) # Modify x-axis
    plt.ylabel('Frequency', fontsize=7) # Modify y-axis
    return plt


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):

    fig_dpi = 400

    training_set = load_json(f"{cfg.data.interim_dir}/train_set.json")
    testing_set = load_json(f"{cfg.data.interim_dir}/test_set.json")

    tokenizer = configure_tokenizer(cfg.open_source_model)
    
    tokenized_train_set = [tokenizer(trial['document'] ) for trial in training_set['ids']]
    tokenized_test_set = [tokenizer(trial['document'] ) for trial in testing_set['ids']]

    plt = plot_data_lengths(tokenized_train_set, tokenized_test_set, fig_dpi)

    figures_dir = cfg.figures.dir
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    plt.savefig(f'{figures_dir}/token_count_hist_redbar_{fig_dpi}.png', bbox_inches='tight')


if __name__ == "__main__":
    main()