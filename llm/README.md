# Reproducibility Guide

## Configuration

Make sure to configure the corresponding variables in the [config.yaml](conf/config.yaml) file as required for each section. I will also share the specific variables in the config that can be modified for each script.


> [!NOTE]
> If the fine-tuning datasets or fine-tuned models are not found locally, the script will load them from [HuggingFace](https://huggingface.co/nalkhou)

Make sure to navigate to the `llm` directory to run the scripts.

## Process CIViC Dataset
<details>
<summary>Downloding CIVIC Dataset</summary>

To download the [CIViC](https://civicdb.org/welcome) Dataset that has the biomarkers, run the command below:

```
wget -P data/raw https://civicdb.org/downloads/01-Dec-2023/01-Dec-2023-VariantSummaries.tsv
```
</details>


<details>
<summary>CIViC Configuration</summary>

<pre>
<span style="color:purple;">civic:</span>
  <span style="color:purple;">raw_file:</span> ${data.raw_dir}/01-Dec-2023-VariantSummaries.tsv
  <span style="color:purple;">processed_file:</span> ${data.processed_dir}/civic_processed.csv
  <span style="color:purple;">variant_syn_file:</span> ${data.processed_dir}/variants_synonyms.csv
  <span style="color:purple;">gene_syn_file:</span> ${data.processed_dir}/gene_synonyms.csv
</pre>
</details>

<br>
After making sure all the configurations are correct for CIViC, run the command below to process the data:

```
python -m scripts.process_civic
```

This should generate three csv files: 
1. `data/processed/civic_processed.csv`
2. `data/processed/gene_synonyms.csv`
3. `data/processed/variants_synonyms.csv`


## Reproduce GENIE analysis

<details>

<summary> Downloding AACR Project GENIE data (15.1-public)</summary>

To download the [AACR Project GENIE data]((https://www.aacr.org/professionals/research/aacr-project-genie/aacr-project-genie-data/)) make sure to first register and follow the steps in [SYNAPSE](https://genie.synapse.org/Access). Make sure to donwload the correct release, <i>15.1-public</i>.

```
wget -P data/raw https://civicdb.org/downloads/01-Dec-2023/01-Dec-2023-VariantSummaries.tsv
```
</details>

<details>
<summary>AACR GENIE Configuration</summary>

<pre>
<span style="color:purple;">aacr:</span>
  <span style="color:purple;">clinical_sample:</span> ${data.raw_dir}/aacr_genie/data_clinical_sample.txt
  <span style="color:purple;">data_mutations:</span> ${data.raw_dir}/aacr_genie/data_mutations_extended.txt
  <span style="color:purple;">data_cna:</span> ${data.raw_dir}/aacr_genie/data_CNA.txt
  <span style="color:purple;">data_sv:</span> ${data.raw_dir}/aacr_genie/data_sv.txt
</pre>
</details>

<br>
1. To process the data and compute the percentage of patients having at least one biomarker found in CIViC run the script below. This should print out the percentage and generate `data/processed/patient_with_biomarkers.csv` containing the list of the matching patients with information about their clinical and mutational profile.  <br>

<b> Make sure you've downloaded and processed the CIViC dataset since the aacr analysis is dependent on its list of biomarkers </b>

```
python -m scripts.aacr_analysis.py
```

2. To generate patient cancer distrubtion run the script below:

```
python -m scripts.plot_cancer_patient_distribution
```
This should save the plot in `figures`


## Hermes-2-Pro-Mistral Fine-tuning with Direct Preferenne Optimization

<details>
<summary>DPO Fine-tuning Configuration</summary>

<pre>
<span style="color:purple;">DPO_FT:</span>
  <span style="color:purple;">open_source_model:</span> NousResearch/Hermes-2-Pro-Mistral-7B
  <span style="color:purple;">fine_tuned_model:</span> Hermes-FT
  <span style="color:purple;">fine_tuning_train:</span> ${data.processed_dir}/negative.jsonl
  <span style="color:purple;">fine_tuning_test:</span> ${data.processed_dir}/ft_test.jsonl
  <span style="color:purple;">beta:</span> 0.1
  <span style="color:purple;">learning_rate:</span> 5e-5
  <span style="color:purple;">max_steps:</span> 200
  <span style="color:purple;">warmup_steps:</span> 100
  <span style="color:purple;">per_device_train_batch_size:</span> 1
  <span style="color:purple;">gradient_accumulation_steps:</span> 4
  <span style="color:purple;">logging_steps:</span> 1
  <span style="color:purple;">max_length:</span> 4800

<span style="color:purple;">LoRA:</span>
  <span style="color:purple;">r:</span> 2
  <span style="color:purple;">lora_alpha:</span> 4
  <span style="color:purple;">lora_dropout:</span> 0.05
  <span style="color:purple;">target_modules:</span> ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'] 
</details>
</pre>

<br>

> [!NOTE]
> - To repeat the fine-tuning for <b><i>Hermes-FT</i></b>, fine-tuned with the manually annotated dataset only, you should use the training set `data/processed/negative.jsonl`.
<br> However, to repeat the fine-tuning for the model <b><i>Hermes-synth-FT</i></b> you should change the `fine_tuned_model` to <i>Hermes-synth-FT</i> and the `fine_tuned_train` to `${data.simulated_dir}/negative.jsonl`


After finishing configuring the variables run the command below to start the training:

```
python -m scripts.dpo_train
```

## Hermes-2-Pro-Mistral Evaluation

<details> 
<summary>Hermes Evaluation</summary>

```
HERMES_EVAL:
  open_source_model: Hermes-FT
  test_set: ${data.processed_dir}/ft_test.jsonl
  open_source_eval_file: Fine_tune_a_Mistral_7b_model_with_DPO_zero-shot_zero-shot_loong_r_2_alpha_4.json # Changes depending on what we are evaluating!
```

You can choose to evaluate the open source model <i>Hermes-synth-FT</i> or even the base model <i>NousResearch/Hermes-2-Pro-Mistral-7B</i>

</details> 

<br>

To run the evaluation, use the command below:

```
python -m scripts.evaluate_hermes_models
```


## OpenAI models Evaluation

<details> 
<summary>GPT models Evaluation</summary>

```
GPT_EVAL:
  n_shot: 0
  model: gpt-3.5-turbo
  test_set: ${data.processed_dir}/ft_test.jsonl
  train_set: ${data.processed_dir}/ft_train.jsonl
  OUTPUT_PROMPTS:
    zero_shot: 0shot
    one_shot: 1shot
    two_shot: 2shot
    prompt_chain: 2CoP
  
PROMPT_FILES:
  gpt_zero_shot: prompts/zero-shot.json
  gpt_one_shot: prompts/one-shot.json
  gpt_two_shot: prompts/two-shot.json
  gpt_chain_one: prompts/chain_1.json
  gpt_chain_two: prompts/chain_2.json
```

<br>

You can configure which OpenAI model to use for evaluation and specify whether to use few-shot learning (with n_shot set to 0, 1 or 2). Select the desired n_shot, and the script will automatically use the corresponding prompt file.

</details>

<br>

> [!IMPORTANT]
> - Ensure you set your `OPENAI_API_KEY` variable. For example, you can do this by running: export `OPENAI_API_KEY="your_api_key"`

<br>

1. To evaluate the model with zero-shot and few-shot prompting, run the command below:
```
python -m scripts.evaluate_gpt_fewshots.py
```

2. To evaluate the model with prompt chaining, run the command below:

```
python -m scripts.evaluate_gpt_chain_of_prompts
```

