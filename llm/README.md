# Reproducibility Guide

## Configurations

```
export OPENAI_API_KEY="your_api_key"
```

------- Make sure to configure the corresponding variables in the config file [config.yaml](conf/config.yaml) as required. (mention that below for each part i will share the corresponding configuration section in that file that one could modify)

## Process CIViC Dataset

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
1. To process the data and compute the percentage of patients having at least one biomarker found in CIViC run the script below. This should print out the percentage and generate `data/processed/patient_with_biomarkers.csv` containing the list of the matching patients with information about their clinical and mutational profile.  

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
  <span style="color:purple;">fine_tuned_model:</span> nalkhou/Hermes-FT
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
> To repeat the fine-tuning for Hermes-FT, fine-tuned with the manually annotated dataset only, you should use the training set `data/processed/negative.jsonl`. However, to repeat the fine-tuning for the model <i>Hermes-synth-FT</i> you should change the `fine_tuned_train` to `${data.simulated_dir}/negative.jsonl`

After finishing configuring the variables run the command below to start the training:

```
python -m scripts.dpo_train
```

