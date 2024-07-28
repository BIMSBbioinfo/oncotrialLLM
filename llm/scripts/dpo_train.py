"""
Description: This script fine-tunes the open-source model NousResearch/Hermes-2-Pro-Mistral-7B using Direct Preference Optimization (DPO) for the task of genomic biomarker extraction from clinical trial data. It utilizes Hugging Face's Transformers library, PEFT (Prompt Engineering for Transformers), and TRL (Text Representation Learning) for fine-tuning and training.

Usage:
python script_name.py

Note:
Ensure that the specified pre-trained model and tokenizer are available and compatible. Adjust training arguments and configurations as needed based on resource constraints and task requirements.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from datasets import Dataset
from trl import DPOTrainer
import torch
import hydra

import json
import gc

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    # model
    model_name = cfg.DPO_FT.open_source_model
    new_model = cfg.DPO_FT.fine_tuned_model

    # read jsonl file
    with open(cfg.DPO_FT.fine_tuning_train, 'r') as file:
        data = [json.loads(line) for line in file]

    system_prompt = f"""<|im_start|>system\nYou are a helpful assistant that extracts only genomic biomarkers from the supplied clinical trial data and responds in JSON format. Here's the json schema you must adhere to:<schema>{{\"inclusion_biomarker\": [[]], \"exclusion_biomarker\": [[]]}}</schema>\nIn this context, limit the extraction of genomic biomarkers to the following categories: gene alteration (mutation, fusion, rearrangement, copy number alteration, deletion, insertion, translocation), pathway alterations, gene expression, protein expression, pathway expression, HLA, TMB (tumor molecular burden, TMB-H or TMB-L), MSI (microsatellite instability, MSI-H, MSI-L, MSS, microsatellite stable) status, gene pathway alteration like dMMR (deficient Mismatch Repair Pathway) or pMMR (proficient Mismatch Repair), and protein status (HER2, ER, PgR, PD-L1).\n\nDo not extract non-genomic biomarkers, which refer to any indicators not directly related to genetic or genomic information. Ignore information such as age, medical conditions, potential pregnancy, disease stage, allergies, treatment history, drugs, therapies, treatment, histology, and tumor cancer types, diseases, HIV, infections, and more. Also, ignore information about levels, scores, doses, expression ratios, and illnesses. Do not consider biomarkers related to model experimental animals, historical data, or previous studies.\n\nPreserve logical connections (AND, OR) between genomic biomarkers. Group 'AND'-linked genomic biomarkers in the same list, and place 'OR'-linked genomic biomarkers in separate lists. Treat main bullets in \"Inclusion Criteria\" as AND logic, and \"Exclusion Criteria\" as OR logic, unless specified otherwise. Handle ambiguous logic in the sentence as OR.\n\nEnsure each genomic biomarker is a string with the gene name preceding the variant. Remove the words \"gene\", \"allele\", \"status\", and \"mutation\" (when a specific variant is given). Make the variant singular and noun-based. Replace \"mutant\" with \"mutation\". Include a space between the gene name, its variant if they are connected. Include a space between the hormone name and its status if they are connected. Replace \"positive expression\" with \"expression\" and symbols \"-\" and \"+\" with \"negative\" and \"positive\" respectively, except in MSI status or known fusions separated by \"-\". Add \"germline\" or \"somatic\" terms in parentheses at the end of the corresponding biomarker. Ignore biomarkers mentioned as \"exceptions\" or after \"other than\". Handle synonyms in parentheses by extracting the genomic biomarker but ignoring the synonym. Extract each genomic biomarker once. Expand the genomic biomarkers when needed.\n\nTo summarize, extract only genomic biomarkers from the supplied clinical trial data, focusing on the categories mentioned above. Ignore any non-genomic biomarkers and unrelated information such as age, medical conditions, treatment history, cancer, drugs, therapies, histology, levels and scores. If no genomic biomarkers are found, return empty lists in JSON. Do not make assumptions or add biomarkers. Do not add any biomarkers that are not explicitly mentioned in the input, and do not make assumptions about potential genomic biomarkers. Ensure output list contains only lists of strings when there exist genomic biomarkers in the input, following this example: {{\"inclusion_biomarker\": [[\"GeneA variantA\"], [\"GeneX variantY]], \"exclusion_biomarker\": []}}. Do not \\escape. Do not repeat a genomic biomarker.<|im_end|>\n"""

    user = f"""<|im_start|>user\nExtract the genomic biomarker from the clinical trial below. Just generate the JSON object without explanation."""

    user_end = f"""\n<|im_end|>\n<|im_start|>assistant"""

    data_set_new = {"prompt": [], "chosen": [], "rejected": []}
    for da in data:
        data_set_new["prompt"].append(system_prompt + user + da["input"] + user_end)
        data_set_new["chosen"].append(str(da["output"]))
        data_set_new["rejected"].append(da["rejected"])

    # Convert data to Hugging Face dataset
    dataset = Dataset.from_dict(data_set_new)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA configuration https://huggingface.co/docs/peft/en/package_reference/lora
    peft_config = LoraConfig(
        r=cfg.LoRA.r,  # Lora attention dimension (the “rank”).
        lora_alpha=cfg.LoRA.lora_alpha,  # The alpha parameter for Lora scaling.
        lora_dropout=cfg.LoRA.lora_dropout,  # The dropout probability for Lora layers.
        bias="none",  # Bias type for LoRA.
        task_type="CAUSAL_LM",
        target_modules=cfg.LoRA.target_modules # The names of the modules to apply the adapter to.
    )
    # Quantization configuration for QLoRA https://huggingface.co/docs/transformers/en/main_classes/quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
        bnb_4bit_use_double_quant=True,  # This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.
        bnb_4bit_quant_type="nf4",  # This sets the quantization data type in the bnb.nn.Linear4Bit layers
        bnb_4bit_compute_dtype=torch.bfloat16  # This sets the computational type which might be different than the input type.
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

    # Training arguments https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.DPO_FT.per_device_train_batch_size,  # The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.
        gradient_accumulation_steps=cfg.DPO_FT.gradient_accumulation_steps,  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        gradient_checkpointing=True,    # To use gradient checkpointing to save memory at the expense of slower backward pass.
        learning_rate=cfg.DPO_FT.learning_rate,  # The initial learning rate for the optimizer
        lr_scheduler_type="cosine",
        max_steps=cfg.DPO_FT.max_steps,  # the total number of training steps to perform.
        save_strategy="no",
        logging_steps=cfg.DPO_FT.logging_steps,
        output_dir=new_model,
        optim="paged_adamw_32bit",  # The optimizer to use
        warmup_steps=cfg.DPO_FT.warmup_steps,  # Number of steps used for a linear warmup from 0 to learning_rate
        bf16=False  # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU.
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=cfg.DPO_FT.beta,
        max_length=cfg.DPO_FT.max_length,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()
    # Save artifacts
    dpo_trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")

    # Flush memory
    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)


if __name__ == "__main__":
    main()