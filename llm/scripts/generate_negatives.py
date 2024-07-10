"""
Description:
    This script generates negative examples for a given dataset using a pre-trained Language Model (LM). 
    It generates prompts based on predefined templates and tokenizes them using the LM's tokenizer. 
    The LM then generates responses, which are considered as negative examples. 
    The generated negative examples are stored in a JSONL file.

Output:
    The script generates negative examples based on the provided dataset and stores them in a JSONL file named 'negative.jsonl'.
    Each line in the JSONL file represents a negative example in JSON format.

Example:
    python generate_negative_examples.py
"""

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import os
import json

from utils.jsons import load_json


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    try:
        train_dataset = load_json(f"{cfg.data.processed_dir}/ft_train.jsonl")
    except Exception as e:
        print(f"Loading data from HuggingFace: {e}")
        dataset = load_dataset('nalkhou/clinical-trials', split=['train', 'validation', 'test'])
        train_dataset = dataset[0]

    print("Datasets Loaded")

    base_model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")


    def generate_and_tokenize_prompt(example):
        text = f"""<|im_start|>system\nYou are a helpful assistant that extracts only genomic biomarkers from the supplied clinical trial data and responds in JSON format. Here's the json schema you must adhere to:<schema>{{\"inclusion_biomarker\": [[]], \"exclusion_biomarker\": [[]]}}</schema>\nIn this context, limit the extraction of genomic biomarkers to the following categories: gene alteration (mutation, fusion, rearrangement, copy number alteration, deletion, insertion, translocation), pathway alterations, gene expression, protein expression, pathway expression, HLA, TMB (tumor molecular burden, TMB-H or TMB-L), MSI (microsatellite instability, MSI-H, MSI-L, MSS, microsatellite stable) status, gene pathway alteration like dMMR (deficient Mismatch Repair Pathway) or pMMR (proficient Mismatch Repair), and protein status (HER2, ER, PgR, PD-L1).\n\nDo not extract non-genomic biomarkers, which refer to any indicators not directly related to genetic or genomic information. Ignore information such as age, medical conditions, potential pregnancy, disease stage, allergies, treatment history, drugs, therapies, treatment, histology, and tumor cancer types, diseases, HIV, infections, and more. Also, ignore information about levels, scores, doses, expression ratios, and illnesses. Do not consider biomarkers related to model experimental animals, historical data, or previous studies.\n\nPreserve logical connections (AND, OR) between genomic biomarkers. Group 'AND'-linked genomic biomarkers in the same list, and place 'OR'-linked genomic biomarkers in separate lists. Treat main bullets in \"Inclusion Criteria\" as AND logic, and \"Exclusion Criteria\" as OR logic, unless specified otherwise. Handle ambiguous logic in the sentence as OR.\n\nEnsure each genomic biomarker is a string with the gene name preceding the variant. Remove the words \"gene\", \"allele\", \"status\", and \"mutation\" (when a specific variant is given). Make the variant singular and noun-based. Replace \"mutant\" with \"mutation\". Include a space between the gene name, its variant if they are connected. Include a space between the hormone name and its status if they are connected. Replace \"positive expression\" with \"expression\" and symbols \"-\" and \"+\" with \"negative\" and \"positive\" respectively, except in MSI status or known fusions separated by \"-\". Add \"germline\" or \"somatic\" terms in parentheses at the end of the corresponding biomarker. Ignore biomarkers mentioned as \"exceptions\" or after \"other than\". Handle synonyms in parentheses by extracting the genomic biomarker but ignoring the synonym. Extract each genomic biomarker once. Expand the genomic biomarkers when needed.\n\nTo summarize, extract only genomic biomarkers from the supplied clinical trial data, focusing on the categories mentioned above. Ignore any non-genomic biomarkers and unrelated information such as age, medical conditions, treatment history, cancer, drugs, therapies, histology, levels and scores. If no genomic biomarkers are found, return empty lists in JSON. Do not make assumptions or add biomarkers. Do not add any biomarkers that are not explicitly mentioned in the input, and do not make assumptions about potential genomic biomarkers. Ensure output list contains only lists of strings when there exist genomic biomarkers in the input, following this example: {{\"inclusion_biomarker\": [[\"GeneA variantA\"], [\"GeneX variantY]], \"exclusion_biomarker\": []}}. Do not \\escape. Do not repeat a genomic biomarker.<|im_end|>\n<|im_start|>user\nExtract the genomic biomarker from the clinical trial below. Just generate the JSON object without explanation.\n{example}\n<|im_end|>\n<|im_start|>assistant"""
        return tokenize(text)


    max_length = 4682  # This was an appropriate max length for my dataset

    # redefine the tokenize function and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize(prompt):
        result = tokenizer(
            prompt,
            max_length=max_length,
            return_tensors="pt",
        )
        result["labels"] = result["input_ids"]
        return result


    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

    model.eval()
    negative = {}
    for train_data in tokenized_train_dataset:
        with torch.no_grad():
            negative['input'] = train_data['input']
            negative['output'] = train_data['output']
            model_output = model.generate(torch.tensor(train_data["input_ids"]), temperature=0, do_sample=False, max_new_tokens=500,)
            negative['rejected'] = tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).split("|im_start|> assistant \n")[-1]

            # Check if the file exists
            if not os.path.exists(f'{cfg.data.processed}/negative.jsonl'):
                # Create the file if it doesn't exist
                with open(f'{cfg.data.processed}/negative.jsonl', 'w') as f:
                    pass
            # Store the negative examples in the json file in append mode
            with open(f'{cfg.data.processed}/negative.jsonl', 'a') as f:
                json.dump(negative, f)
                f.write('\n')

