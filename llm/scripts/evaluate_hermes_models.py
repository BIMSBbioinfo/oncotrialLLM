#!/usr/bin/env python
# coding: utf-8

"""
Description:
    This script is used to test the Hermes model's performance on a clinical trials dataset. It loads the dataset, tokenizes the input prompts, generates responses using the model, and evaluates the predicted biomarkers against the ground truth. The evaluation metrics include precision, recall, F1 score, and accuracy for both inclusion and exclusion biomarkers.

Output:
    The script generates evaluation results for the Hermes model's performance on the clinical trials dataset. It stores the results in a JSON file.

Example:
    python test_hermes.py
"""

# # Inference

import re
import torch
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.jsons import (
    load_json,
    flatten_lists_in_dict,
    dump_json,
    loads_json)
from utils.evaluation import evaluate_predictions, save_eval, get_metrics


dataset = load_dataset('nalkhou/clinical-trials', split=['train', 'validation', 'test'])

train_dataset = dataset[0]
validation_dataset = dataset[1]
test_dataset = dataset[2]

print("Train dataset:")
print(train_dataset)
print("\nValidation dataset:")
print(validation_dataset)
print("\nTest dataset:")
print(test_dataset)


base_model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

model = PeftModel.from_pretrained(base_model, "mistral-7B-ctrials-finetune/checkpoint-180")


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


test_dataset_dataset = test_dataset.map(generate_and_tokenize_prompt)


def evaluate_predictions(predicted_output, ground_truth, entity):
    tp, tn, fp, fn = 0, 0, 0, 0

    predicted_biomarkers = [set(map(str.lower, inner_list)) for inner_list in predicted_output[entity]]
    actual_biomarkers = [set(map(str.lower, inner_list)) for inner_list in ground_truth[entity]]

    if predicted_biomarkers == [] and actual_biomarkers == []:
        tn = 1
    else:
        # Calculate True Positives
        for predicted_set in predicted_biomarkers:
            if predicted_set in actual_biomarkers:
                tp += 1

        # Calculate False Negatives
        for actual in actual_biomarkers:
            if actual not in predicted_biomarkers:
                fn += 1

        # Calculate False Positives
        for predicted_set in predicted_biomarkers:
            if predicted_set not in actual_biomarkers:
                fp += 1

    return tp, tn, fp, fn


def get_metrics(tp, tn, fp, fn):
    # Calculate precision, recall, F1 score, and accuracy
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    f2_score = 5 * (precision * recall) / ((4*precision) + recall) if  ((4*precision) + recall) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    return precision, recall, f1_score, accuracy, f2_score


def threshold_accuracy(accuracy, threshold=50):
    """
    Determine if the accuracy meets or exceeds a specified threshold percentage.

    Parameters:
    - accuracy (float): The accuracy score calculated as the ratio of correctly predicted criteria to the total number of predicted criteria.
    - threshold (float): The minimum percentage of accuracy required to return True. Default is 50.

    Returns:
    - threshold_met (bool): True if the accuracy percentage meets or exceeds the specified threshold, False otherwise.

    Note:
    - The function assumes that accuracy is provided as a float value between 0 and 1, where 1 represents 100% accuracy.
    - The threshold is specified as a percentage, and the comparison is done by multiplying the accuracy by 100 and checking if it meets or exceeds the threshold.
    """
    threshold_met = accuracy * 100 >= threshold
    return threshold_met


def compute_evals(response, actual):
    # Evaluate results with DNF
    evals_dnf_inclusion = evaluate_predictions(response, actual, 'inclusion_biomarker')
    evals_dnf_exclusion = evaluate_predictions(response, actual, 'exclusion_biomarker')

    # Evaluate results without DNF
    flat_response = flatten_lists_in_dict(response)
    flat_actual = flatten_lists_in_dict(actual)
    evals_extract_incl = evaluate_predictions(flat_response, flat_actual, "inclusion_biomarker")
    evals_extract_exl = evaluate_predictions(flat_response, flat_actual, "exclusion_biomarker")

    return evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl


tp_inc, tn_inc, fp_inc, fn_inc = [], [], [], []
tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf = [], [], [], []

tp_ex, tn_ex, fp_ex, fn_ex = [], [], [], []
tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf = [], [], [], []

predicted_list = []
actual_list = []
failed_list = []


for i in test_dataset_dataset:
    try:
        actual = i["output"]
        model_output = model.generate(torch.tensor(i["input_ids"]), temperature=0,do_sample=False,max_new_tokens= 20000,)
        response = tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).split("|im_start|> assistant \n")[-1]
        print("response")
        try:
            # Finds the 'output' field
            pattern = re.search(r"'output': ({.*?})", response)
            if pattern:
                # Extract the 'input' and 'output' content
                output_content = pattern.group(1)
                # Parse the 'output' content to a Python dictionary
                response_parsed = eval(output_content)
            else:
                response_parsed = loads_json(re.sub(r'\}\}$(?!\})', '}', response.replace("'","\"")))

        except Exception as e:
            print(f"Failed to parse the json response: {response}: {e}")
            j = {"input": i["input"], "actual": actual, "predicted": response}
            failed_list.append(j)

            if actual == {'inclusion_biomarker': [], 'exclusion_biomarker': []}:
                evals_dnf_inclusion = evals_dnf_exclusion = evals_extract_incl = evals_extract_exl = (0,0,1,0)
            else:
                response = {'inclusion_biomarker': [], 'exclusion_biomarker': []}
                evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response, actual)
            save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
            save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
            save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
            save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)
            continue

        predicted_list.append(response_parsed)
        actual_list.append(actual)
    except Exception as e:
        print(f"Trial Failed: {e}")
        continue

    # Metrics
    evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response_parsed, actual)
    save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
    save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
    save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
    save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)


# Get Precision, recall, f1 score and accuracy
inc = get_metrics(tp=sum(tp_inc), tn=sum(tn_inc), fp=sum(fp_inc), fn=sum(fn_inc))
ex = get_metrics(tp=sum(tp_ex), tn=sum(tn_ex), fp=sum(fp_ex), fn=sum(fn_ex))
inc_dnf = get_metrics(tp=sum(tp_inc_dnf), tn=sum(tn_inc_dnf), fp=sum(fp_inc_dnf), fn=sum(fn_inc_dnf))
ex_dnf = get_metrics(tp=sum(tp_ex_dnf), tn=sum(tn_ex_dnf), fp=sum(fp_ex_dnf), fn=sum(fn_ex_dnf))


results = {
    "prompt": """<|im_start|>system\nYou are a helpful assistant that extracts only genomic biomarkers from the supplied clinical trial data and responds in JSON format. Here's the json schema you must adhere to:<schema>{{\"inclusion_biomarker\": [[]], \"exclusion_biomarker\": [[]]}}</schema>\nIn this context, limit the extraction of genomic biomarkers to the following categories: gene alteration (mutation, fusion, rearrangement, copy number alteration, deletion, insertion, translocation), pathway alterations, gene expression, protein expression, pathway expression, HLA, TMB (tumor molecular burden, TMB-H or TMB-L), MSI (microsatellite instability, MSI-H, MSI-L, MSS, microsatellite stable) status, gene pathway alteration like dMMR (deficient Mismatch Repair Pathway) or pMMR (proficient Mismatch Repair), and protein status (HER2, ER, PgR, PD-L1).\n\nDo not extract non-genomic biomarkers, which refer to any indicators not directly related to genetic or genomic information. Ignore information such as age, medical conditions, potential pregnancy, disease stage, allergies, treatment history, drugs, therapies, treatment, histology, and tumor cancer types, diseases, HIV, infections, and more. Also, ignore information about levels, scores, doses, expression ratios, and illnesses. Do not consider biomarkers related to model experimental animals, historical data, or previous studies.\n\nPreserve logical connections (AND, OR) between genomic biomarkers. Group 'AND'-linked genomic biomarkers in the same list, and place 'OR'-linked genomic biomarkers in separate lists. Treat main bullets in \"Inclusion Criteria\" as AND logic, and \"Exclusion Criteria\" as OR logic, unless specified otherwise. Handle ambiguous logic in the sentence as OR.\n\nEnsure each genomic biomarker is a string with the gene name preceding the variant. Remove the words \"gene\", \"allele\", \"status\", and \"mutation\" (when a specific variant is given). Make the variant singular and noun-based. Replace \"mutant\" with \"mutation\". Include a space between the gene name, its variant if they are connected. Include a space between the hormone name and its status if they are connected. Replace \"positive expression\" with \"expression\" and symbols \"-\" and \"+\" with \"negative\" and \"positive\" respectively, except in MSI status or known fusions separated by \"-\". Add \"germline\" or \"somatic\" terms in parentheses at the end of the corresponding biomarker. Ignore biomarkers mentioned as \"exceptions\" or after \"other than\". Handle synonyms in parentheses by extracting the genomic biomarker but ignoring the synonym. Extract each genomic biomarker once. Expand the genomic biomarkers when needed.\n\nTo summarize, extract only genomic biomarkers from the supplied clinical trial data, focusing on the categories mentioned above. Ignore any non-genomic biomarkers and unrelated information such as age, medical conditions, treatment history, cancer, drugs, therapies, histology, levels and scores. If no genomic biomarkers are found, return empty lists in JSON. Do not make assumptions or add biomarkers. Do not add any biomarkers that are not explicitly mentioned in the input, and do not make assumptions about potential genomic biomarkers. Ensure output list contains only lists of strings when there exist genomic biomarkers in the input, following this example: {{\"inclusion_biomarker\": [[\"GeneA variantA\"], [\"GeneX variantY]], \"exclusion_biomarker\": []}}. Do not \\escape. Do not repeat a genomic biomarker.<|im_end|>\n<|im_start|>user\nExtract the genomic biomarker from the clinical trial below. Just generate the JSON object without explanation.sample inserted here\n<|im_end|>\n<|im_start|>assistant""",
    "predicted_size": len(predicted_list),
    "Model": base_model_id,
    "Precited": predicted_list,
    "Actual": actual_list,
    "Failed": failed_list,
    "tp_inclusion": tp_inc,
    "fp_inclusion": fp_inc,
    "tn_inclusion": tn_inc,
    "fn_inclusion": fn_inc,
    "tp_exclusion": tp_ex,
    "tn_exclusion": tn_ex,
    "fp_exclusion": fp_ex,
    "fn_exclusion": fn_ex,
    "Inclusion Precision": [inc[0]],
    "Inclusion Recall": [inc[1]],
    "Inclusion F1": [inc[2]],
    "Inclusion Acc": [inc[3]],
    "Inclusion F2": [inc[4]],
    "Exclusion Precision": [ex[0]],
    "Exclusion Recall": [ex[1]],
    "Exclusion F1": [ex[2]],
    "Exclusion Acc": [ex[3]],
    "Exclusion F2": [ex[4]],
    "Inclusion DNF Precision": [inc_dnf[0]],
    "Inclusion DNF Recall": [inc_dnf[1]],
    "Inclusion DNF F1": [inc_dnf[2]],
    "Inclusion DNF Acc": [inc_dnf[3]],
    "Inclusion DNF F2": [inc_dnf[4]],
    "Exclusion DNF Precision": [ex_dnf[0]],
    "Exclusion DNF Recall": [ex_dnf[1]],
    "Exclusion DNF F1": [ex_dnf[2]],
    "Exclusion DNF Acc": [ex_dnf[3]],
    "Exclusion DNF F2": [ex_dnf[4]],
}


output_file = "Fine-Tuned-180-Mistral_7b_model_zero-shot.json"
try:
    # Read existing data from the file, if it exists
    existing_data = load_json(output_file)
except FileNotFoundError:
    existing_data = {}

# Append the new data to the existing results list
if "results" in existing_data:
    existing_data["results"].append(results)
else:
    existing_data["results"] = [results]

dump_json(existing_data, output_file)
