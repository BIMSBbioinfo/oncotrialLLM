"""
Description:
    This script evaluates the performance of a Language Model (LLM) using a test set and multiple prompts. 
    It generates predictions based on the provided prompts and evaluates the accuracy of the predictions against 
    the ground truth labels. The evaluation includes metrics such as precision, recall, F1 score, accuracy, and latency.
"""

import hydra
import progressbar
from loguru import logger
from datasets import load_dataset
from langchain.prompts import load_prompt

import os
import sys
import time

from modules.gpt_handler import GPTHandler
from utils.evaluation import compute_evals, save_eval, get_metrics
from utils.jsons import (
    load_jsonl,
    load_json,
    dump_json,
    loads_json)


def load_prompt_file(file_path):
    """
    Check if the prompt file exists and load its content.

    Args:
        file_path (str): The path to the prompt file.

    Returns:
        str: The content of the prompt file if it exists, otherwise None.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    return load_prompt(file_path)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    model = cfg.GPT_EVAL.model
    prompt_1 = cfg.PROMPT_FILES.gpt_chain_one
    prompt_2 = cfg.PROMPT_FILES.gpt_chain_two

    log_dir = cfg.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"chain_of_prompts_{model}_log.txt")
    logger.add(log_filename, level="INFO", format="{time} - {name} - {level} - {message}")

    logger.info(f"Initializing evaluation for model: {model}")
    logger.info(f"Log file will be saved in: {log_filename}")

    # Load test set
    try:
        test_set = load_jsonl(cfg.GPT_EVAL.test_set)
    except Exception as e:
        logger.error(f"Loading Test data from HuggingFace: {e}")
        dataset = load_dataset(f'{cfg.HuggingFace}/manual_annotated_data', split=['train', 'validation', 'test'])
        test_set = dataset[2]

    try:
        # Load the first prompt file
        first_prompt_template = load_prompt_file(prompt_1)

        # Load the second prompt file
        second_prompt_template = load_prompt_file(prompt_2)
    except FileNotFoundError as e:
        logger.error(f"Template File {e.filename} does not exist: {e}")
        sys.exit(1)

    try:
        gpthandler = GPTHandler(cfg)
        # Set up LLM chain for the first prompt
        chain_1 = gpthandler.setup_gpt(
            model_name=model,
            prompt=first_prompt_template)

        # Set up LLM chain for the second prompt
        chain_2 = gpthandler.setup_gpt(
            model_name=model,
            prompt=second_prompt_template)

    except Exception as e:
        logger.error(f"Failed to set up LLM chain: {e}")
        sys.exit(1)

    logger.info(f"Chain 1: {first_prompt_template.template}")
    logger.info(f"Chain 2: {second_prompt_template.template}")

    start_time = time.time()

    tp_inc, tn_inc, fp_inc, fn_inc = [], [], [], []
    tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf = [], [], [], []

    tp_ex, tn_ex, fp_ex, fn_ex = [], [], [], []
    tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf = [], [], [], []
    predicted_list, actual_list, failed_prediction = [], [], []

    bar = progressbar.ProgressBar(maxval=len(test_set), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    counter = 0
    for i in test_set:
        counter += 1
        bar.update(counter)
        try:
            logger.info(f"@ trial {counter}")

            actual = i['output']
            input_trial = i['input']

            response_1 = chain_1.run({'trial': input_trial})
            response = chain_2.run({'input_list': response_1})

            try:
                response_parsed = loads_json(response)
            except Exception as e:
                logger.error(f"Trial {counter} Failed to parse JSON output: {e}")
                failed_prediction.append(response)
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

            logger.info(f"Predicted: {response_parsed}")
            logger.info(f"Actual: {actual}")

            # Metrics
            evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response_parsed, actual)
            save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
            save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
            save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
            save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)

            logger.info("\n")
        except Exception as e:
            logger.error(f"Trial {counter} Failed: {e}")

    end_time = time.time()
    latency = end_time - start_time
    logger.info(f"Latency: {latency} seconds")
    logger.info("\n\n\n")

    # Get Precision, recall, f1 score and accuracy
    inc = get_metrics(tp=sum(tp_inc), tn=sum(tn_inc), fp=sum(fp_inc), fn=sum(fn_inc))
    ex = get_metrics(tp=sum(tp_ex), tn=sum(tn_ex), fp=sum(fp_ex), fn=sum(fn_ex))
    inc_dnf = get_metrics(tp=sum(tp_inc_dnf), tn=sum(tn_inc_dnf), fp=sum(fp_inc_dnf), fn=sum(fn_inc_dnf))
    ex_dnf = get_metrics(tp=sum(tp_ex_dnf), tn=sum(tn_ex_dnf), fp=sum(fp_ex_dnf), fn=sum(fn_ex_dnf))
    results = {
        "Model": model,
        "correct_size": len(predicted_list),
        "failed_size": len(failed_prediction),
        "Precited": predicted_list,
        "Actual": actual_list,
        "Failed": failed_prediction,
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
        "Latency (seconds)": latency
    }
    bar.finish()

    output_file = f"{model}_{cfg.GPT_EVAL.OUTPUT_PROMPTS.prompt_chain}.json"

    try:
        # Read existing data from the file, if it exists
        existing_data = load_json(output_file)
    except FileNotFoundError:
        logger.error(f"Output file {output_file} does not exist")
        existing_data = {}

    # Append the new data to the existing results list
    if "results" in existing_data:
        existing_data["results"].append(results)
    else:
        existing_data["results"] = [results]

    dump_json(existing_data, output_file)


if __name__ == "__main__":
    main()