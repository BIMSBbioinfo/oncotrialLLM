import hydra
from omegaconf import DictConfig
import progressbar
from datasets import load_dataset
from loguru import logger

import os
import sys
import time
from langchain.prompts import load_prompt

from modules.gpt_handler import GPTHandler
from utils.jsons import (
    load_jsonl,
    load_json,
    dump_json,
    loads_json)
from utils.evaluation import compute_evals, save_eval, get_metrics


def find_trial(k, trials):
    trial = next((t for t in trials if t['trial_id'] == k), None)
    if trial:
        return trial['document'], trial['output']
    else:
        return None, None


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    n_shot = cfg.GPT_EVAL.n_shot
    model = cfg.GPT_EVAL.model
    # Set up loguru
    log_dir = cfg.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{n_shot}_shot_{model}_log.txt")
    logger.add(log_filename, level="INFO", format="{time} - {name} - {level} - {message}")

    # Load test set
    try:
        test_set = load_jsonl(cfg.GPT_EVAL.test_set)
    except Exception as e:
        logger.error(f"Loading Test data from HuggingFace: {e}")
        dataset = load_dataset(f'{cfg.HuggingFace}/manual_annotated_data', split=['train', 'validation', 'test'])
        test_set = dataset[2]

    # Load train set for few-shot
    if n_shot > 0:
        try:
            train_set = load_jsonl(cfg.GPT_EVAL.train_set)
        except Exception as e:
            logger.error(f"Loading Train data from HuggingFace: {e}")
            dataset = load_dataset(f'{cfg.HuggingFace}/manual_annotated_data', split=['train', 'validation', 'test'])
            train_set = dataset[0]

    # selecting prompt and output filename based on n_shot
    if n_shot == 0:
        template_file = cfg.PROMPT_FILES.gpt_zero_shot
        filename = f"{model}_{cfg.GPT_EVAL.OUTPUT_PROMPTS.zero_shot}.json"
    elif n_shot == 1:
        template_file = cfg.PROMPT_FILES.gpt_one_shot
        filename = f"{model}_{cfg.GPT_EVAL.OUTPUT_PROMPTS.one_shot}.json"
    else:
        template_file = cfg.PROMPT_FILES.gpt_two_shot
        filename = f"{model}_{cfg.GPT_EVAL.OUTPUT_PROMPTS.two_shot}.json"

    try:
        # check if prompts file exists
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"The file '{template_file}' does not exist.")
        prompt_template = load_prompt(template_file)
    except FileNotFoundError as e:
        logger.error(f"Template File {template_file} does not exist: {e}")
        sys.exit(1)

    # set up LLM chain
    try:
        gpthandler = GPTHandler(cfg)
        llm_chain = gpthandler.setup_gpt(model_name=model, prompt=prompt_template)
    except Exception as e:
        logger.error(f"Failed to set up GPTHandler {e}")
        sys.exit(1)

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

            if n_shot == 0:
                response = llm_chain({'trial': input_trial})
            else:
                example_id = "NCT03383575"
                example_doc, example_output = find_trial(example_id, train_set['ids'])
                example = f"""{example_doc}\nexample JSON:{example_output}"""
                if n_shot == 2:
                    example_id = "NCT05484622"
                    example_doc, example_output = find_trial(example_id, train_set['ids'])
                    example_2 = f"""{example_doc}\nJSON:{example_output}"""

                    response = llm_chain({'trial': input_trial, 'example': example, 'example2': example_2})
                else:
                    response = llm_chain({'trial': input_trial, 'example': example})
            logger.info(f"Actual: {actual} \n Response: {response}")
            try:
                response['text']
            except Exception as e:
                logger.error(f"Trial {counter} Failed to generate text output: {e}")
            try:
                response_parsed = loads_json(response['text'])
            except TypeError as e:
                logger.error(f"Trial {counter} Failed to parse text output: {e}")
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

            # Metrics
            evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response_parsed, actual)
            save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
            save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
            save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
            save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)

        except Exception as e:
            logger.error(f"Trial {counter} Failed: {e}")

    end_time = time.time()
    latency = end_time - start_time

    # Get Precision, recall, f1 score and accuracy
    inc = get_metrics(tp=sum(tp_inc), tn=sum(tn_inc), fp=sum(fp_inc), fn=sum(fn_inc))
    ex = get_metrics(tp=sum(tp_ex), tn=sum(tn_ex), fp=sum(fp_ex), fn=sum(fn_ex))
    inc_dnf = get_metrics(tp=sum(tp_inc_dnf), tn=sum(tn_inc_dnf), fp=sum(fp_inc_dnf), fn=sum(fn_inc_dnf))
    ex_dnf = get_metrics(tp=sum(tp_ex_dnf), tn=sum(tn_ex_dnf), fp=sum(fp_ex_dnf), fn=sum(fn_ex_dnf))
    results = {
        "prompt": prompt_template.template,
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
        "Latency (seconds)": latency,
    }
    bar.finish()

    # setting up output file
    results_dir = cfg.data.results_dir

    output_file = os.path.join(results_dir, filename)
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