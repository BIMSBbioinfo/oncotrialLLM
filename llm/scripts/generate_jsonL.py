"""
Description: This script generates train and test sets from a previously annotated dataset in JSON format. It splits the dataset into train and test subsets, appends the clinical trial text for each id, and saves the resulting sets to JSON files.
"""

import hydra
from sklearn.model_selection import train_test_split

import json

from utils.jsons import write_jsonl, to_jsonl, load_json, dump_json


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    # Load JSON file
    try:
        annotated_path = f"{cfg.data.interim_dir}/random_t_annotation_500_42.json"
        annotated = load_json(annotated_path)
    except Exception as e:
        print(f"Error loading annotated JSON file: {e}")
        return


    # convert dict to list of dict
    list_annotated = [{'trial_id': trial_id, "output": {"inclusion_biomarker": trial_data.get('inclusion_biomarker', []), "exclusion_biomarker": trial_data.get('exclusion_biomarker', [])}, "document": trial_data['document']} for trial_id, trial_data in annotated.items()]
    train_size = int(len(list_annotated) * cfg.split_params.train_percent/100)

    # Split data into train and test
    try:
        training_data, test_data = train_test_split(list_annotated,
                                                    train_size=train_size,
                                                    random_state=cfg.split_params.random_state)
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Save train and test sets to JSON files
    try:
        dump_json(data={"size": len(training_data), "ids": training_data},
                  file_path=f"{cfg.data.interim_dir}/train_set_test.json")

        dump_json(data={"size": len(test_data), "ids": test_data},
                  file_path=f"{cfg.data.interim_dir}/test_set_test.json")
    except Exception as e:
        print(f"Error saving train/test sets to JSON files: {e}")


    #  Split the JSONL dataset into train, validation, and test sets.
    
    # Extend training set with synthetic data if provided
    try:
        syn_data = load_json(f"{cfg.data.processed_dir}/gpt4_simulated_trials.json")
        aug_train_set = training_data.copy()
        aug_train_set.extend(syn_data)  # Extend the training_data list with syn_data
    except Exception as e:
        print("Could not load simulated data {e}")

    try:
        # Convert datasets to JSONL format
        train_messages = to_jsonl(training_data)
        test_messages = to_jsonl(test_data)
    except Exception as e:
        print("Could not convert data to JSONL: {e}")

    try:
        augmented_train_messages = to_jsonl(aug_train_set)
    except Exception as e:
        print("could not convert augmented data to jsonL: {e}")


    try:
        # Split the training set into train and validation sets
        train_list, validation_list = train_test_split(train_messages, test_size=0.2, random_state=42)
        # Write data to JSONL files
        write_jsonl(f'{cfg.data.processed_dir}/ft_train.jsonl', train_list)
        write_jsonl(f'{cfg.data.processed_dir}/ft_validation.jsonl', validation_list)
        write_jsonl(f'{cfg.data.processed_dir}/ft_test.jsonl', test_messages)
    except json.JSONDecodeError as e:
        print(f"Error writing to file: {e}")

    try:
        aug_train_list, aug_validation_list = train_test_split(augmented_train_messages, test_size=0.2, random_state=42)
        # Write data to JSONL files
        write_jsonl(f'{cfg.data.simulated_dir}/ft_train.jsonl', aug_train_list)
        write_jsonl(f'{cfg.data.simulated_dir}/ft_validation.jsonl', aug_validation_list)
    except json.JSONDecodeError as e:
        print(f"Error writing to file: {e}")




if __name__ == "__main__":
    main()
