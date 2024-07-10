import json
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def dump_json(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except TypeError as e:
        print(f"Unable to serialize the object: {e}")


def loads_json(json_str):
    return json.loads(json_str)


def flatten_lists_in_dict(input_dict):
    """
    Flatten lists in a dictionary.

    Parameters:
    - input_dict (dict): Input dictionary containing lists.

    Returns:
    - dict: Output dictionary with flattened lists.

    Example:
    - Input: {"inclusion": [["A", "B"], ["C"]], "exclusion": [['k']]}
    - Output: {"inclusion": ["A", "B", "C"], "exclusion": ['k']}
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            flattened_list = [item for sublist in value for item in (sublist if isinstance(sublist, list) else [sublist])]
            output_dict[key] = flattened_list
        else:
            output_dict[key] = value
    return output_dict

def write_jsonl(output_file, data_list):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the data to the JSONL file
    with open(output_file, 'w') as outfile:
        for entry in data_list:
            json.dump(entry, outfile)
            outfile.write('\n')


def to_jsonl(dataset):
    messages = []
    for trial in dataset:
        try:
            trial_id = trial.get('trial_id', None)
            if trial_id:
                if trial_id == "NCT04017130":  # skip this, outlier
                    continue
            document_key = 'document' if trial.get('document') else 'input'
            t_doc = trial[document_key]
            t_output = trial['output']

            current_message = {"input": t_doc, "output": t_output}

            messages.append(current_message)
        except (KeyError, IndexError) as e:
            print(f"Error processing trial: {trial['trial_id']} - {e}")
    return messages
