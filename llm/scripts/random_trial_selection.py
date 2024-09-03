"""
This script generates random biomarkers from the Civic dataset,
queries them in a ChromaDB collection,
and saves the trial Id and Document returned for training and testing purposes.
"""
import hydra
import pandas as pd
from loguru import logger

import random
import sys

from utils.jsons import dump_json
from modules.chromadb_handler import ChromaDBHandler


def get_civic_biomarkers(civic):
    """
    Extract unique biomarkers from the given CIViC (Clinical Interpretations
    of Variants in Cancer) dataset.

    Parameters:
        - civic (pd.DataFrame): The CIViC dataset containing gene
        and variant information.

    Returns:
        - numpy.ndarray: An array of unique biomarkers formed by concatenating
        gene and variant columns.
    """
    try:
        civic['biomarkers'] = civic['gene'] + " " + civic['variant']
        return civic['biomarkers'].unique()
    except KeyError as e:
        logger.error(f"Keyerror in civic data: {e}")


def get_random_nums(seed, input_size, output_size):
    """
    Generate a list of random numbers sampled without replacement
    from a given range.

    Parameters:
    - seed (int): The seed value for the random number generator.
    - input_size (int): The size of the range from which to sample random numbers.
    - output_size (int): The number of random numbers to generate.

    Returns:
    - list[int]: A list of unique random numbers sampled without replacement.
    """
    random.seed(seed)
    return random.sample(range(0, input_size), output_size)


def generate_random_data(civic_path, trials, size=500, seed=42):
    """
    Generate random data by querying a ChromaDB collection with a randomly
    selected set of biomarkers.

    Parameters:
        - civic_path (str): The file path to the Civic dataset containing gene and variant information.
        - persist_dir (str): The path to the directory where the ChromaDB collection is persisted.
        - size (int, optional): The number of random biomarkers to select. Default is 250.

    Returns:
        - dict: A dictionary containing query results with randomly selected biomarkers.
        The dictionary has keys 'ids' and 'documents', where 'ids' is a list of document IDs,
        and 'documents' is a list of corresponding document content.
    """
    try:
    # Get civic biomarkers list
        civic = pd.read_csv(civic_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to read file: {e.filename}")
        sys.exit(1)
    biomarkers = get_civic_biomarkers(civic)
    # Generate the random biomarkers list
    random_numbers = get_random_nums(seed, len(biomarkers), size)
    try:
        selected_biomarkers = biomarkers[random_numbers]
        results = trials.query(query_texts=selected_biomarkers,
                            n_results=1,
                            include=['documents'])  # return example {'ids': [['NCT04489433']], 'embeddings': None, 'documents': None, 'metadatas': None, 'distances': None}
        return results, selected_biomarkers
    except IndexError as e:
        logger.error(f"Index Error: {e}")
    except Exception as e:
        logger.error(f"Failed to randomly select trials: {e}")

def create_trials_list(ids, trials):
    trials_list = []
    queried_data = trials.get(ids=ids)
    for i in range(len(ids)):
        trial_dict = {
            'id': queried_data['ids'][i],
            'document': queried_data['documents'][i]
        }
        trials_list.append(trial_dict)

    return trials_list
    

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    try:
        # load collection
        trials = ChromaDBHandler(cfg.chromadb.persist_dir, cfg.chromadb.collection_name).collection
        
        results, biomarkers = generate_random_data(cfg.civic.processed_file, trials)
        unique_ids = list(set([id_val[0] for id_val in results['ids']]))   # example ['NCT05252403', 'NCT05435248', 'NCT04374877']

        results = create_trials_list(unique_ids, trials)

        dump_json(data={
            "size": len(unique_ids), 
            "trials": results},
                file_path=f"{cfg.data.raw_dir}/random_trials.json")
        
        dump_json(data={"size": len(biomarkers), "biomarkers": list(biomarkers)},
                file_path=f"{cfg.data.raw_dir}/biomarkers_list.json")
    except Exception as e:
        logger.error(f"Failed to select raandom trials: {e}")


if __name__ == "__main__":
    main()