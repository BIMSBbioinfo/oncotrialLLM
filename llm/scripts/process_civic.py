"""
This script processes CIViC data, cleans variant names, standardizes them,
and retrieves gene synonyms using the NCBI Datasets API.
"""

import re
import hydra
import pandas as pd
from ncbi.datasets.openapi import ApiClient as DatasetsApiClient
from ncbi.datasets.openapi import ApiException as DatasetsApiException
from ncbi.datasets import GeneApi as DatasetsGeneApi

from typing import List


def clean_mutation_pattern(input_string):
    # Define the mutation pattern
    mutation_pattern = r'\(?C\..+\)?'
    # Find the first match for mutation pattern
    mutation_match = re.search(mutation_pattern, input_string)
    # Remove the mutation if found and not at the beginning
    if mutation_match and mutation_match.start() > 0:
        input_string = input_string.replace(mutation_match.group(), '')
    return input_string.strip()


def clean_variant(variant):
    """
    Clean and standardize variant names.

    Parameters:
        - variant (str): Variant name.

    Returns:
        - str: Cleaned and standardized variant name.
    """
    variant = re.sub(r'EX(\d+)', r'EXON \1', variant)
    variant = re.sub(r'\bDEL\b', 'DELETION', variant)
    # Remove the "AND" between variants
    variant = re.sub(r'\bAND \b', '', variant)
    pattern = r'([A-Za-z]\d+[A-Za-z])-([A-Za-z]\d+[A-Za-z])'
    cleaned_variant = re.sub(pattern, r'\1 \2', variant)
    # remove the pattern c.\d+\w>\w when it follows a variant
    cleaned_variant = clean_mutation_pattern(cleaned_variant)
    return cleaned_variant


def get_civic_genes(civic):
    genes_list = list(civic['gene'].unique())
    return genes_list


def get_genes_synonym(gene_symbols: List[str], output_file):
    taxon = "human"
    input_dict = {}
    output_list = []

    with DatasetsApiClient() as api_client:
        gene_api = DatasetsGeneApi(api_client)
        try:
            gene_reply = gene_api.gene_metadata_by_tax_and_symbol(gene_symbols, taxon)
            for gene in gene_reply.genes:
                if gene.warnings:
                    continue
                if gene.gene.synonyms:
                    input_dict[gene.gene.symbol] = {'synonyms': gene.gene.synonyms,
                                                    'query': gene.query[0]}
                else:
                    continue
            output_list = [{'gene': details['query'], 'symbol': gene, 'synonym': synonym} for gene, details in input_dict.items() for synonym in details['synonyms']]
        except DatasetsApiException as e:
            print(f"Exception when calling GeneApi: {e}\n")

    df = pd.DataFrame(output_list)
    df.to_csv(output_file, index=None)


def expand_variant_aliases(variants_df, output_file):
    """
    Separates variant_aliases and populates each alias with the gene and variant on a new line.

    Args:
        variants_df (pd.DataFrame): DataFrame with columns gene, variant, and variant_aliases.

    Returns:
        pd.DataFrame: DataFrame with expanded variant_aliases.
    """

    variants_df = variants_df[['gene', 'variant', 'variant_aliases']]

    variants_df = variants_df.dropna(subset=['variant_aliases']).reset_index(drop=True)

    variants_df = variants_df.applymap(str.upper)

    expanded_data = []
    for _, row in variants_df.iterrows():
        gene = row['gene']
        variant = row['variant'].replace("::", "-")
        aliases = row['variant_aliases'].split(',')
        aliases = [item.replace("::", "-") for item in aliases]
        for alias in aliases:
            expanded_data.append({'gene': gene, 'variant': variant, 'variant_alias': alias.strip()})
    expanded_df = pd.DataFrame(expanded_data)

    expanded_df.to_csv(output_file, index=None)


def process_civic_data(civic, output_file):

    # Converting gene names and variants to uppercase
    civic = civic.assign(gene=civic['gene'].str.upper(),
                         variant=civic['variant'].str.upper())

    # Standardizing variant name
    civic['variant'] = civic['variant'].apply(clean_variant)

    # Saving Processed Data
    civic.to_csv(output_file, index=False)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    civic = pd.read_table(cfg.civic.raw_file)
    expand_variant_aliases(civic, cfg.civic.variant_syn_file)

    civic = civic[['gene', 'variant']].drop_duplicates().reset_index(drop=True)
    process_civic_data(civic, cfg.civic.processed_file)
    get_genes_synonym(get_civic_genes(civic), cfg.civic.gene_syn_file)



if __name__ == "__main__":
    main()


    