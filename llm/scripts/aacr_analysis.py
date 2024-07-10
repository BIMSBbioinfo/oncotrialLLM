import re
import hydra
import pandas as pd

import json

from modules.biomarker_handler import BiomarkerHandler 


def load_data(clinical_sample_file, mutations_file, biomarkers_file, gene_file, variant_file, civic_file, cna_file, sv_file):
    try:
        # Load data files
        data_clinical_sample = pd.read_csv(clinical_sample_file, sep="\t", comment='#')
        data_mutations = pd.read_csv(mutations_file, sep="\t")
        with open(biomarkers_file, 'r') as f:
            biomarkers_list = json.load(f)['biomarkers']
        gene_synonyms = pd.read_csv(gene_file)
        variant_synonyms = pd.read_csv(variant_file)
        civic_df = pd.read_csv(civic_file)
        data_cna = pd.read_csv(cna_file, sep="\t") # > 1 AMP, <-1 DEL
        # Handle SV -- Work on this now...
        data_sv =  pd.read_csv(sv_file, sep="\t")
        data_sv = data_sv[['Sample_Id', 'Site1_Hugo_Symbol',
            'Site2_Hugo_Symbol', 'Event_Info', 'Site1_Description', 'Site2_Description','Class']]
        return data_clinical_sample, data_mutations, biomarkers_list, gene_synonyms, variant_synonyms, civic_df, data_sv, data_cna
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data_mutations):
    data_mutations_selected = data_mutations[['Hugo_Symbol', 'Variant_Classification', 'Variant_Type',
                                              'dbSNP_RS', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'Mutation_Status',
                                              'HGVSc', 'HGVSp', 'HGVSp_Short', 'Exon_Number']]
    data_mutations_selected['HGVSp_Short'] = data_mutations_selected['HGVSp_Short'].str.replace('p.', '')
    return data_mutations_selected


def map_gene_symbols(data_mutations_selected, gene_synonyms):
    synonym_mapping = pd.melt(
        gene_synonyms,
        id_vars=['gene'],
        value_vars=['symbol', 'synonym'],
        var_name='type',
        value_name='synonym'
    ).drop(columns='type').dropna().drop_duplicates().reset_index(drop=True)
    synonym_mapping = synonym_mapping[synonym_mapping['synonym'] != '']
    synonym_mapping = synonym_mapping.set_index('synonym')['gene'].to_dict()
    # Replace Hugo_Symbol with corresponding gene if it exists in the mapping
    data_mutations_selected['Hugo_Symbol'] = data_mutations_selected['Hugo_Symbol'].map(synonym_mapping).fillna(data_mutations_selected['Hugo_Symbol'])
    # drop rows where Hugo_Symbol wasn't found in the original gene list
    data_mutations_selected = data_mutations_selected[data_mutations_selected['Hugo_Symbol'].isin(gene_synonyms['gene'].unique())].drop_duplicates().reset_index(drop=True)
    return data_mutations_selected


def create_biomarkers_df(biomarkers_list, variant_synonyms_df, civic_df):
    larger_biomarkers_list = BiomarkerHandler.populate_biomarkers(biomarkers_list, variant_synonyms_df, civic_df)
    biomarkers_data = [b.split(" ", 1) for b in set(larger_biomarkers_list)]
    biomarkers_df = pd.DataFrame(biomarkers_data, columns=['gene', 'variant'])
    return biomarkers_df


def filter_mutations(data_mutations_selected, biomarkers_df):
    filtered_data_mutations = data_mutations_selected.merge(
        biomarkers_df,
        left_on=['Hugo_Symbol', 'HGVSp_Short'],
        right_on=['gene', 'variant'],
        how='inner'
    )
    filtered_data_mutations = filtered_data_mutations.drop(columns=['gene', 'variant'])
    return filtered_data_mutations


def process_cna(data_cna, biomarkers_df, gene_synonyms):
    cna_biomarkers = biomarkers_df[biomarkers_df.variant.isin(["AMPLIFICATION", "DELETION"])]
    cna_gene_synonyms = gene_synonyms[gene_synonyms.gene.isin(cna_biomarkers.gene)]
    data_cna = map_gene_symbols(data_cna, cna_gene_synonyms)
    filtered_data_cna = data_cna.merge(
            cna_biomarkers,
            left_on=['Hugo_Symbol'],
            right_on=['gene'],
            how='inner'
        )
    filtered_data_cna = filtered_data_cna.drop(columns=['gene', 'variant']).drop_duplicates().reset_index(drop=True) # the drop dups takes time, find out how to avoid it before
    def categorize_samples(data_cna):
        # Extract gene names and sample names
        gene_names = data_cna['Hugo_Symbol']
        sample_names = data_cna.columns[1:]  # Excluding the 'Hugo_Symbol' column
        # Create a mask for amplification and deletion
        amplification_mask = data_cna[sample_names] > 1
        deletion_mask = data_cna[sample_names] < -1
        # Stack the masks and filter the True values
        amplifications = amplification_mask.stack().reset_index()
        deletions = deletion_mask.stack().reset_index()
        # Rename columns
        amplifications.columns = ['Hugo_Symbol_Index', 'Sample', 'Amplified']
        deletions.columns = ['Hugo_Symbol_Index', 'Sample', 'Deleted']
        # Filter only rows where the condition is True
        amplifications = amplifications[amplifications['Amplified']].drop(columns='Amplified')
        deletions = deletions[deletions['Deleted']].drop(columns='Deleted')
        # Map the Hugo_Symbol from the original DataFrame
        amplifications['Hugo_Symbol'] = amplifications['Hugo_Symbol_Index'].map(gene_names)
        deletions['Hugo_Symbol'] = deletions['Hugo_Symbol_Index'].map(gene_names)
        # Select relevant columns and add status
        amplifications = amplifications[['Hugo_Symbol', 'Sample']].assign(HGVSp_Short='AMPLIFICATION')
        deletions = deletions[['Hugo_Symbol', 'Sample']].assign(HGVSp_Short='DELETION')
        # Concatenate the results
        results = pd.concat([amplifications, deletions]).reset_index(drop=True)
        return results
    results = categorize_samples(filtered_data_cna)
    results_match = filter_mutations(results, cna_biomarkers)
    results.rename(columns={'Sample': 'SAMPLE_ID'}, inplace=True)
    return results_match


def process_sv(data_sv, biomarkers_df):
    def is_categorical_mutation(description):
        categories = ['DELETION', 'DUPLICATION', 'INSERTION', 'TRANSLOCATION', 'FUSION', "INVERSION"] # categories found in sv data
        for category in categories:
            if category == description:
                return category
        return None
    matches_sv = []
    # Iterate through each biomarker in the biomarkers_df
    for _, row in biomarkers_df.iterrows():
        gene = row['gene'].upper()
        variant = row['variant'].upper()
        category = is_categorical_mutation(variant)
        # Find matching rows in data_sv
        matching_rows = data_sv.loc[(data_sv['Site1_Hugo_Symbol'] == gene) | (data_sv['Site1_Hugo_Symbol'] == gene)]
        if category:
            # If it is a categorical mutation, check against the Class column
            matched_sv = matching_rows.loc[(matching_rows['Class'].str.upper() == category) & (matching_rows['Site2_Hugo_Symbol'].isna())]
        else:
            # Split variant by either '-' or '::'
            variant_parts = re.split(r'(-|::)', variant)
            if len(variant_parts) == 3:
                # If the variant consists of exactly two parts (fusion), match both parts in Site1 and Site2
                gene1, sep, gene2 = variant_parts
                matched_sv = matching_rows[((matching_rows['Site1_Hugo_Symbol'].str.upper() == gene1) & 
                                            (matching_rows['Site2_Hugo_Symbol'].str.upper() == gene2))]
            else:
                continue
     # Append matches to the results list
        for _, match in matched_sv.iterrows():
            matches_sv.append({
                'SAMPLE_ID': match['Sample_Id'],
                "gene": gene,
                "var": variant,
                "Site1_Hugo_Symbol": match["Site1_Hugo_Symbol"],
                "Site2_Hugo_Symbol": match["Site2_Hugo_Symbol"],
                'Match_Type': match['Class'],
                'SV_Detail': match['Event_Info']
            })
    return pd.DataFrame(matches_sv)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    # Load data
    data_clinical_sample, data_mutations, biomarkers_list, gene_synonyms, variant_synonyms, civic_df, data_sv, data_cna = load_data(cfg.aacr.clinical_sample, cfg.aacr.data_mutations, f"{cfg.data.interim_dir}/biomarkers_list.json", cfg.civic.gene_syn_file, cfg.civic.variant_syn_file, cfg.civic.processed_file, cfg.aacr.data_cna, cfg.aacr.data_sv)

    # Preprocess data
    data_mutations_selected = preprocess_data(data_mutations)

    # Map gene symbols
    data_mutations_selected = map_gene_symbols(data_mutations_selected, gene_synonyms)

    # Create biomarkers DataFrame
    biomarkers_df = create_biomarkers_df(biomarkers_list, variant_synonyms, civic_df)

    # Filter mutations based on biomarkers
    samples_with_mutation = filter_mutations(data_mutations_selected, biomarkers_df) # 43808 --> compare to see what diff between the 2 lines of codes
    samples_with_mutation.rename(columns={'Tumor_Sample_Barcode': 'SAMPLE_ID'}, inplace=True)

    samples_with_cna = process_cna(data_cna, biomarkers_df, gene_synonyms)

    samples_with_sv = process_sv(data_sv, biomarkers_df)

    sample_with_biomarker = pd.concat([samples_with_cna, samples_with_mutation, samples_with_sv])

    # get patient ID from sample ID
    patients_with_biomarker = pd.merge(data_clinical_sample, sample_with_biomarker, how='inner', on='SAMPLE_ID')

    num_total_patients = len(set(data_clinical_sample['PATIENT_ID']))
    num_patient_with_biomarker = len(set(patients_with_biomarker['PATIENT_ID']))

    percentage_matching = (num_patient_with_biomarker / num_total_patients) * 100

    print(f"The percentage of patients matching the biomarkers is: {percentage_matching:.2f}%")

    patients_with_biomarker.to_csv(f"{cfg.data.processed_dir}/patient_with_biomarkers_NEW.csv", index=False)


if __name__ == "__main__":
    main()



