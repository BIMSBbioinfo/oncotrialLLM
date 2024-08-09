import re
import pandas as pd
from Levenshtein import distance

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed


class BiomarkerHandler:
    mutation_category_regex = re.compile(r'((Non-|\*|[A-Z]+)\d+(_[A-Z]\d+(>[A-Z]+)?)?(del|delins|ins)?([A-Z]+|fs|\*)?(\s\(c\.\d+[A-Z]+\d+(>[A-Z]+)?\))?|((Exon|Intron) \d+( \w+)?\s)?mutation)', re.IGNORECASE)

    @staticmethod
    @lru_cache(maxsize=None)
    def find_pattern_match(variant_str, pattern_regex):
        match = pattern_regex.search(variant_str)
        return match.group(0) if match else variant_str

    @staticmethod
    def populate_categorical_pattern(civic_gene, pattern_regex):
        civic_gene['matched_variant'] = civic_gene['variant'].apply(lambda v: BiomarkerHandler.find_pattern_match(v, pattern_regex))
        return civic_gene['matched_variant'].unique().tolist()

    @staticmethod
    @lru_cache(maxsize=None)
    def alignment_distance(s1, s2, score_cutoff=None, weights=(1, 1, 1)):
        if len(weights) != 3:
            raise ValueError("Weights must be provided as a tuple of three integers.")
        return distance(s1, s2, score_cutoff=score_cutoff, weights=weights)

    @staticmethod
    def find_closest_match(target, candidates, max_distance=1, weights=(1, 1, 2)):
        target_upper = target.upper()
        distances = candidates.apply(lambda x: BiomarkerHandler.alignment_distance(target_upper, x.upper(), weights=weights))
        return candidates[distances <= max_distance].unique().tolist()

    @staticmethod
    def compare_to_synonym(target, synonyms_df, cols, max_distance, index_col, weights=(1, 1, 2)):
        target_upper = target.upper()
        closest_candidate = []
        for column in cols:
            df_candidates = synonyms_df[column].dropna().unique()
            closest_candidate.extend([index for index in synonyms_df.index if index in BiomarkerHandler.find_closest_match(target_upper, pd.Series(df_candidates), max_distance, weights)])
        return list(set(closest_candidate))

    @staticmethod
    def post_process_variant(gene, variant, variant_synonyms_df, civic_df, distance_thresh=1, weights=(1, 2, 2)):
        matched_variant = None
        civic_gene = civic_df[civic_df['gene'] == gene]
        if not civic_gene.empty:
            if "mutation" in variant.lower():
                matched_variant = BiomarkerHandler.populate_categorical_pattern(civic_gene, BiomarkerHandler.mutation_category_regex)
            else:
                matches = variant_synonyms_df[variant_synonyms_df.gene == gene]
                matches_variant = matches[
                    (matches['variant'].str.lower() == variant.lower()) |
                    (matches['variant_alias'].str.lower() == variant.lower())
                ]
                if not matches_variant.empty:
                    new_variant = matches_variant['variant'].iloc[0]
                    variant = new_variant
                vars_cols = ["variant", "variant_alias"]
                matched_variant = BiomarkerHandler.compare_to_synonym(variant, matches, vars_cols, distance_thresh, "variant", weights)
        return matched_variant

    @staticmethod
    def populate_biomarkers(biomarkers_list, variant_synonyms_df, civic_df, max_workers=4):
        larger_biomarkers_list = []
        cache = {}

        def process_biomarker(b):
            gene, variant = b.split(" ", 1)
            cache_key = (gene, variant)
            if cache_key in cache:
                new_biom = cache[cache_key]
            else:
                new_biom = BiomarkerHandler.post_process_variant(gene, variant, variant_synonyms_df, civic_df)
                cache[cache_key] = new_biom
            if new_biom:
                return [f"{gene} {v}" for v in new_biom]
            return [b]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_biomarker = {executor.submit(process_biomarker, b): b for b in biomarkers_list}
            for future in as_completed(future_to_biomarker):
                larger_biomarkers_list.extend(future.result())

        for b in larger_biomarkers_list.copy():
            gene, variant = b.split(" ", 1)
            variants = variant_synonyms_df[variant_synonyms_df.gene == gene]
            current_variants = variants[variants.variant == variant]
            aliases = current_variants.variant_alias.values
            larger_biomarkers_list.extend([f"{gene} {v}" for v in aliases if v not in larger_biomarkers_list])

        return larger_biomarkers_list
