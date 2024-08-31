import hydra
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import sys
import os

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    try:
        try:
            patients_with_biomarker = pd.read_csv(f"{cfg.data.processed_dir}/patient_with_biomarkers.csv")
        except FileNotFoundError as e:
            logger.error(f"File not found {e.filename}")
            sys.exit(1)

        # Count patients per cancer type
        try:
            patient_counts = patients_with_biomarker['CANCER_TYPE'].value_counts()
        except KeyError:
            raise KeyError("The column 'CANCER_TYPE' is missing from the data.")

        # Calculate percentages
        patient_percentage = (patient_counts / patient_counts.sum()) * 100

        # Select top cancer types
        top_percentage = patient_percentage.head(20)

        # Plotting with matplotlib
        plt.figure(figsize=(3.54,3.54), dpi=600)
        top_percentage.plot(kind='bar', color='C0')
        #plt.title('Top 30 Cancer Types by Number of Patients')
        plt.xlabel('Cancer Type', fontsize=8)
        plt.ylabel('Percentage of Patients\nwith a biomarker (%)', fontsize=8)
        plt.xticks(rotation=90, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()

        figures_dir = cfg.figures.dir
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        plt.savefig(f'{figures_dir}/aacr_patient_cancer_distribution_perc.png')
    except KeyError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
