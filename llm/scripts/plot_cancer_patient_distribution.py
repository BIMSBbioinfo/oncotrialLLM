import hydra
import pandas as pd
import matplotlib.pyplot as plt
import os

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    patients_with_biomarker = pd.read_csv(f"{cfg.data.processed_dir}/patient_with_biomarkers.csv")

    # Count patients per cancer type
    patient_counts = patients_with_biomarker['CANCER_TYPE'].value_counts()

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

if __name__ == "__main__":
    main()
