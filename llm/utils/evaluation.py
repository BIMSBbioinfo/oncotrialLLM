"""
Clinical Trial Evaluation Script

This script includes functions for evaluating predictions in clinical trial
inclusion and exclusion criteria.
The `evaluate_predictions` function calculates precision, recall, F1 score,
and accuracy based on predicted and ground truth criteria.
It supports flexible input formats, allowing for both flat lists and lists of
lists representing conjunctions of criteria.
The `threshold_accuracy` function determines if the accuracy meets or exceeds
a specified threshold percentage.
Both functions assume accuracy is represented as a float between 0 and 1,
and the threshold is a percentage.
"""


def evaluate_predictions(predicted_output, ground_truth, entity):
    tp, tn, fp, fn = 0, 0, 0, 0

    predicted_biomarkers = [set(map(str.lower, inner_list)) for inner_list in predicted_output[entity]]
    actual_biomarkers = [set(map(str.lower, inner_list)) for inner_list in ground_truth[entity]]

    if predicted_biomarkers == [] and actual_biomarkers == []:
        tn = 1
    else:
        # Calculate True Positives
        for predicted_set in predicted_biomarkers:
            if predicted_set in actual_biomarkers:
                tp += 1

        # Calculate False Negatives
        for actual in actual_biomarkers:
            if actual not in predicted_biomarkers:
                fn += 1

        # Calculate False Positives
        for predicted_set in predicted_biomarkers:
            if predicted_set not in actual_biomarkers:
                fp += 1

    return tp, tn, fp, fn


def get_metrics(tp, tn, fp, fn):
    # Calculate precision, recall, F1 score, and accuracy
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    f2_score = 5 * (precision * recall) / ((4*precision) + recall) if  ((4*precision) + recall) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    return precision, recall, f1_score, accuracy, f2_score


def threshold_accuracy(accuracy, threshold=50):
    """
    Determine if the accuracy meets or exceeds a specified threshold percentage.

    Parameters:
    - accuracy (float): The accuracy score calculated as the ratio of correctly predicted criteria to the total number of predicted criteria.
    - threshold (float): The minimum percentage of accuracy required to return True. Default is 50.

    Returns:
    - threshold_met (bool): True if the accuracy percentage meets or exceeds the specified threshold, False otherwise.

    Note:
    - The function assumes that accuracy is provided as a float value between 0 and 1, where 1 represents 100% accuracy.
    - The threshold is specified as a percentage, and the comparison is done by multiplying the accuracy by 100 and checking if it meets or exceeds the threshold.
    """
    threshold_met = accuracy * 100 >= threshold
    return threshold_met


def save_eval(tp, tn, fp, fn, evals):
    true_p, true_n, false_p, false_n = evals
    tp.append(true_p)
    tn.append(true_n)
    fp.append(false_p)
    fn.append(false_n)
