import os
import sys
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment, NeuralNetworkFineTunedTrainigExperiment


if __name__ == "__main__":

    single_subjec_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(ready).json')
    single_subjec_experiments: list[NeuralNetworkSingleSubjectTrainingExperiment] = [
        cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in single_subjec_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
    
    fine_tuned_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(biased).json')
    fine_tuned_experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in fine_tuned_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]

    statistics_by_subject = []
    for subject_id in range(0, 5):
        subject_self_training_experiments = [experiment for experiment in single_subjec_experiments if experiment.subject_id == subject_id]
        subject_self_training_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_self_training_experiments]
        
        subject_fine_tuning_experiments = [experiment for experiment in fine_tuned_experiments if experiment.subject_id == subject_id]
        subject_fine_tuning_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_fine_tuning_experiments]
        
        statistics_by_subject.append({
            "single_subject": {
                "mean" : np.mean(subject_self_training_f1_scores),
                "std": np.std(subject_self_training_f1_scores)
            },
            "fine_tuned": {
                "mean" : np.mean(subject_fine_tuning_f1_scores),
                "std": np.std(subject_fine_tuning_f1_scores)
            }
        })

    self_training_f1_score_means = [statistics["single_subject"]["mean"] for statistics in statistics_by_subject]
    self_training_f1_score_stds = [statistics["single_subject"]["std"] for statistics in statistics_by_subject]

    fine_tuning_f1_score_means = [statistics["fine_tuned"]["mean"] for statistics in statistics_by_subject]
    fine_tuning_f1_score_stds = [statistics["fine_tuned"]["std"] for statistics in statistics_by_subject]

    threshold_p_value = 0.05


    print()
    print("Null hypothesis: Fine-tuned model gives the same accuracy as single-subject model")
    print("Alternative hypothesis: Fine-tuned model gives higher accuracy than single-subject model")
    
    print("Paried T-test:")
    t_mean_stat, t_mean_p_value = ttest_rel(fine_tuning_f1_score_means, self_training_f1_score_means, alternative='greater')
    t_mean_conclusion = f"p-value: {t_mean_p_value:.3f}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score means are statistically greater than subject-specific model F1-score means. Fine-tuned model has higher accuracy." if t_mean_p_value < threshold_p_value else f"p-value: {t_mean_p_value}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score means cannot be proved to be statistically greater than subject-specific model F1-score means. Fine-tuned model is not proved to have higher accuracy."
    print(t_mean_conclusion)

    print("Wilcoxon signed-rank test:")
    w_mean_stat, w_mean_p_value = wilcoxon(fine_tuning_f1_score_means, self_training_f1_score_means, alternative='greater')
    w_mean_conclusion = f"p-value: {w_mean_p_value:.3f}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score means are statistically greater than subject-specific model F1-score means. Fine-tuned model has higher accuracy." if w_mean_p_value < threshold_p_value else f"p-value: {t_mean_p_value}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score means cannot be proved to be statistically greater than subject-specific model F1-score means. Fine-tuned model is not proved to have higher accuracy."
    print(w_mean_conclusion)


    print()
    print("Null hypothesis: Fine-tuned model has as much stable stable accuracies as single-subject model")
    print("Alternative hypothesis: Fine-tuned model has more stable accuracies than single-subject model")

    print("Paried T-test:")
    t_std_stat, t_std_p_value = ttest_rel(fine_tuning_f1_score_stds, self_training_f1_score_stds, alternative='less') 
    t_std_conclusion = f"p-value: {t_std_p_value:.3f}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score STDs are statistically less than subject-specific model F1-score STDs. Fine-tuned model is more stable." if t_std_p_value < threshold_p_value else f"p-value: {t_mean_p_value}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score STDs cannot be proved to be statistically less than subject-specific model F1-score STDs. Fine-tuned model is not proved to be more stable."
    print(t_std_conclusion)

    print("Wilcoxon signed-rank test:")
    w_std_stat, w_std_p_value = wilcoxon(fine_tuning_f1_score_stds, self_training_f1_score_stds, alternative='less')
    w_std_conclusion = f"p-value: {w_std_p_value:.3f}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score STDs are statistically less than subject-specific model F1-score STDs. Fine-tuned model is more stable." if w_mean_p_value < threshold_p_value else f"p-value: {t_mean_p_value}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score STDs cannot be proved to be statistically less than subject-specific model F1-score STDs. Fine-tuned model is not proved to be more stable."
    print(w_std_conclusion)

    print('The end')