import os
import sys
import numpy as np
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkFineTunedTrainigExperiment


if __name__ == "__main__":

    training_results = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(biased).json')

    experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
    
    statistics_by_subject = []
    for subject_id in range(0, 22):
        subject_experiments = [experiment for experiment in experiments if experiment.subject_id == subject_id]
        subject_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_experiments]
        statistics_by_subject.append({
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        })


    f1_scores = [result.test_result["f1_score"] for result in experiments]
    f1_score_mean = np.mean(f1_scores) # ?
    f1_score_std = np.std(f1_scores) # ?
    print(f"mean F1-score: {f1_score_mean}")
    print(f"std F1-score: {f1_score_std}")

    print('The end')