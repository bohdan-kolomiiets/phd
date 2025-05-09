import os
import sys
import numpy as np
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkOtherSubjectsTrainingExperiment


if __name__ == "__main__":

    training_results = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(ready).json')

    experiments = [
        cast(NeuralNetworkOtherSubjectsTrainingExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkOtherSubjectsTrainingExperiment)]
    
    train_subjects_f1_scores = [experiment.test_result["f1_score"]["train_subjects"] for experiment in experiments]
    train_subjects_f1_score_mean = np.mean(train_subjects_f1_scores) 
    train_subjects_f1_score_std = np.std(train_subjects_f1_scores)
    print("train subjects:")
    print(f"mean F1-score: {train_subjects_f1_score_mean}")
    print(f"std F1-score: {train_subjects_f1_score_std}")

    test_subjects_f1_scores = [experiment.test_result["f1_score"]["test_subjects"] for experiment in experiments]
    test_subjects_f1_score_mean = np.mean(test_subjects_f1_scores) 
    test_subjects_f1_score_std = np.std(test_subjects_f1_scores)
    print("test subjects:")
    print(f"mean F1-score: {test_subjects_f1_score_mean}")
    print(f"std F1-score: {test_subjects_f1_score_std}")

    print('The end')