import os
import sys
import numpy as np
from libemg.datasets import *
from typing import cast


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment




if __name__ == "__main__":

    training_results = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(ready) copy.json')

    single_subject_results: list[NeuralNetworkSingleSubjectTrainingExperiment] = [
        cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
    

    statistics_by_subject = []
    for subject_id in range(0, 22):
        subject_records = [result for result in single_subject_results if result.subject_id == subject_id]
        subject_f1_scores = [result.test_result["f1_score"] for result in subject_records]

        statistics_by_subject.append({
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        })


    f1_scores = [result.test_result["f1_score"] for result in single_subject_results]
    mean = np.mean(f1_scores) # 0.8685460978848195
    std = np.std(f1_scores) # 0.08909870973418495

    print('end')