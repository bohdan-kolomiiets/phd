import os
import random
import sys
import shutil
import json
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.model_selection import LeaveOneOut
import torch
from torch.utils.tensorboard import SummaryWriter
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier
from typing import cast


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from global_utils.print_with_date import printd
from global_utils.model_checkpoint import ModelCheckpoint
from utils.libemg_deep_learning import make_data_loader
from utils.libemg_offline_data_handler_utils import get_standardization_params, apply_standardization_params, split_by_sets
from utils.neural_networks.libemg_cnn_v1 import CNN_V1 as CNN
from utils.subject_repetitions_cross_validation import generate_repetitions_folds
from utils.training_results import TrainingResults, NeuralNetworkSingleSubjectTrainingResult




if __name__ == "__main__":

    training_results = TrainingResults.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(ready) copy.json')

    single_subject_results: list[NeuralNetworkSingleSubjectTrainingResult] = [
        cast(NeuralNetworkSingleSubjectTrainingResult, result) for result in training_results.data if isinstance(result, NeuralNetworkSingleSubjectTrainingResult)]
    

    statistics_by_subject = []
    for subject_id in [str(subject_id) for subject_id in range(0, 22)]:
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