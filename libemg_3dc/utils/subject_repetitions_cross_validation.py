import random
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np

def generate_random_folds(items, n_splits, n_items):
    for _ in range(n_splits):
        fold_indexes = random.sample(range(len(items)), n_items)
        yield fold_indexes

def select_test_repetition_folds(reps: np.ndarray):
    # for test_indexes in generate_random_folds(reps, n_splits=3, n_items=1):
    #     train_indexes = np.setdiff1d(range(len(reps)), test_indexes)
    #     yield reps[train_indexes], reps[test_indexes]
        
    loo = LeaveOneOut()
    for train_indexes, test_indexes in loo.split(reps):
        yield reps[train_indexes], reps[test_indexes]


def select_validate_repetition_folds(reps: np.ndarray):
    # for validate_indexes in generate_random_folds(reps, n_splits=3, n_items=1):
    #     train_indexes = np.setdiff1d(range(len(reps)), validate_indexes)
    #     yield reps[train_indexes], reps[validate_indexes]
        
    loo = LeaveOneOut()
    for train_indexes, validate_indexes in loo.split(reps):
        yield reps[train_indexes], reps[validate_indexes]


def generate_repetitions_folds(all_repetitions: np.ndarray):
    """
    all_reps = np.array([1,2,3,4,5,6,7,8])
    LOO: LOO for test reps(8), LOO for validate reps (7) = 56
    Custom: 3 folds of 2 test reps, 3 folds of 2 validate reps = 9
    Custom: 3 folds of 1 test rep, 3 folds of 1 validate reps = 9
    Custom + LDO: 3 folds of 1 test rep, LOO for validate reps (7) = 21 !!!
    """
    folds = []
    for (non_test_reps, test_reps) in select_test_repetition_folds(all_repetitions):
        for (train_reps, validate_reps) in select_validate_repetition_folds(non_test_reps):
            folds.append({
                'train_reps': train_reps,
                'validate_reps': validate_reps,
                'test_reps': test_reps 
            })
    return folds