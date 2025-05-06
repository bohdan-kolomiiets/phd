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


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from global_utils.print_with_date import printd
from global_utils.model_checkpoint import ModelCheckpoint
from utils.libemg_deep_learning import make_data_loader
from utils.libemg_offline_data_handler_utils import get_standardization_params, apply_standardization_params, split_by_sets
from utils.neural_networks.libemg_cnn_v1 import CNN_V1 as CNN
from utils.subject_repetitions_cross_validation import generate_repetitions_folds


def add_model_graph_to_tensorboard(model, dataloader, tensorboard_writer):
    data, labels = next(iter(dataloader))
    data = CNN._try_move_to_accelerator(data)
    tensorboard_writer.add_graph(model, data)
    tensorboard_writer.flush()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=False)

def create_tensorboard_writer(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    return SummaryWriter(folder_path)


seed = 123
seeds = [0, 1, 7, 42, 123, 1337, 2020, 2023] # to check variance or stability

num_subjects = 22
num_epochs = 50
batch_size = 64

# Adam optimizer params
adam_learning_rate = 1e-3
adam_weight_decay=0 # 1e-5

# ReduceLROnPlateau scheduler params
reduceLROnPlateau_factor=0.7
reduceLROnPlateau_patience=3

transfer_strategy = 'feature_extractor_with_fc_reset'

results_base_path = 'libemg_3dc/prove_pretraining_helps/pretraining_with_finetuning/cnn_v1_results.json'
if os.path.exists(results_base_path): 
    with open(results_base_path) as results_file:
        results = json.load(results_file)
else:
    results = []


def create_pre_training_log_callback(subject_fold_name):
    
    def pre_training_log_callback(epoch, epoch_trloss, epoch_tracc, epoch_valoss, epoch_vaacc):
        training_curves = results["subject-folds"][subject_fold_name]["pre-training"]["repetition-folds"]["training-curves"]
        
        if training_curves == None:
            training_curves = []
        
        training_curves.append({
            "epoch": epoch,
            "training_accuracy": epoch_tracc,
            "training_loss": epoch_trloss,
            "validation_loss": epoch_valoss,
            "validation_accuracy": epoch_vaacc
        })
        
    return pre_training_log_callback


from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
for train, test in kf.split([1,2,3,4,5,6,7,8]):
    print("%s %s" % (train, test))


if __name__ == "__main__":
    
    set_seed(seed)

    all_subject_ids = list(range(0,num_subjects))

    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']

    subject_llo = LeaveOneOut()
    for (pre_subject_ids, post_subject_ids) in subject_llo.split(all_subject_ids):

        subject_fold_name=f'LLO-{','.join(pre_subject_ids)}'
        results_iteration_path = f'{results_base_path}/llo-{pre_subject_ids[0]}'

        # pretrain model
        pre_checkpoint_path=results["subject-folds"][subject_fold_name]["pre-training"]["repetition-folds"]["training-curves"]
        pre_model_checkpoint = ModelCheckpoint(pre_checkpoint_path, verbose=False)
        if os.path.isfile(pre_checkpoint_path): 
            print('Load existing pre-trained model')
            pre_model_state, pre_model_config = pre_model_checkpoint.load_best_model_config()
            standardization_mean = pre_model_config['standardization_mean']
            standardization_std = pre_model_config['standardization_std']
            generator = torch.Generator().manual_seed(seed)
            pre_model = CNN(n_output=pre_model_config['n_output'], n_channels=pre_model_config['n_channels'], n_samples=pre_model_config['n_samples'], n_filters=pre_model_config['n_filters'], generator=generator)
            pre_model.load_state_dict(pre_model_state)
        else:
            print('Pre-train model')
            # prepare pretraining data
            pre_measurements  = odh.isolate_data("subjects", pre_subject_ids)
            (pre_train_measurements, pre_validate_measurements, pre_test_measurements) = split_by_sets(pre_measurements)
            # apply standardization
            standardization_mean, standardization_std = get_standardization_params(pre_train_measurements)
            pre_train_measurements = apply_standardization_params(pre_train_measurements, standardization_mean, standardization_std)
            pre_validate_measurements = apply_standardization_params(pre_validate_measurements, standardization_mean, standardization_std)
            pre_test_measurements = apply_standardization_params(pre_test_measurements, standardization_mean, standardization_std)
            # perform windowing
            pre_train_windows, pre_train_metadata = pre_train_measurements.parse_windows(200,100)
            pre_validate_windows, pre_validate_metadata = pre_validate_measurements.parse_windows(200,100)
            pre_test_windows, pre_test_metadata = pre_test_measurements.parse_windows(200,100)

            n_output = len(np.unique(np.vstack(pre_train_metadata['classes'])))
            n_channels = pre_train_windows.shape[1]
            n_samples = pre_train_windows.shape[2]
            n_filters = batch_size

            pre_training_log_callback = create_pre_training_log_callback(subject_fold_name=f'LLO-{','.join(pre_subject_ids)}')
            generator = torch.Generator().manual_seed(seed)
            pre_model = CNN(n_output, n_channels, n_samples, n_filters = batch_size, generator=generator)
            pre_model_checkpoint.set_config("n_output", n_output)
            pre_model_checkpoint.set_config("n_channels", n_channels)
            pre_model_checkpoint.set_config("n_samples", n_samples)
            pre_model_checkpoint.set_config("n_filters", n_filters)
            pre_model_checkpoint.set_config("standardization_mean", standardization_mean)
            pre_model_checkpoint.set_config("standardization_std", standardization_std)
            pre_dataloader_dictionary = {
                "training_dataloader": make_data_loader(pre_train_windows, pre_train_metadata["classes"], batch_size=batch_size, generator=generator),
                "validation_dataloader": make_data_loader(pre_validate_windows, pre_validate_metadata["classes"], batch_size=batch_size, generator=generator)
                }
            pre_model.fit(pre_dataloader_dictionary, num_epochs, adam_learning_rate, adam_weight_decay, verbose=True, model_checkpoint=pre_model_checkpoint, training_log_callback=pre_training_log_callback)
            pre_model_state, pre_model_config = pre_model_checkpoint.load_best_model_config()
            pre_model.load_state_dict(pre_model_state)
        
            pre_classifier = EMGClassifier(None)
            pre_classifier.model = pre_model
            predicted_classes, class_probabilities = pre_classifier.run(pre_test_windows)
            print('Pre-training finished.\nMetrics: \n', sklearn.metrics.classification_report(y_true=pre_test_metadata['classes'], y_pred=predicted_classes, output_dict=False))


        # prepare finetuning data
        post_subject_ids = [0] # substitude to the one from pretraining set to check if I didn't break the model, it should work great on the data is has seen already
        post_measurements =  odh.isolate_data("subjects", post_subject_ids)
        repetition_folds = generate_repetitions_folds(all_repetitions=[1,2,3,4,5,6,7,8])
        for repetition_fold in repetition_folds:

        (post_train_measurements, post_validate_measurements, post_test_measurements) = split_by_sets(post_measurements)
        # apply standardization
        post_train_measurements = apply_standardization_params(post_train_measurements, standardization_mean, standardization_std)
        post_validate_measurements = apply_standardization_params(post_validate_measurements, standardization_mean, standardization_std)
        post_test_measurements = apply_standardization_params(post_test_measurements, standardization_mean, standardization_std)
        # perform windowing
        post_train_windows, post_train_metadata = post_train_measurements.parse_windows(200,100)
        post_validate_windows, post_validate_metadata = post_validate_measurements.parse_windows(200,100)
        post_test_windows, post_test_metadata = post_test_measurements.parse_windows(200,100)

        # finetune model
        generator = torch.Generator().manual_seed(seed)
        post_model = CNN(n_output=pre_model_config['n_output'], n_channels=pre_model_config['n_channels'], n_samples=pre_model_config['n_samples'], n_filters=pre_model_config['n_filters'], generator=generator)
        post_model.load_state_dict(pre_model_state)
        post_model.apply_transfer_strategy(strategy=transfer_strategy) # comment to check reproducibility

        post_checkpoint_path=f'{results_iteration_path}/checkpoints/{transfer_strategy}.pt'
        post_model_checkpoint = ModelCheckpoint(post_checkpoint_path, verbose=False)
        post_dataloader_dictionary = {
            "training_dataloader": make_data_loader(post_train_windows, post_train_metadata["classes"], batch_size=batch_size, generator=generator),
            "validation_dataloader": make_data_loader(post_validate_windows, post_validate_metadata["classes"], batch_size=batch_size, generator=generator)
            }
        post_model.fit(post_dataloader_dictionary, num_epochs, adam_learning_rate, adam_weight_decay, verbose=True, model_checkpoint=post_model_checkpoint)
        post_model_state, _ = post_model_checkpoint.load_best_model_config()
        post_model.load_state_dict(post_model_state)

        post_classifier = EMGClassifier(None)
        post_classifier.model = post_model
        predicted_classes, class_probabilities = post_classifier.run(post_test_windows)
        print('Post-training: \n', sklearn.metrics.classification_report(y_true=post_test_metadata['classes'], y_pred=predicted_classes, output_dict=False))
