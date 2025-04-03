import os
import random
import sys
import shutil
import socket
from datetime import datetime
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from global_utils.print_with_date import printd
from global_utils.early_stopping import EarlyStopping
from global_utils.model_checkpoint import ModelCheckpoint
from utils.libemg_deep_learning import make_data_loader


def get_standardization_params(odh):
    ''' Computes the mean and standard deviation of data contained in an OfflineDataHandler.

    Parameters
    ----------
    odh: OfflineDataHandler   
        The data that parameters will be computed from.

    Returns
    ------- 
    mean: np.ndarray
        channel-wise means.
    std:  np.ndarray
        channel-wise standard deviations.
    '''
    data = np.concatenate(odh.data)
    filter_mean = np.mean(data,axis=0)
    filter_std  = np.std(data, axis=0)
    assert (filter_std != 0).any()
    return filter_mean, filter_std

def apply_standardization_params(odh, mean_by_channels, std_by_channels):
    for record_index in range(len(odh.data)):
        odh.data[record_index] = (odh.data[record_index] - mean_by_channels) / std_by_channels
    return odh


class CNN(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256, generator=None, tensorboard_writer:SummaryWriter=None):
        super().__init__()

        self.generator = generator

        self.tensorboard_writer = tensorboard_writer

        self.n_output = n_output

        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels # 10
        l1_filters = n_filters # 64
        l2_filters = n_filters // 2 # 32
        l3_filters = n_filters // 4 # 16
        
        # setup layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(l0_filters, l1_filters, kernel_size=5), # Size([batches, 10, 200]) -> Size([batches, 64, 196])
            nn.BatchNorm1d(l1_filters), # compute mean and variance over all elements and apply normalization, for each of the 64 output channels 
            nn.ReLU(),
            nn.Conv1d(l1_filters, l2_filters, kernel_size=5), # Size([batches, 64, 196]) -> Size([batches, 32, 192])
            nn.BatchNorm1d(l2_filters),
            nn.ReLU(),
            nn.Conv1d(l2_filters, l3_filters, kernel_size=5),  # Size([batches, 32, 192]) -> Size([batches, 16, 188])
            nn.BatchNorm1d(l3_filters),
            nn.ReLU()
        )

        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.convolutional_layers(example_input) # Size([1, 16, 188])
        size_after_conv = conv_output.view(-1).shape[0] # 16 * 188 = 3008
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, self.n_output) # fully connected layer that transforms 3008 inputs to 11 gesture classes
        
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)

        CNN.initialize_with_glorot_weight_zero_bias(self, generator)

        self = CNN._try_move_to_accelerator(self)
        

    def forward(self, x):
        x = self.convolutional_layers(x) # Size([64, 10, 200]) -> Size([64, 16, 188])
        x = x.view(x.shape[0],-1) # Size([64, 16, 188]) -> Size([64, 3008])
        # x = self.act(x) # fix: redundant
        x = self.output_layer(x) # Size([64, 3008]) -> Size([64, 11])
        # x = self.softmax(x) # fix: incorrect, CrossEntropyLoss expects raw logits from the linear layer as internally it does softmax calculations by itself, otherwise leads to tiny gradients and slow training
        return x

    @staticmethod
    def _try_move_to_accelerator(obj):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.accelerator.is_available():
            obj = obj.to(torch.accelerator.current_accelerator())
        return obj
    
    # TODO: each module has modules() inside, you can make it static and pass any layer to initialize it this way
    # TODO: during training try to call this method with layer passed and check if it will reset it 
    # TODO: compare with calling module.reset_parameters() before calling glorot_weight_zero_bias
    # TODO: rename this emthod?
    @staticmethod
    def initialize_with_glorot_weight_zero_bias(module, generator=None):
        """
        Based on
        https://robintibor.github.io/braindecode/source/braindecode.torch_ext.html#module-braindecode.torch_ext.init
        Initalize parameters of all modules by initializing weights with glorot uniform/xavier initialization, and setting biases to zero. Weights from batch norm layers are set to 1.
        """
        for sum_module in module.modules():
            if hasattr(sum_module, "weight"):
                if not ("BatchNorm" in sum_module.__class__.__name__):
                    nn.init.xavier_uniform_(sum_module.weight, gain=1, generator=generator)
                else:
                    nn.init.constant_(sum_module.weight, 1)
            if hasattr(sum_module, "bias"):
                if sum_module.bias is not None:
                    nn.init.constant_(sum_module.bias, 0)


    def fit(self, dataloader_dictionary, verbose, model_checkpoint: ModelCheckpoint):

        early_stopping = EarlyStopping(patience=4, acceptable_delta=0.03, verbose=True)

        optimizer = optim.Adam(self.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=adam_learning_rate/100)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
        loss_function = nn.CrossEntropyLoss()
        # setup a place to log training metrics
        self.log = {"training_loss":[],
                    "validation_loss": [],
                    "training_accuracy": [],
                    "validation_accuracy": []}
        # now start the training
        for epoch in range(num_epochs):
            
            #training set
            self.train()
            for data, labels in dataloader_dictionary["training_dataloader"]:
                optimizer.zero_grad()
                data = CNN._try_move_to_accelerator(data)
                labels = CNN._try_move_to_accelerator(labels)
                output = self.forward(data) # Size([64, 10, 200]) -> Size([64, 11])
                loss = loss_function(output, labels) # labels.shape == Size([64])
                loss.backward()
                optimizer.step()
                acc = (torch.argmax(output, dim=1) == labels).float().mean()

                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc.item())]
            
            # validation set
            self.eval()
            for data, labels in dataloader_dictionary["validation_dataloader"]:
                data = CNN._try_move_to_accelerator(data)
                labels = CNN._try_move_to_accelerator(labels)
                output = self.forward(data)
                loss = loss_function(output, labels)
                acc = (torch.argmax(output, dim=1) == labels).float().mean()
                # log it
                self.log["validation_loss"] += [(epoch, loss.item())]
                self.log["validation_accuracy"] += [(epoch, acc.item())]
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in self.log['training_accuracy'] if i[0]==epoch])
                epoch_valoss = np.mean([i[1] for i in self.log['validation_loss'] if i[0]==epoch])
                epoch_vaacc  = np.mean([i[1] for i in self.log['validation_accuracy'] if i[0]==epoch])
                printd(f"{epoch+1}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  valoss:{epoch_valoss:.2f}  vaacc:{epoch_vaacc:.2f}")
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalars(main_tag='Training loss', tag_scalar_dict= { 'training_loss': epoch_trloss }, global_step=epoch)
                    self.tensorboard_writer.add_scalars(main_tag='Training accuracy', tag_scalar_dict= { 'training_accuracy': epoch_tracc }, global_step=epoch)
                    self.tensorboard_writer.add_scalars(main_tag='Validation loss', tag_scalar_dict= { 'validation_loss': epoch_valoss }, global_step=epoch)
                    self.tensorboard_writer.add_scalars(main_tag='Validation accuracy', tag_scalar_dict= { 'validation_accuracy': epoch_vaacc }, global_step=epoch)
                    self.tensorboard_writer.add_scalars(
                        main_tag='Learning curve',
                        tag_scalar_dict= { 
                            'training_loss': epoch_trloss, 
                            'training_accuracy': epoch_tracc, 
                            'validation_loss': epoch_valoss, 
                            'validation_accuracy': epoch_vaacc 
                        },
                        global_step=epoch
                    )
                    self.tensorboard_writer.flush()
            scheduler.step(epoch_valoss)
            print("Current LR:", scheduler.get_last_lr())

            model_checkpoint.save_if_better(epoch_valoss, self)

            early_stopping(epoch_valoss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.eval()

    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = CNN._try_move_to_accelerator(x)
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = CNN._try_move_to_accelerator(x)
        y = self.forward(x)
        y = self.softmax(y)
        return y.cpu().detach().numpy()
    
    def apply_transfer_strategy(self, strategy):
        """
        strategy: 
            - "finetune_with_fc_reset" (don't freeze any layers, reset FC output layer)
            - "finetune_without_fc_reset" (don't freeze any layers, no reset)
            - "feature_extractor_with_fc_reset" (freeze convolutional layers, reset FC output layer) 
            - "feature_extractor_without_fc_reset" (freeze convolutional layers, no reset)
        """
        if strategy == "finetune_with_fc_reset":
            for parameter in self.parameters():
                parameter.requires_grad = True
            self.output_layer = nn.Linear(self.output_layer.in_features, self.n_output) #  Kaiming Uniform for weights, zeros for bias
            CNN.initialize_with_glorot_weight_zero_bias(self.output_layer, self.generator) # Glorot uniform/xavier for weights, zero for bias
            self.output_layer = CNN._try_move_to_accelerator(self.output_layer)
        
        elif strategy == "finetune_without_fc_reset":
            for parameter in self.parameters():
                parameter.requires_grad = True

        elif strategy == "feature_extractor_with_fc_reset":
            for parameter in self.convolutional_layers.parameters():
                parameter.requires_grad = False
            self.output_layer = nn.Linear(self.output_layer.in_features, self.n_output) #  Kaiming Uniform for weights, zeros for bias
            CNN.initialize_with_glorot_weight_zero_bias(self.output_layer, self.generator) # Glorot uniform/xavier for weights, zero for bias
            self.output_layer = CNN._try_move_to_accelerator(self.output_layer)

        elif strategy == "feature_extractor_without_fc_reset":
            for parameter in self.convolutional_layers.parameters():
                parameter.requires_grad = False

        else:
            raise ValueError("Invalid strategy. Choose: 'finetune', 'feature_extractor', or 'continue_all'.")

        return self


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

def split_by_sets(dataset):
    """
    all_repetition_ids = np.unique(np.concatenate(odh.reps)) # [0, 1, 2, 3]

    returns (train_measurements, validate_measurements, test_measurements)
    """
    train_measurements = dataset.isolate_data("sets",[0])
    non_train_measurements  = dataset.isolate_data("sets",[1])
    validate_measurements = non_train_measurements.isolate_data("reps",[0, 1])
    test_measurements = non_train_measurements.isolate_data("reps",[2, 3])
    return (train_measurements, validate_measurements, test_measurements)


seed = 123
seeds = [0, 1, 7, 42, 123, 1337, 2020, 2023] # to check variance or stability

num_subjects = 1
num_epochs = 50
batch_size = 64

# Adam optimizer params
adam_learning_rate = 1e-3
adam_weight_decay=0 # 1e-5

# ReduceLROnPlateau scheduler params
reduceLROnPlateau_factor=0.7
reduceLROnPlateau_patience=3

transfer_strategy = None


def get_experiment_name(detail = ''): 
    optimizer_config = f'Adam(learning_rate={adam_learning_rate},weight_decay={adam_weight_decay})'
    # scheduler_config = f'ReduceLROnPlateau(factor={reduceLROnPlateau_factor},patience={reduceLROnPlateau_patience})'
    scheduler_config = f'CosineAnnealingLR(T_max={num_epochs}, eta_min={adam_learning_rate/100})'
    return f'transfer_learning.{transfer_strategy}.{detail}.subjects={num_subjects},num_epochs={num_epochs},batch_size={batch_size},{optimizer_config},{scheduler_config}'


experiments = [
    { 'transfer_strategy': "finetune_with_fc_reset" },
    { 'transfer_strategy': "finetune_without_fc_reset" },
    { 'transfer_strategy': "feature_extractor_with_fc_reset" },
    { 'transfer_strategy': "feature_extractor_without_fc_reset" }
]

print("torch version: ", torch.__version__)
if __name__ == "__main__":
    
    set_seed(seed)

    all_subject_ids = list(range(0,num_subjects))

    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']

    for experiment in experiments:
        num_subjects = experiment['num_subjects'] if 'num_subjects' in experiment else num_subjects
        num_epochs = experiment['num_epochs'] if 'num_epochs' in experiment else num_epochs
        batch_size = experiment['batch_size'] if 'batch_size' in experiment else batch_size
        
        adam_learning_rate = experiment['adam_learning_rate'] if 'adam_learning_rate' in experiment else adam_learning_rate
        adam_weight_decay = experiment['adam_weight_decay'] if 'adam_weight_decay' in experiment else adam_weight_decay

        reduceLROnPlateau_factor = experiment['reduceLROnPlateau_factor'] if 'reduceLROnPlateau_factor' in experiment else reduceLROnPlateau_factor
        reduceLROnPlateau_patience = experiment['reduceLROnPlateau_patience'] if 'reduceLROnPlateau_patience' in experiment else reduceLROnPlateau_patience

        transfer_strategy = experiment['transfer_strategy'] if 'transfer_strategy' in experiment else transfer_strategy

        printd('experiment: ', get_experiment_name())

        post_subject_ids = [0]
        pre_subject_ids = [subject_id for subject_id in all_subject_ids if subject_id not in post_subject_ids]

        # pretrain model
        pre_tensorboard_writer = create_tensorboard_writer(folder_path=os.path.join("tensorboard", 'libemg_3dc', get_experiment_name(detail='pre')))
        pre_checkpoint_path=f'libemg_3dc/transfer_learning/checkpoints/pretrained.pt'
        pre_model_checkpoint = ModelCheckpoint(pre_checkpoint_path, verbose=False)
        if os.path.isfile(pre_checkpoint_path): 
            print('Load existing pre-trained model')
            pre_model_state, pre_model_config = pre_model_checkpoint.load_best_model_config()
            standardization_mean = pre_model_config['standardization_mean']
            standardization_std = pre_model_config['standardization_std']
            generator = torch.Generator().manual_seed(seed)
            pre_model = CNN(n_output=pre_model_config['n_output'], n_channels=pre_model_config['n_channels'], n_samples=pre_model_config['n_samples'], n_filters=pre_model_config['n_filters'], generator=generator, tensorboard_writer=None)
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

            generator = torch.Generator().manual_seed(seed)
            pre_model = CNN(n_output, n_channels, n_samples, n_filters = batch_size, generator=generator, tensorboard_writer=pre_tensorboard_writer)
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
            pre_model.fit(pre_dataloader_dictionary, verbose=True, model_checkpoint=pre_model_checkpoint)
            pre_model_state, pre_model_config = pre_model_checkpoint.load_best_model_config()
            pre_model.load_state_dict(pre_model_state)
        
            pre_classifier = EMGClassifier(None)
            pre_classifier.model = pre_model
            predicted_classes, class_probabilities = pre_classifier.run(pre_test_windows)
            print('Pre-training finished.\nMetrics: \n', sklearn.metrics.classification_report(y_true=pre_test_metadata['classes'], y_pred=predicted_classes, output_dict=False))


        # prepare finetuning data
        post_subject_ids = [0] # substitude to the one from pretraining set to check if I didn't break the model, it should work great on the data is has seen already
        post_measurements =  odh.isolate_data("subjects", post_subject_ids)
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
        post_tensorboard_writer = create_tensorboard_writer(folder_path=os.path.join("tensorboard", 'libemg_3dc', get_experiment_name(detail='post')))
        generator = torch.Generator().manual_seed(seed)
        post_model = CNN(n_output=pre_model_config['n_output'], n_channels=pre_model_config['n_channels'], n_samples=pre_model_config['n_samples'], n_filters=pre_model_config['n_filters'], generator=generator, tensorboard_writer=post_tensorboard_writer)
        post_model.load_state_dict(pre_model_state)
        # post_model.apply_transfer_strategy(strategy=transfer_strategy) # comment to check reproducibility

        post_checkpoint_path=f'libemg_3dc/transfer_learning/checkpoints/{transfer_strategy}.pt'
        post_model_checkpoint = ModelCheckpoint(post_checkpoint_path, verbose=False)
        post_dataloader_dictionary = {
            "training_dataloader": make_data_loader(post_train_windows, post_train_metadata["classes"], batch_size=batch_size, generator=generator),
            "validation_dataloader": make_data_loader(post_validate_windows, post_validate_metadata["classes"], batch_size=batch_size, generator=generator)
            }
        post_model.fit(post_dataloader_dictionary, verbose=True, model_checkpoint=post_model_checkpoint)
        post_model_state, _ = post_model_checkpoint.load_best_model_config()
        post_model.load_state_dict(post_model_state)

        post_classifier = EMGClassifier(None)
        post_classifier.model = post_model
        predicted_classes, class_probabilities = post_classifier.run(post_test_windows)
        print('Post-training: \n', sklearn.metrics.classification_report(y_true=post_test_metadata['classes'], y_pred=predicted_classes, output_dict=False))

# TODO:
# - achieve reproducibility - maybe I need to pass optimizer as well as ChatGPT proposes
# - run updated network with fixes on the "split by subjects" setup 
# - compare with the case if I tried to fit on the new subject without finetuning on him (baseline1)
# - compare with the case if you were training only on this subject (baseline1)
# - do cycle of excluding subjects one by one, store, metrics - calculate metrics for baselines, and for transfer learning, calculate mean and std for imporovements across subjects 

# Intersting: F-score can depend significantly on the seed (0.79 - 0.88)
# Will need to track variation across experiemnts to get average: generator = torch.Generator().manual_seed(seed + i)

# intermediate results:
# finetune_with_fc_reset -    macro avg       0.85      0.83      0.83      1052
# finetune_without_fc_reset - macro avg       0.87      0.84      0.84      1052
# feature_extractor_with_fc_reset - macro avg       0.49      0.45      0.44      1052
# feature_extractor_without_fc_reset - macro avg       0.48      0.45      0.43      1052