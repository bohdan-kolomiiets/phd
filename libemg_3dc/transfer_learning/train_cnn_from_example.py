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
from utils.libemg_deep_learning import make_data_loader


class CNN(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256, generator=None, tensorboard_writer:SummaryWriter=None):
        super().__init__()

        self.tensorboard_writer = tensorboard_writer

        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels # 10
        l1_filters = n_filters # 64
        l2_filters = n_filters // 2 # 32
        l3_filters = n_filters // 4 # 16
        # let's manually setup those layers
        # simple layer 1
        self.conv1 = nn.Conv1d(l0_filters, l1_filters, kernel_size=5)
        # Input:  (batch_size, 10, 200) → through conv1 → Output: (batch_size, 64, 196)
        self.bn1   = nn.BatchNorm1d(l1_filters) # compute mean and variance over all elements and apply normalization, for each of the 64 output channels 
        # simple layer 2
        self.conv2 = nn.Conv1d(l1_filters, l2_filters, kernel_size=5)
        self.bn2   = nn.BatchNorm1d(l2_filters)
        # simple layer 3
        self.conv3 = nn.Conv1d(l2_filters, l3_filters, kernel_size=5)
        self.bn3   = nn.BatchNorm1d(l3_filters)
        # and we need an activation function:
        self.act = nn.ReLU()

        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.conv_only(example_input) # Size([1, 16, 188])
        size_after_conv = conv_output.view(-1).shape[0] # 16 * 188 = 3008
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, n_output) # fully connected layer that transforms 3008 inputs to 11 gesture classes
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)

        self.glorot_weight_zero_bias(generator)

        self = CNN._try_move_to_accelerator(self)
        

    def conv_only(self, x):
        try:
            x = self.conv1(x) # Size([batches, 10, 200]) -> Size([batches, 64, 196])
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x) # Size([batches, 64, 196]) -> Size([batches, 32, 192])
            x = self.bn2(x) 
            x = self.act(x)
            x = self.conv3(x) # Size([batches, 32, 192]) -> Size([batches, 16, 188])
            x = self.bn3(x)
            x = self.act(x)
        except Exception as ex:
            printd(ex)
        return x

    def forward(self, x):
        x = self.conv_only(x) # Size([64, 10, 200]) -> Size([64, 16, 188])
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
    
    def glorot_weight_zero_bias(self, generator=None):
        """
        Based on
        https://robintibor.github.io/braindecode/source/braindecode.torch_ext.html#module-braindecode.torch_ext.init
        """
        for module in self.modules():
            if hasattr(module, "weight"):
                if not ("BatchNorm" in module.__class__.__name__):
                    nn.init.xavier_uniform_(module.weight, gain=1, generator=generator)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def fit(self, dataloader_dictionary, verbose, checkpoints_path):

        early_stopping = EarlyStopping(patience=4, acceptable_change_percentage=0.03, verbose=True, path=checkpoints_path)

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
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
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
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
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

            early_stopping(epoch_valoss, self)
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
        return y.cpu().detach().numpy()


def train_CNN_classifier(train_windows, train_metadata, validate_windows, validate_metadata, test_windows, test_metadata):

    generator = torch.Generator()
    generator.manual_seed(seed)

    writer_log_dir=os.path.join("tensorboard", 'libemg_3dc', get_experiment_name())
    if os.path.exists(writer_log_dir):
        shutil.rmtree(writer_log_dir)
    writer = SummaryWriter(writer_log_dir)

    n_output = len(np.unique(np.vstack(train_metadata['classes'])))
    n_channels = train_windows.shape[1]
    n_samples = train_windows.shape[2]
    model = CNN(n_output, n_channels, n_samples, n_filters = batch_size, generator=generator, tensorboard_writer=writer)

    train_dataloader = make_data_loader(train_windows, train_metadata["classes"], batch_size=batch_size, generator=generator)
    valid_dataloader = make_data_loader(validate_windows, validate_metadata["classes"], batch_size=batch_size, generator=generator)
    dataloader_dictionary = {
        "training_dataloader": train_dataloader,
        "validation_dataloader": valid_dataloader
        }
    
    # add_model_graph_to_tensorboard(model, train_dataloader, writer)

    # model.load_state_dict(torch.load(f'{get_experiment_name()}.pt'))
    model.fit(dataloader_dictionary, verbose=True, checkpoints_path=f'libemg_3dc/split_by_samples/checkpoints/{get_experiment_name()}.pt')


    classifier = EMGClassifier(None)
    classifier.model = model

    predicted_classes, class_probabilities = classifier.run(test_windows)

    # classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    # accuracy = classification_report['accuracy']
    # printd(F'CNN Accuracy: {accuracy:.3f}')
    # f1_score = classification_report['macro avg']['f1-score']
    # printd(F'CNN F1-score: {f1_score:.3f}')
    print(sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))

    # om = OfflineMetrics()
    # metrics = ['CA','AER','INS','REJ_RATE','CONF_MAT','RECALL','PREC','F1']    
    # results = om.extract_offline_metrics(metrics, y_true=test_metadata['classes'], y_predictions=predicted_classes, null_label=2)
    # for key in results:
    #     printd(f"{key}: {results[key]}")


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

seed = 123
seeds = [0, 1, 7, 42, 123, 1337, 2020, 2023] # to check variance or stability

num_subjects = 2

num_epochs = 50

batch_size = 64

# Adam optimizer params
adam_learning_rate = 1e-3
adam_weight_decay=0 # 1e-5

# ReduceLROnPlateau scheduler params
reduceLROnPlateau_factor=0.7
reduceLROnPlateau_patience=3


experiments = [
    { 'batch_size': 64 }
]

def get_experiment_name(): 
    optimizer_config = f'Adam(learning_rate={adam_learning_rate},weight_decay={adam_weight_decay})'
    # scheduler_config = f'ReduceLROnPlateau(factor={reduceLROnPlateau_factor},patience={reduceLROnPlateau_patience})'
    scheduler_config = f'CosineAnnealingLR(T_max={num_epochs}, eta_min={adam_learning_rate/100})'
    return f'Split by repetition. subjects={num_subjects},num_epochs={num_epochs},batch_size={batch_size},{optimizer_config},{scheduler_config}'


if __name__ == "__main__":
    
    set_seed(seed)

    all_subject_ids = list(range(0,num_subjects))
    
    # printd('Started fetching data')
    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']
    # printd('Finished fetching data')

    # all_repetition_ids = np.unique(np.concatenate(odh.reps)) # [0, 1, 2, 3]
    train_measurements = odh.isolate_data("sets",[0])
    non_train_measurements  = odh.isolate_data("sets",[1])
    validate_measurements = non_train_measurements.isolate_data("reps",[0, 1])
    test_measurements = non_train_measurements.isolate_data("reps",[2, 3])

    # apply a standardization: (x - mean)/std
    # printd('Started applying filters')
    filter = Filter(sampling_frequency=1000)
    filter_dic = {
        "name": "standardize",
        "data": train_measurements
    }
    filter.install_filters(filter_dic)
    filter.filter(train_measurements)
    filter.filter(validate_measurements)
    filter.filter(test_measurements)
    # printd('Fiinished applying filters')

    # perform windowing
    # printd('Started parsing windows')
    train_windows, train_metadata = train_measurements.parse_windows(200,100)
    validate_windows, validate_metadata = validate_measurements.parse_windows(200,100)
    test_windows, test_metadata = test_measurements.parse_windows(200,100)
    # printd('Finished parsing windows')

    for experiment in experiments:
        num_subjects = experiment['num_subjects'] if 'num_subjects' in experiment else num_subjects
        num_epochs = experiment['num_epochs'] if 'num_epochs' in experiment else num_epochs
        batch_size = experiment['batch_size'] if 'batch_size' in experiment else batch_size
        
        adam_learning_rate = experiment['adam_learning_rate'] if 'adam_learning_rate' in experiment else adam_learning_rate
        adam_weight_decay = experiment['adam_weight_decay'] if 'adam_weight_decay' in experiment else adam_weight_decay

        reduceLROnPlateau_factor = experiment['reduceLROnPlateau_factor'] if 'reduceLROnPlateau_factor' in experiment else reduceLROnPlateau_factor
        reduceLROnPlateau_patience = experiment['reduceLROnPlateau_patience'] if 'reduceLROnPlateau_patience' in experiment else reduceLROnPlateau_patience

        printd('experiment: ', get_experiment_name())
        train_CNN_classifier(train_windows, train_metadata, validate_windows, validate_metadata, test_windows, test_metadata)