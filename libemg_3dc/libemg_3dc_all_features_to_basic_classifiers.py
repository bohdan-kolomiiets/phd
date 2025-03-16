import os
import sys
from datetime import datetime
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://libemg.github.io/libemg/documentation/features/features.html#feature-performance


# _3DCDataset
    # Participant1
        # test
            # EMG
                # 3dc_EMG_gesture_0_0.txt
                # ...
                # 3dc_EMG_gesture_0_10
        # train

_print=print
def print(*args, **kw):
    _print("[%s]" % (datetime.now()),*args, **kw)



class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, classes, batch_size=64):
    # first we make the object that holds the data
    obj = DL_input_data(windows, classes)
    # and now we make a dataloader with that object
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    # this function is used internally by the dataloader (see line 46)
    # it describes how we stitch together the examples into a batch
    signals, labels = [], []
    for signal, label in batch:
        # concat signals onto list signals
        signals += [signal]
        labels += [label]
    # convert back to tensors
    signals = torch.stack(signals)
    labels = torch.stack(labels).long()
    return signals, labels

class CNN(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256):
        super().__init__()
        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels
        l1_filters = n_filters
        l2_filters = n_filters // 2
        l3_filters = n_filters // 4
        # let's manually setup those layers
        # simple layer 1
        self.conv1 = nn.Conv1d(l0_filters, l1_filters, kernel_size=5)
        self.bn1   = nn.BatchNorm1d(l1_filters)
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
        conv_output   = self.conv_only(example_input)
        size_after_conv = conv_output.view(-1).shape[0]
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, n_output)
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)

        self.__try_move_to_accelerator(self)
        

    def conv_only(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.conv_only(x)
        x = x.view(x.shape[0],-1)
        x = self.act(x)
        x = self.output_layer(x)
        return self.softmax(x)

    def __try_move_to_accelerator(self, obj):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.accelerator.is_available():
            obj = obj.to(torch.accelerator.current_accelerator())
        return obj


    def fit(self, dataloader_dictionary, learning_rate=1e-3, num_epochs=100, verbose=True):
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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
                data = self.__try_move_to_accelerator(data)
                labels = self.__try_move_to_accelerator(labels)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc.item())]
            # validation set
            self.eval()
            for data, labels in dataloader_dictionary["validation_dataloader"]:
                data = self.__try_move_to_accelerator(data)
                labels = self.__try_move_to_accelerator(labels)
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
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  valoss:{epoch_valoss:.2f}  vaacc:{epoch_vaacc:.2f}")
        self.eval()

    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = self.__try_move_to_accelerator(x)
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = self.__try_move_to_accelerator(x)
        y = self.forward(x)
        return y.cpu().detach().numpy()


def train_CNN_classifier(train_windows, train_metadata, validate_windows, validate_metadata, test_windows, test_metadata):

    train_dataloader = make_data_loader(train_windows, train_metadata["classes"])
    valid_dataloader = make_data_loader(validate_windows, validate_metadata["classes"])

    dataloader_dictionary = {
        "training_dataloader": train_dataloader,
        "validation_dataloader": valid_dataloader
        }
    
    n_output = len(np.unique(np.vstack(train_metadata['classes'])))
    n_channels = train_windows.shape[1]
    n_samples = train_windows.shape[2]
    model = CNN(n_output, n_channels, n_samples, n_filters = 64)
    
    dl_dictionary = {
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "verbose": True
        }
    model.fit(dataloader_dictionary, **dl_dictionary)

    classifier = EMGClassifier(None)
    classifier.model = model

    predicted_classes, class_probabilities = classifier.run(test_windows)

    classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    accuracy = classification_report['accuracy']
    print(F'CNN Accuracy: {accuracy:.3f}')
    f1_score = classification_report['macro avg']['f1-score']
    print(F'CNN F1-score: {f1_score:.3f}')
    _print(sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))


    om = OfflineMetrics()
    metrics = ['CA','AER','INS','REJ_RATE','CONF_MAT','RECALL','PREC','F1']    
    results = om.extract_offline_metrics(metrics, y_true=test_metadata['classes'], y_predictions=predicted_classes, null_label=2)
    for key in results:
        print(f"{key}: {results[key]}")

    return

def train_KNN_classifier(train_features, train_metadata, test_features, test_metadata):
    print('Started training KNN classifier')
    training_features_dict = {}
    training_features_dict['training_features'] = train_features
    training_features_dict['training_labels'] = train_metadata["classes"]
    knn_classifier = EMGClassifier('KNN', model_parameters={'n_neighbors': 5})
    knn_classifier.fit(feature_dictionary=training_features_dict.copy())
    print('Finished training KNN classifier')

    print('Started predicting with KNN classifier')
    predicted_classes, class_probabilities = knn_classifier.run(test_features)
    print('Finished predicting with KNN classifier')

    classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    accuracy = classification_report['accuracy']
    print(F'KNN Accuracy: {accuracy:.3f}')
    f1_score = classification_report['macro avg']['f1-score']
    print(F'KNN F1-score: {f1_score:.3f}')
    _print(sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))

def train_SVM_classifier(train_features, train_metadata, test_features, test_metadata):
    print('Started training SVM classifier')
    training_features_dict = {}
    training_features_dict['training_features'] = train_features
    training_features_dict['training_labels'] = train_metadata["classes"]
    svm_classifier = EMGClassifier('SVM', model_parameters={"kernel": "linear", "probability": True, "random_state": 0})
    svm_classifier.fit(feature_dictionary=training_features_dict.copy())
    print('Finished training SVM classifier')

    print('Started predicting with SVM classifier')
    predicted_classes, class_probabilities = svm_classifier.run(test_features)
    print('Finished predicting with SVM classifier')

    classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    accuracy = classification_report['accuracy']
    print(F'SVM Accuracy: {accuracy:.3f}')
    f1_score = classification_report['macro avg']['f1-score']
    print(F'SVM F1-score: {f1_score:.3f}')
    _print(sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))
    

if __name__ == "__main__":
    
    all_subject_ids = list(range(0,22))
    # all_subject_ids = list(range(0,5))
    
    train_subject_ids, non_train_subject_ids = train_test_split(all_subject_ids, test_size=0.3)

    print('Started fetching data')
    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']
    print('Finished fetching data')

    print('Started separating train/test data')
    train_subjects = odh.isolate_data("subjects", train_subject_ids) 
    non_train_subjects = odh.isolate_data("subjects", non_train_subject_ids) 
    
    print('Finished separating train/test data')

    # apply a standardization: (x - mean)/std
    print('Started applying filters')
    filter = Filter(sampling_frequency=1000)
    filter_dic = {
        "name": "standardize",
        "data": train_subjects
    }
    filter.install_filters(filter_dic)
    filter.filter(train_subjects)
    filter.filter(non_train_subjects)
    print('Fiinished applying filters')

    validate_subject_ids, test_subject_ids = train_test_split(non_train_subject_ids, test_size=0.5)
    validate_subjects = non_train_subjects.isolate_data("subjects", validate_subject_ids) 
    test_subjects = non_train_subjects.isolate_data("subjects", test_subject_ids) 

    # perform windowing
    print('Started parsing windows')
    train_windows, train_metadata = train_subjects.parse_windows(200,100)
    non_train_windows, non_train_metadata = non_train_subjects.parse_windows(200,100)
    validate_windows, validate_metadata = validate_subjects.parse_windows(200,100)
    test_windows, test_metadata = test_subjects.parse_windows(200,100)
    print('Finished parsing windows')

    # train_CNN_classifier(train_windows, train_metadata, validate_windows, validate_metadata, test_windows, test_metadata)

    print('Started extracting features')
    feature_extractor = FeatureExtractor()
    all_feature_names = feature_extractor.get_feature_list()[:5]
    train_features = feature_extractor.extract_features(all_feature_names, train_windows)
    non_train_features  = feature_extractor.extract_features(all_feature_names, non_train_windows)
    print('Finished extracting features')

    train_KNN_classifier(train_features, train_metadata, test_features=non_train_features, test_metadata=non_train_metadata)

    train_SVM_classifier(train_features, train_metadata, test_features=non_train_features, test_metadata=non_train_metadata)

    print('the end')