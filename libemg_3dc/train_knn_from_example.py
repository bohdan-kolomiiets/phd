import os
import sys
import sklearn
import sklearn.metrics
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from global_utils.print_with_date import *
from libemg_3dc.utils.stored_features import *

def train_KNN_classifier(train_features, train_metadata, test_features, test_metadata):
    printd('Started training KNN classifier')
    training_features_dict = {}
    training_features_dict['training_features'] = train_features
    training_features_dict['training_labels'] = train_metadata["classes"]
    knn_classifier = EMGClassifier('KNN', model_parameters={'n_neighbors': 5})
    knn_classifier.fit(feature_dictionary=training_features_dict.copy())
    printd('Finished training KNN classifier')

    printd('Started predicting with KNN classifier')
    predicted_classes, class_probabilities = knn_classifier.run(test_features)
    printd('Finished predicting with KNN classifier')

    classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    accuracy = classification_report['accuracy']
    printd(F'KNN Accuracy: {accuracy:.3f}')
    f1_score = classification_report['macro avg']['f1-score']
    printd(F'KNN F1-score: {f1_score:.3f}')
    print(sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))


if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_metadata, train_features = read_features(f"{script_dir}/data/features/train")
    validate_metadata, validate_features = read_features(f"{script_dir}/data/features/validate")
    test_metadata, test_features = read_features(f"{script_dir}/data/features/test")
    non_train_features = concatenate_feature_sets(set1=validate_features, set2=test_features)
    non_train_metadata = concatenate_feature_sets(set1=validate_metadata, set2=test_metadata)
    
    train_subject_ids = np.unique(train_metadata['subjects'])
    validate_subject_ids = np.unique(validate_metadata['subjects'])
    test_subject_ids = np.unique(test_metadata['subjects'])
    print(f"train subject ids ({len(train_subject_ids)}): {train_subject_ids}")
    print(f"validate subject ids ({len(validate_subject_ids)}): {validate_subject_ids}")
    print(f"test subject ids ({len(test_subject_ids)}): {test_subject_ids}")
    
    train_KNN_classifier(train_features, train_metadata, test_features=non_train_features, test_metadata=non_train_metadata)

    printd('The end')
    