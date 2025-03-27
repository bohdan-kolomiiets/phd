import os
import sys
import time
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from global_utils.print_with_date import *
from libemg_3dc.utils.stored_features import *

def train_SVM_classifier(train_features, train_metadata, test_features, test_metadata):
    printd('Started training SVM classifier')
    start = time.perf_counter()

    training_features_dict = {}
    training_features_dict['training_features'] = train_features
    training_features_dict['training_labels'] = train_metadata["classes"]

    svm_classifier = EMGClassifier(None)
    svm_classifier.model = CalibratedClassifierCV(estimator=SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3), cv=5)
    # svm_classifier = EMGClassifier('SVM', model_parameters={"kernel": "linear", "probability": True, "random_state": 0, "verbose": False, "cache_size": 1800})

    svm_classifier.fit(feature_dictionary=training_features_dict.copy())
    printd(f'Finished training SVM classifier. Took {(time.perf_counter() - start):.2f}s')

    printd('Started predicting with SVM classifier')
    predicted_classes, class_probabilities = svm_classifier.run(test_features)
    printd('Finished predicting with SVM classifier')

    classification_report = sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True)
    accuracy = classification_report['accuracy']
    printd(F'SVM Accuracy: {accuracy:.3f}')
    f1_score = classification_report['macro avg']['f1-score']
    printd(F'SVM F1-score: {f1_score:.3f}')
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
    
    train_SVM_classifier(train_features, train_metadata, test_features=non_train_features, test_metadata=non_train_metadata)

    printd('The end')
    