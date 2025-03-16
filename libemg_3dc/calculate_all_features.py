import os
import time
import numpy as np
from libemg.datasets import *
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter

from global_utils.print_with_date import *


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # all_subject_ids = list(range(0,22))
    # train_subject_ids, non_train_subject_ids = train_test_split(all_subject_ids, test_size=0.3)
    # validate_subject_ids, test_subject_ids = train_test_split(non_train_subject_ids, test_size=0.5)

    train_subject_ids = [8, 21, 14, 3, 20, 15, 18, 10, 0, 17, 6, 19, 16, 11, 5]
    validate_subject_ids = [1, 9, 12]
    test_subject_ids = [2, 13, 4, 7]
    non_train_subject_ids = validate_subject_ids + test_subject_ids
    all_subject_ids = train_subject_ids + non_train_subject_ids

    printd('Started fetching data')
    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']
    printd('Finished fetching data')

    printd('Started separating train/test data')
    train_subjects = odh.isolate_data("subjects", train_subject_ids) 
    non_train_subjects = odh.isolate_data("subjects", non_train_subject_ids) 
    
    printd('Finished separating train/test data')

    # apply a standardization: (x - mean)/std
    printd('Started applying filters')
    filter = Filter(sampling_frequency=1000)
    filter_dic = {
        "name": "standardize",
        "data": train_subjects
    }
    filter.install_filters(filter_dic)
    filter.filter(train_subjects)
    filter.filter(non_train_subjects)
    printd('Fiinished applying filters')

    validate_subjects = non_train_subjects.isolate_data("subjects", validate_subject_ids) 
    test_subjects = non_train_subjects.isolate_data("subjects", test_subject_ids) 

    # perform windowing
    printd('Started parsing windows')
    train_windows, train_metadata = train_subjects.parse_windows(200,100)
    validate_windows, validate_metadata = validate_subjects.parse_windows(200,100)
    test_windows, test_metadata = test_subjects.parse_windows(200,100)
    printd('Finished parsing windows')

    os.makedirs(f"{script_dir}/data/features/train", exist_ok=True)
    for metadata_key in train_metadata.keys():
        np.savetxt(f"{script_dir}/data/features/train/metadata_{metadata_key}.csv", train_metadata[metadata_key], fmt="%d", delimiter=",")

    os.makedirs(f"{script_dir}/data/features/validate", exist_ok=True)
    for metadata_key in validate_metadata.keys():
        np.savetxt(f"{script_dir}/data/features/validate/metadata_{metadata_key}.csv", validate_metadata[metadata_key], fmt="%d", delimiter=",")

    os.makedirs(f"{script_dir}/data/features/test", exist_ok=True)
    for metadata_key in test_metadata.keys():
        np.savetxt(f"{script_dir}/data/features/test/metadata_{metadata_key}.csv", test_metadata[metadata_key], fmt="%d", delimiter=",")

    
    printd('Started extracting features')
    feature_extractor = FeatureExtractor()
    all_feature_names = feature_extractor.get_feature_list()
    resource_expensive_feature_names = ['SAMPEN', 'FUZZYEN']
    all_feature_names.sort(key=lambda s: 1 if s in resource_expensive_feature_names else 0)

    for feature_name in all_feature_names:
        try:
            start = time.perf_counter()

            train_features = feature_extractor.extract_features([feature_name], train_windows)
            np.savetxt(f"{script_dir}/data/features/train/{feature_name}.csv", train_features[feature_name], delimiter=",")

            validate_features  = feature_extractor.extract_features([feature_name], validate_windows)
            np.savetxt(f"{script_dir}/data/features/validate/{feature_name}.csv", validate_features[feature_name], delimiter=",")

            test_features  = feature_extractor.extract_features([feature_name], test_windows)
            np.savetxt(f"{script_dir}/data/features/test/{feature_name}.csv", test_features[feature_name], delimiter=",")
            
            printd(f"{feature_name} - {(time.perf_counter() - start):.2f} s")
        except Exception as e:
            printd(f"{feature_name} - error occurred", e)

    printd('Finished extracting features')

    printd('The end')