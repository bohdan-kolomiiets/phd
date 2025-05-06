import os
from typing import Union
from pathlib import Path
import uuid
import numpy as np
import json
from abc import ABC, abstractmethod


class TrainingResult(ABC):

    _registry = []

    @classmethod
    def register(cls, subclass):
        cls._registry.append(subclass)
        return subclass

    @staticmethod
    @abstractmethod
    def can_create_from_json_dict(json_dict: dict):
        pass

    @staticmethod
    @abstractmethod
    def from_json_dict(json_dict: dict):
        pass

    @staticmethod
    def from_json_dict(json_dict: dict):
        for subclass in TrainingResult._registry:
            if subclass.can_create_from_json_dict(json_dict):
                return subclass.from_json_dict(json_dict)
        raise ValueError("No subclass can handle the provided JSON")


class TrainingResults:
    
    def __init__(self, path: Union[str, Path], results: list[TrainingResult]  = None):
        self.path = path
        self.results = results or []

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingResults":

        results = []

        path = Path(path)
        if os.path.exists(path): 
            with open(path, mode='r', encoding='utf-8') as file:
                json_dicts = json.load(fp=file) 

            for json_dict in json_dicts:
                try:
                    result = TrainingResult.from_json_dict(json_dict)
                    results.append(result)
                except Exception as e:
                    print(f'Skipped invalid result: {e}')
            
        return cls(path, results)
    

    def append(self, result: TrainingResult):
        self.results.append(result)
    

    def cleanup(self, model_type: str, experiment_type: str):
        self.results = [result for result in self.results if result.model_type != model_type and result.experiment_type != experiment_type]

    def save(self):
        with open(self.path, "w") as file:
            json.dump([result.to_json_dict() for result in self.results], file, indent=2)




@TrainingResult.register
class NeuralNetworkSingleSubjectTrainingResult(TrainingResult):
    """
    {
        "id": "{guid}",
        "type": "single-subject", // 22*8*7=1232
        "subject": 0,
        "repetitions-split": {
            "training-reps": [1,2,3,4,5,6],
            "validation-reps": [7],
            "test-reps": [8]
        },
        "training-curves": [],
        "test-report": {}
    },
    """

    model_type = 'NN'
    experiment_type = 'single-subject'


    def __init__(self, 
                 id: str,
                 subject_id: str, 
                 training_repetitions: np.ndarray, 
                 validation_repetitions: np.ndarray, 
                 test_repetitions: np.ndarray,
                 training_data: list[str],
                 test_result: dict):
        
        self.id = id
        self.subject_id = subject_id 
        self.training_repetitions = training_repetitions
        self.validation_repetitions = validation_repetitions
        self.test_repetitions = test_repetitions
        self.training_data = training_data
        self.test_result = test_result


    @classmethod
    def create(cls, 
               subject_id: str, 
               training_repetitions: np.ndarray, validation_repetitions: np.ndarray, test_repetitions: np.ndarray):
        
        metadata = {
            "model_type": cls.model_type,
            "experiment_type": cls.experiment_type,
            "subject_id": subject_id,
            "training_repetitions": json.dumps(training_repetitions.tolist()),
            "validation_repetitions": json.dumps(validation_repetitions.tolist()),
            "test_repetitions": json.dumps(test_repetitions.tolist())
        }
        id = ".".join(f"{k}:{v}" for k, v in metadata.items())

        return cls(
            id=id,
            subject_id=subject_id, 
            training_repetitions= metadata["training_repetitions"], 
            validation_repetitions= metadata["validation_repetitions"], 
            test_repetitions=metadata["test_repetitions"],
            training_data=[],
            test_result=None
        )


    def save_training_data(self, epoch_training_data: dict): 
        self.training_data.append(json.dumps(epoch_training_data))

    def save_test_result(self, classification_report: dict): 
        self.test_result = {
            "f1_score": classification_report['macro avg']['f1-score'],
            "report": json.dumps(classification_report)
        }


    def to_json_dict(self):
        return {
            "id": self.id,
            # "model_type": self.__class__.model_type,
            # "experiment_type": self.__class__.experiment_type,
            # "subject_id": self.subject_id,
            # "repetitions_split": {
            #     "training": json.dumps(self.training_repetitions.tolist()),
            #     "validation": json.dumps(self.validation_repetitions.tolist()),
            #     "test": json.dumps(self.test_repetitions.tolist())
            # },
            "training_data": self.training_data,
            "test_result": self.test_result
            # "model-filename": f"{self.id}.pt",
            # "tensorboard-directory": self.id
        }
    
    
    @classmethod
    def can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id")
        if id is None:
            return False
        return f"model_type:{cls.model_type}" in id and f"experiment_type:{cls.experiment_type}" in id 
        # return json_dict.get("model_type") == cls.model_type and json_dict.get("experiment_type") == cls.experiment_type


    @classmethod
    def from_json_dict(cls, json_dict):
        
        metadata = {key: value for key, value in (pair.split(":") for pair in json_dict["id"].split("."))}

        return cls(
            id=json_dict["id"],
            subject_id=str(metadata["subject_id"]), 
            training_repetitions=np.array(json.loads(metadata["training_repetitions"])), 
            validation_repetitions=np.array(json.loads(metadata["validation_repetitions"])), 
            test_repetitions=np.array(json.loads(metadata["test_repetitions"])), 
            training_data=json_dict["training_data"],
            test_result=json_dict["test_result"]
        )
    

    def __repr__(self):
        return self.id