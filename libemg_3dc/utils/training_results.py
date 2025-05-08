import os
from typing import Union
from pathlib import Path
import uuid
import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class TrainingResult(ABC):
    id: str
    model_type: str
    experiment_type: str

    _registry = []

    @classmethod
    def register(cls, subclass):
        cls._registry.append(subclass)
        return subclass

    @classmethod
    @abstractmethod
    def _can_create_from_json_dict(cls, json_dict: dict) -> bool:
        pass

    @classmethod
    @abstractmethod
    def _from_json_dict(cls, json_dict: dict) -> "TrainingResult":
        pass

    @abstractmethod
    def to_json_dict(self) -> dict:
        pass

    def __repr__(self):
        return self.id

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "TrainingResult":
        for subclass in cls._registry:
            try:
                if subclass._can_create_from_json_dict(json_dict):
                    return subclass._from_json_dict(json_dict)
            except Exception as e:
                print(f"Failed to create subclass: {e}")
                raise e 
        raise ValueError("No subclass can handle the provided JSON")
    



class TrainingResults:
    
    def __init__(self, path: Union[str, Path], results: list[TrainingResult]  = None):
        self.path = path
        self.data = results or []

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
        self.data.append(result)
    

    def cleanup(self, model_type: str, experiment_type: str):
        self.data = [result for result in self.data if result.model_type != model_type and result.experiment_type != experiment_type]

    def save(self):
        with open(self.path, "w") as file:
            json.dump([result.to_json_dict() for result in self.data], file, indent=2)



@TrainingResult.register
@dataclass
class UnknownTrainingResult(TrainingResult):
    model_type: str = field(init=False, default='unknown')
    experiment_type: str = field(init=False, default='unknown')

    def to_json_dict(self):
        return {
            "id": self.id
        }

    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"experiment_type:unknown" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        metadata = dict(pair.split(":") for pair in json_dict["id"].split("."))
        return cls(
            id=json_dict["id"]
        )

@TrainingResult.register
@dataclass
class NeuralNetworkSingleSubjectTrainingResult(TrainingResult):
    subject_id: str
    training_repetitions: np.ndarray
    validation_repetitions: np.ndarray
    test_repetitions: np.ndarray
    training_data: list[str] = field(default_factory=list)
    test_result: dict = field(default=None)

    model_type: str = field(init=False, default='NN')
    experiment_type: str = field(init=False, default='single-subject')

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
            training_repetitions=training_repetitions, 
            validation_repetitions=validation_repetitions, 
            test_repetitions=test_repetitions
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
            "training_data": self.training_data,
            "test_result": self.test_result
        }
    
    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"model_type:{cls.model_type}" in id and f"experiment_type:{cls.experiment_type}" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        
        metadata = dict(pair.split(":") for pair in json_dict["id"].split("."))

        return cls(
            id=json_dict["id"],
            subject_id=str(metadata["subject_id"]), 
            training_repetitions=np.array(json.loads(metadata["training_repetitions"])), 
            validation_repetitions=np.array(json.loads(metadata["validation_repetitions"])), 
            test_repetitions=np.array(json.loads(metadata["test_repetitions"])), 
            training_data=json_dict.get("training_data", []),
            test_result=json_dict.get("test_result", None)
        )