
### 1. Finetuning the ConvNet
- Start from pretrained weights for all layers
- Do not freeze any layers
- Continue training on subject-specific data, updating all weights (Conv + FC)
- You can optionally reset the FC layer (especially if class labels or style changes)

📌 Goal: Let the model adapt fully to the new subject.


### ✅ 2. ConvNet as Fixed Feature Extractor
- tart from pretrained weights
- Freeze all convolutional layers (no updates to feature extractor)
- Reset the final FC layer (random init)
- Train only the FC layer on subject-specific data

📌 Goal: Use pretrained Conv layers as a static feature extractor, and just learn a new classifier head for the new subject.


### Option 1 "Reinitialize FC, Unfreeze FC"
#### If you have enough subject-specific data for fine-tuning, and you think there is a value in slightly adapting the convolutional filters to the new subject.
1. Train all layers on all subjects except one.
2. Reinitialize final FC layer
3. Unfreeze final FC layer to finetune on the excluded subject

### Option 2 "Reinitialize FC, Unfreeze all"
#### If you have very little data from the new subject, and want to avoid overfitting. FC layer is quick to learn and sufficient to classify known features.
1. Train all layers on all subjects except one.
2. Reinitialize final FC layer
3. Unfreeze all layers to finetune on the excluded subject

### Option 3 "Unfreeze FC"
1. Train all layers on all subjects except one.
2. Reinitialize final FC layer
3. Unfreeze final FC layer to finetune on the excluded subject

### Option 4 "Unfreeze all"
1. Train all layers on all subjects except one.
2. Reinitialize final FC layer
3. Unfreeze all layers to finetune on the excluded subject

#### Pros and cons for keeping final FC layer 
The final FC layer has already seen enough samples of each class across subjects and learned general boundaries. During finetuning, its weights can adapt gradually to the new subject’s version of those classes. This is especially useful if: all subjects perform the same set of movements (i.e., the classes are the same), and the inter-subject variance is not too extreme.
But:
	•	You may inherit biases from previous subjects.
	•	The FC layer may resist adapting to the new subject because it starts from a non-neutral position (wrong priors).
	•	Especially with little data, the model may stick too close to its pretrained boundaries, leading to under-adaptation.