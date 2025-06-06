

## DS1: 3DC

https://libemg.github.io/libemg/documentation/data/data.html

https://github.com/LibEMG/3DCDataset

https://libemg.github.io/libemg/documentation/data/features/data/features.html#feature-performance

### Dataset files structure

```
 _3DCDataset
    # Participant1
        # test
            # EMG
                # 3dc_EMG_gesture_0_0.txt
                # ...
                # 3dc_EMG_gesture_0_10
        # train
```


### Deep Learning for Electromyographic Hand Gesture Signal Classification Using Transfer Learning (Ulysse Cotˆ e-Allard)
https://sci-hub.se/https://ieeexplore.ieee.org/document/8630679 

This work’s hypothesis is that general, informative features can be learned from the large amounts of data generated by aggregating the signals of multiple users, thus reducing the recording burden while enhancing gesture recognition. Consequently, this paper proposes applying transfer learning on aggregated data from multiple users, while leveraging the capacity of deep learning algorithms to learn discriminant features from large datasets. Two datasets comprised of 19 and 17 able-bodied participants respectively (the first one is employed for pre-training) were recorded for this work, using the Myo Armband. A third Myo Armband dataset was taken from the NinaPro database and is comprised of 10 able-bodied participants. Three different deep learning networks employing three different modalities as input (raw EMG, Spectrograms and Continuous Wavelet Transform (CWT)) are tested on the second and third dataset. The proposed transfer learning scheme is shown to systematically and significantly enhance the performance for all three networks on the two datasets, achieving an offline accuracy of 98.31% for 7 gestures over 17 participants for the CWT-based ConvNet and 68.98% for 18 gestures over 10 participants for the raw EMG-based ConvNet.


INTRODUCTION:
A previous work [7] has already shown that learning simultaneously from multiple subjects significantly enhances the ConvNet’s performance whilst reducing the size of the required training dataset typically seen with deep learning algorithms.

Labeled Data Acquisition Protocol:
During recording, participants were instructed to stand up and have their forearm parallel to the floor and supported by themselves. For each of them, the armband was systematically tightened to its maximum and slid up the user’s forearm, until the circumference of the armband matched that of the forearm. This was done in an effort to reduce bias from the researchers, and to emulate the wide variety of armband positions that endusers without prior knowledge of optimal electrode placement might use (see Fig. 2). While the electrode placement was not controlled for, the orientation of the armband was always such that the blue light bar on the Myo was facing towards the hand of the subject. Note that this is the case for both left and right handed subjects. The raw sEMG data of the Myo is what is made available with this dataset.

TRANSFER LEARNING:
As the data recording was purposefully as unconstrained as possible, the armband’s orientation from one subject to another can vary widely. As such, to allow for the use of TL, automatic alignment is a necessary first step. The alignment for each subject was made by identifying the most active channel (calculated using the IEMG feature) for each gesture on the first subject. On subsequent subjects, the channels were then circularly shifted until their activation for each gesture matched those of the first subject as closely as possible.