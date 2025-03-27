

## ReduceLROnPlateau,factor=0.5,patience=4
```
              precision    recall  f1-score   support

           0       0.80      0.99      0.88      1537
           1       0.20      0.26      0.23      1544
           2       0.62      0.33      0.43      1539
           3       0.29      0.17      0.21      1548
           4       0.54      0.59      0.56      1530
           5       0.32      0.19      0.24      1517
           6       0.30      0.14      0.19      1539
           7       0.70      0.48      0.57      1536
           8       0.31      0.26      0.28      1536
           9       0.26      0.37      0.31      1536
          10       0.19      0.44      0.27      1539

    accuracy                           0.38     16901
   macro avg       0.41      0.38      0.38     16901
weighted avg       0.41      0.38      0.38     16901
```
```
[2025-03-22 23:53:38.389783] 0: trloss:2.23  tracc:0.32  valoss:2.25  vaacc:0.29
[2025-03-22 23:53:46.906141] 1: trloss:2.07  tracc:0.49  valoss:2.22  vaacc:0.32
[2025-03-22 23:53:55.672727] 2: trloss:2.01  tracc:0.55  valoss:2.22  vaacc:0.32
[2025-03-22 23:54:04.960120] 3: trloss:1.97  tracc:0.58  valoss:2.19  vaacc:0.35
[2025-03-22 23:54:13.884488] 4: trloss:1.95  tracc:0.61  valoss:2.19  vaacc:0.35
[2025-03-22 23:54:21.511757] 5: trloss:1.93  tracc:0.63  valoss:2.17  vaacc:0.37
[2025-03-22 23:54:29.069723] 6: trloss:1.91  tracc:0.65  valoss:2.16  vaacc:0.38
[2025-03-22 23:54:36.721412] 7: trloss:1.90  tracc:0.66  valoss:2.19  vaacc:0.35
[2025-03-22 23:54:44.474662] 8: trloss:1.89  tracc:0.67  valoss:2.18  vaacc:0.36
[2025-03-22 23:54:52.585935] 9: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-22 23:55:00.116603] 10: trloss:1.87  tracc:0.68  valoss:2.18  vaacc:0.36
[2025-03-22 23:55:07.693142] 11: trloss:1.86  tracc:0.69  valoss:2.15  vaacc:0.39
[2025-03-22 23:55:15.533351] 12: trloss:1.86  tracc:0.70  valoss:2.14  vaacc:0.40
[2025-03-22 23:55:23.146441] 13: trloss:1.85  tracc:0.70  valoss:2.15  vaacc:0.39
[2025-03-22 23:55:31.002057] 14: trloss:1.85  tracc:0.71  valoss:2.17  vaacc:0.37
[2025-03-22 23:55:38.676556] 15: trloss:1.84  tracc:0.71  valoss:2.16  vaacc:0.39
[2025-03-22 23:55:46.395672] 16: trloss:1.84  tracc:0.72  valoss:2.15  vaacc:0.39
[2025-03-22 23:55:53.893389] 17: trloss:1.83  tracc:0.72  valoss:2.17  vaacc:0.37
[2025-03-22 23:56:01.442364] 18: trloss:1.81  tracc:0.74  valoss:2.15  vaacc:0.38
[2025-03-22 23:56:08.957408] 19: trloss:1.79  tracc:0.76  valoss:2.14  vaacc:0.40
[2025-03-22 23:56:16.539488] 20: trloss:1.78  tracc:0.77  valoss:2.15  vaacc:0.39
[2025-03-22 23:56:24.157034] 21: trloss:1.78  tracc:0.78  valoss:2.15  vaacc:0.39
[2025-03-22 23:56:31.645776] 22: trloss:1.77  tracc:0.79  valoss:2.15  vaacc:0.39
[2025-03-22 23:56:39.100248] 23: trloss:1.77  tracc:0.79  valoss:2.13  vaacc:0.41
[2025-03-22 23:56:46.736604] 24: trloss:1.76  tracc:0.79  valoss:2.16  vaacc:0.38
[2025-03-22 23:56:54.329714] 25: trloss:1.76  tracc:0.79  valoss:2.14  vaacc:0.40
[2025-03-22 23:57:02.052412] 26: trloss:1.76  tracc:0.80  valoss:2.15  vaacc:0.40
[2025-03-22 23:57:09.481674] 27: trloss:1.76  tracc:0.80  valoss:2.13  vaacc:0.41
[2025-03-22 23:57:17.014821] 28: trloss:1.76  tracc:0.80  valoss:2.14  vaacc:0.40
[2025-03-22 23:57:24.490720] 29: trloss:1.75  tracc:0.80  valoss:2.15  vaacc:0.39
[2025-03-22 23:57:31.970409] 30: trloss:1.75  tracc:0.80  valoss:2.14  vaacc:0.40
[2025-03-22 23:57:39.446678] 31: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
[2025-03-22 23:57:47.203010] 32: trloss:1.75  tracc:0.80  valoss:2.14  vaacc:0.40
[2025-03-22 23:57:55.568079] 33: trloss:1.75  tracc:0.80  valoss:2.15  vaacc:0.39
[2025-03-22 23:58:03.047507] 34: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:58:10.539540] 35: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:58:18.097491] 36: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
[2025-03-22 23:58:25.585397] 37: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:58:33.315146] 38: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:58:40.974825] 39: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:58:48.590215] 40: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
[2025-03-22 23:58:56.988115] 41: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:59:05.376398] 42: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
[2025-03-22 23:59:13.183886] 43: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.40
[2025-03-22 23:59:20.706895] 44: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:59:28.742846] 45: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:59:36.854889] 46: trloss:1.75  tracc:0.81  valoss:2.14  vaacc:0.40
[2025-03-22 23:59:45.343555] 47: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
[2025-03-22 23:59:53.359364] 48: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.40
[2025-03-23 00:00:00.976122] 49: trloss:1.75  tracc:0.81  valoss:2.15  vaacc:0.39
```


## ReduceLROnPlateau,factor=0.7,patience=3
1st time
```
              precision    recall  f1-score   support

           0       0.64      0.79      0.70      1690
           1       0.42      0.13      0.20      1696
           2       0.49      0.62      0.55      1684
           3       0.15      0.30      0.20      1659
           4       0.76      0.49      0.59      1692
           5       0.48      0.45      0.46      1685
           6       0.29      0.14      0.19      1685
           7       0.67      0.69      0.68      1686
           8       0.46      0.52      0.49      1674
           9       0.21      0.14      0.17      1675
          10       0.31      0.42      0.36      1683

    accuracy                           0.43     18509
   macro avg       0.44      0.43      0.42     18509
weighted avg       0.44      0.43      0.42     18509
```
2nd time with seed
```
              precision    recall  f1-score   support

           0       0.73      0.95      0.82      1677
           1       0.27      0.23      0.24      1687
           2       0.49      0.24      0.32      1683
           3       0.26      0.60      0.36      1681
           4       0.55      0.68      0.61      1671
           5       0.00      0.00      0.00      1683
           6       0.21      0.47      0.29      1681
           7       0.63      0.16      0.25      1687
           8       0.34      0.31      0.33      1684
           9       0.20      0.08      0.11      1681
          10       0.23      0.22      0.23      1683

    accuracy                           0.36     18498
   macro avg       0.35      0.36      0.32     18498
weighted avg       0.35      0.36      0.32     18498
```
1st time
```
[2025-03-23 00:06:13.432927] 0: trloss:2.24  tracc:0.29  valoss:2.19  vaacc:0.36
[2025-03-23 00:06:20.889831] 1: trloss:2.11  tracc:0.45  valoss:2.17  vaacc:0.37
[2025-03-23 00:06:28.458789] 2: trloss:2.05  tracc:0.51  valoss:2.15  vaacc:0.39
[2025-03-23 00:06:35.800960] 3: trloss:2.00  tracc:0.56  valoss:2.13  vaacc:0.41
[2025-03-23 00:06:43.235915] 4: trloss:1.97  tracc:0.59  valoss:2.12  vaacc:0.42
[2025-03-23 00:06:50.576340] 5: trloss:1.94  tracc:0.62  valoss:2.11  vaacc:0.42
[2025-03-23 00:06:58.107180] 6: trloss:1.92  tracc:0.64  valoss:2.10  vaacc:0.44
[2025-03-23 00:07:05.377155] 7: trloss:1.90  tracc:0.66  valoss:2.09  vaacc:0.46
[2025-03-23 00:07:12.679174] 8: trloss:1.88  tracc:0.68  valoss:2.09  vaacc:0.46
[2025-03-23 00:07:20.084600] 9: trloss:1.87  tracc:0.69  valoss:2.09  vaacc:0.46
[2025-03-23 00:07:27.438312] 10: trloss:1.86  tracc:0.70  valoss:2.08  vaacc:0.46
[2025-03-23 00:07:34.700058] 11: trloss:1.85  tracc:0.71  valoss:2.07  vaacc:0.47
[2025-03-23 00:07:42.305456] 12: trloss:1.84  tracc:0.71  valoss:2.09  vaacc:0.45
[2025-03-23 00:07:50.074170] 13: trloss:1.84  tracc:0.72  valoss:2.07  vaacc:0.47
[2025-03-23 00:07:57.474883] 14: trloss:1.83  tracc:0.73  valoss:2.07  vaacc:0.48
[2025-03-23 00:08:04.888383] 15: trloss:1.82  tracc:0.73  valoss:2.07  vaacc:0.48
[2025-03-23 00:08:12.259480] 16: trloss:1.82  tracc:0.73  valoss:2.10  vaacc:0.45
[2025-03-23 00:08:19.651357] 17: trloss:1.81  tracc:0.74  valoss:2.09  vaacc:0.45
[2025-03-23 00:08:27.114653] 18: trloss:1.81  tracc:0.75  valoss:2.07  vaacc:0.48
[2025-03-23 00:08:34.675503] 19: trloss:1.80  tracc:0.75  valoss:2.07  vaacc:0.48
[2025-03-23 00:08:42.155610] 20: trloss:1.79  tracc:0.76  valoss:2.07  vaacc:0.47
[2025-03-23 00:08:49.550975] 21: trloss:1.79  tracc:0.76  valoss:2.06  vaacc:0.48
[2025-03-23 00:08:57.075100] 22: trloss:1.79  tracc:0.77  valoss:2.07  vaacc:0.47
[2025-03-23 00:09:04.338998] 23: trloss:1.78  tracc:0.77  valoss:2.07  vaacc:0.47
[2025-03-23 00:09:11.786829] 24: trloss:1.78  tracc:0.77  valoss:2.07  vaacc:0.47
[2025-03-23 00:09:19.207679] 25: trloss:1.78  tracc:0.78  valoss:2.07  vaacc:0.48
[2025-03-23 00:09:26.586650] 26: trloss:1.78  tracc:0.78  valoss:2.07  vaacc:0.47
[2025-03-23 00:09:33.834662] 27: trloss:1.77  tracc:0.78  valoss:2.06  vaacc:0.49
[2025-03-23 00:09:41.226873] 28: trloss:1.77  tracc:0.78  valoss:2.07  vaacc:0.47
[2025-03-23 00:09:48.783457] 29: trloss:1.77  tracc:0.79  valoss:2.06  vaacc:0.48
[2025-03-23 00:09:56.267354] 30: trloss:1.77  tracc:0.79  valoss:2.06  vaacc:0.48
[2025-03-23 00:10:03.516718] 31: trloss:1.77  tracc:0.79  valoss:2.05  vaacc:0.49
[2025-03-23 00:10:10.820840] 32: trloss:1.76  tracc:0.79  valoss:2.06  vaacc:0.48
[2025-03-23 00:10:18.291119] 33: trloss:1.76  tracc:0.79  valoss:2.05  vaacc:0.49
[2025-03-23 00:10:25.566764] 34: trloss:1.76  tracc:0.79  valoss:2.06  vaacc:0.48
[2025-03-23 00:10:33.025232] 35: trloss:1.76  tracc:0.79  valoss:2.05  vaacc:0.49
[2025-03-23 00:10:40.372965] 36: trloss:1.76  tracc:0.79  valoss:2.06  vaacc:0.49
[2025-03-23 00:10:47.956449] 37: trloss:1.76  tracc:0.80  valoss:2.07  vaacc:0.48
[2025-03-23 00:10:55.232233] 38: trloss:1.75  tracc:0.80  valoss:2.05  vaacc:0.49
[2025-03-23 00:11:02.599706] 39: trloss:1.75  tracc:0.80  valoss:2.05  vaacc:0.49
[2025-03-23 00:11:09.944988] 40: trloss:1.75  tracc:0.80  valoss:2.05  vaacc:0.49
[2025-03-23 00:11:17.433689] 41: trloss:1.75  tracc:0.80  valoss:2.06  vaacc:0.49
[2025-03-23 00:11:24.912489] 42: trloss:1.75  tracc:0.80  valoss:2.06  vaacc:0.49
[2025-03-23 00:11:32.340018] 43: trloss:1.75  tracc:0.81  valoss:2.05  vaacc:0.49
[2025-03-23 00:11:39.745444] 44: trloss:1.75  tracc:0.81  valoss:2.06  vaacc:0.48
[2025-03-23 00:11:47.465449] 45: trloss:1.75  tracc:0.81  valoss:2.06  vaacc:0.49
[2025-03-23 00:11:55.016528] 46: trloss:1.75  tracc:0.81  valoss:2.06  vaacc:0.48
[2025-03-23 00:12:02.376283] 47: trloss:1.75  tracc:0.81  valoss:2.05  vaacc:0.49
[2025-03-23 00:12:09.778794] 48: trloss:1.74  tracc:0.81  valoss:2.06  vaacc:0.48
[2025-03-23 00:12:17.323975] 49: trloss:1.74  tracc:0.81  valoss:2.04  vaacc:0.50
```
2nd time with seed
```
[2025-03-23 02:01:12.610681] 0: trloss:2.23  tracc:0.31  valoss:2.22  vaacc:0.32
[2025-03-23 02:01:20.027577] 1: trloss:2.08  tracc:0.48  valoss:2.19  vaacc:0.35
[2025-03-23 02:01:27.559436] 2: trloss:2.00  tracc:0.56  valoss:2.21  vaacc:0.32
[2025-03-23 02:01:35.229148] 3: trloss:1.96  tracc:0.60  valoss:2.20  vaacc:0.34
[2025-03-23 02:01:42.768909] 4: trloss:1.93  tracc:0.63  valoss:2.19  vaacc:0.34
[2025-03-23 02:01:50.164662] 5: trloss:1.91  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 02:01:57.853823] 6: trloss:1.90  tracc:0.66  valoss:2.20  vaacc:0.34
[2025-03-23 02:02:05.485330] 7: trloss:1.88  tracc:0.67  valoss:2.18  vaacc:0.35
[2025-03-23 02:02:12.895930] 8: trloss:1.88  tracc:0.68  valoss:2.18  vaacc:0.36
[2025-03-23 02:02:20.278716] 9: trloss:1.87  tracc:0.69  valoss:2.18  vaacc:0.36
[2025-03-23 02:02:27.757921] 10: trloss:1.86  tracc:0.69  valoss:2.17  vaacc:0.36
[2025-03-23 02:02:35.305139] 11: trloss:1.86  tracc:0.70  valoss:2.17  vaacc:0.37
[2025-03-23 02:02:42.737411] 12: trloss:1.85  tracc:0.70  valoss:2.17  vaacc:0.37
[2025-03-23 02:02:50.098580] 13: trloss:1.85  tracc:0.71  valoss:2.17  vaacc:0.37
[2025-03-23 02:02:57.629844] 14: trloss:1.84  tracc:0.71  valoss:2.18  vaacc:0.36
[2025-03-23 02:03:05.263967] 15: trloss:1.84  tracc:0.72  valoss:2.16  vaacc:0.38
[2025-03-23 02:03:13.265664] 16: trloss:1.84  tracc:0.72  valoss:2.18  vaacc:0.36
[2025-03-23 02:03:21.606833] 17: trloss:1.83  tracc:0.72  valoss:2.17  vaacc:0.36
[2025-03-23 02:03:29.499589] 18: trloss:1.83  tracc:0.73  valoss:2.18  vaacc:0.36
[2025-03-23 02:03:37.323927] 19: trloss:1.83  tracc:0.73  valoss:2.17  vaacc:0.36
[2025-03-23 02:03:44.939753] 20: trloss:1.82  tracc:0.73  valoss:2.17  vaacc:0.36
[2025-03-23 02:03:52.395380] 21: trloss:1.82  tracc:0.73  valoss:2.18  vaacc:0.35
[2025-03-23 02:04:00.485573] 22: trloss:1.82  tracc:0.74  valoss:2.18  vaacc:0.36
[2025-03-23 02:04:08.351863] 23: trloss:1.82  tracc:0.74  valoss:2.17  vaacc:0.37
[2025-03-23 02:04:16.116668] 24: trloss:1.81  tracc:0.74  valoss:2.17  vaacc:0.37
[2025-03-23 02:04:24.241348] 25: trloss:1.81  tracc:0.74  valoss:2.18  vaacc:0.35
[2025-03-23 02:04:32.347003] 26: trloss:1.81  tracc:0.74  valoss:2.18  vaacc:0.35
[2025-03-23 02:04:40.609979] 27: trloss:1.81  tracc:0.74  valoss:2.18  vaacc:0.36
[2025-03-23 02:04:48.211626] 28: trloss:1.81  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:04:55.595442] 29: trloss:1.81  tracc:0.75  valoss:2.19  vaacc:0.34
[2025-03-23 02:05:03.221403] 30: trloss:1.81  tracc:0.75  valoss:2.18  vaacc:0.35
[2025-03-23 02:05:10.641915] 31: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:05:17.952910] 32: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:05:25.288896] 33: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:05:32.680096] 34: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:05:39.976609] 35: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.35
[2025-03-23 02:05:47.346934] 36: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.35
[2025-03-23 02:05:54.727742] 37: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:06:02.068181] 38: trloss:1.80  tracc:0.75  valoss:2.19  vaacc:0.35
[2025-03-23 02:06:09.502945] 39: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:06:16.926290] 40: trloss:1.80  tracc:0.76  valoss:2.18  vaacc:0.35
[2025-03-23 02:06:24.241768] 41: trloss:1.80  tracc:0.75  valoss:2.18  vaacc:0.36
[2025-03-23 02:06:31.585693] 42: trloss:1.80  tracc:0.76  valoss:2.17  vaacc:0.36
[2025-03-23 02:06:38.857743] 43: trloss:1.80  tracc:0.76  valoss:2.18  vaacc:0.36
[2025-03-23 02:06:46.169476] 44: trloss:1.80  tracc:0.76  valoss:2.18  vaacc:0.35
[2025-03-23 02:06:53.383611] 45: trloss:1.80  tracc:0.76  valoss:2.17  vaacc:0.37
[2025-03-23 02:07:01.119432] 46: trloss:1.80  tracc:0.76  valoss:2.18  vaacc:0.35
[2025-03-23 02:07:08.419298] 47: trloss:1.80  tracc:0.76  valoss:2.18  vaacc:0.35
[2025-03-23 02:07:15.621556] 48: trloss:1.80  tracc:0.76  valoss:2.17  vaacc:0.37
[2025-03-23 02:07:23.023228] 49: trloss:1.80  tracc:0.76  valoss:2.19  vaacc:0.34
```

## ReduceLROnPlateau,factor=0.7,patience=1
```
[2025-03-23 00:16:40.249726] 0: trloss:2.26  tracc:0.28  valoss:2.21  vaacc:0.35
[2025-03-23 00:16:47.721773] 1: trloss:2.14  tracc:0.42  valoss:2.18  vaacc:0.36
[2025-03-23 00:16:55.104932] 2: trloss:2.07  tracc:0.48  valoss:2.18  vaacc:0.36
[2025-03-23 00:17:02.544843] 3: trloss:2.04  tracc:0.52  valoss:2.17  vaacc:0.38
[2025-03-23 00:17:09.838201] 4: trloss:2.01  tracc:0.55  valoss:2.16  vaacc:0.38
[2025-03-23 00:17:17.140527] 5: trloss:1.98  tracc:0.57  valoss:2.14  vaacc:0.40
[2025-03-23 00:17:24.578472] 6: trloss:1.96  tracc:0.59  valoss:2.14  vaacc:0.40
[2025-03-23 00:17:32.216140] 7: trloss:1.95  tracc:0.61  valoss:2.15  vaacc:0.39
[2025-03-23 00:17:39.577279] 8: trloss:1.93  tracc:0.62  valoss:2.16  vaacc:0.38
[2025-03-23 00:17:47.070433] 9: trloss:1.92  tracc:0.63  valoss:2.15  vaacc:0.39
[2025-03-23 00:17:54.737200] 10: trloss:1.91  tracc:0.64  valoss:2.15  vaacc:0.39
[2025-03-23 00:18:02.336469] 11: trloss:1.91  tracc:0.65  valoss:2.15  vaacc:0.39
[2025-03-23 00:18:09.960180] 12: trloss:1.90  tracc:0.66  valoss:2.15  vaacc:0.39
[2025-03-23 00:18:17.372490] 13: trloss:1.90  tracc:0.66  valoss:2.14  vaacc:0.40
[2025-03-23 00:18:24.897202] 14: trloss:1.89  tracc:0.66  valoss:2.15  vaacc:0.40
[2025-03-23 00:18:32.510145] 15: trloss:1.89  tracc:0.66  valoss:2.15  vaacc:0.39
[2025-03-23 00:18:40.116664] 16: trloss:1.89  tracc:0.67  valoss:2.14  vaacc:0.40
[2025-03-23 00:18:47.857847] 17: trloss:1.88  tracc:0.67  valoss:2.15  vaacc:0.39
[2025-03-23 00:18:55.541569] 18: trloss:1.88  tracc:0.67  valoss:2.15  vaacc:0.39
[2025-03-23 00:19:03.061065] 19: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:19:10.429064] 20: trloss:1.88  tracc:0.67  valoss:2.15  vaacc:0.39
[2025-03-23 00:19:17.850442] 21: trloss:1.88  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:19:25.227612] 22: trloss:1.88  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:19:32.712296] 23: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:19:40.116581] 24: trloss:1.88  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:19:47.817035] 25: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:19:55.791638] 26: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:20:03.346468] 27: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.40
[2025-03-23 00:20:10.689395] 28: trloss:1.88  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:20:18.038436] 29: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:20:25.775750] 30: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:20:33.699953] 31: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.40
[2025-03-23 00:20:41.306497] 32: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:20:48.649160] 33: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:20:56.039923] 34: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:21:03.779282] 35: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:21:11.331807] 36: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:21:18.649336] 37: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:21:26.081940] 38: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:21:33.558810] 39: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:21:40.949704] 40: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:21:48.292988] 41: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:21:55.753299] 42: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:22:03.437120] 43: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:22:10.782079] 44: trloss:1.87  tracc:0.68  valoss:2.14  vaacc:0.40
[2025-03-23 00:22:18.354288] 45: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:22:26.067930] 46: trloss:1.87  tracc:0.68  valoss:2.16  vaacc:0.38
[2025-03-23 00:22:33.334415] 47: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:22:40.560232] 48: trloss:1.87  tracc:0.68  valoss:2.15  vaacc:0.39
[2025-03-23 00:22:47.940555] 49: trloss:1.87  tracc:0.68  valoss:2.16  vaacc:0.38
```
```
              precision    recall  f1-score   support

           0       0.68      0.60      0.64      1529
           1       0.31      0.13      0.19      1529
           2       0.00      0.00      0.00      1530
           3       0.16      0.64      0.25      1542
           4       0.41      0.33      0.37      1512
           5       0.42      0.15      0.22      1498
           6       0.43      0.20      0.28      1534
           7       0.44      0.32      0.37      1538
           8       0.32      0.47      0.38      1540
           9       0.29      0.33      0.31      1525
          10       0.34      0.21      0.26      1534

    accuracy                           0.31     16811
   macro avg       0.35      0.31      0.30     16811
weighted avg       0.35      0.31      0.30     16811
```

## ReduceLROnPlateau,factor=0.3,patience=1,threshold=1e-2
```
              precision    recall  f1-score   support

           0       0.65      0.95      0.78      1677
           1       0.26      0.27      0.26      1687
           2       0.37      0.26      0.30      1683
           3       0.32      0.34      0.33      1681
           4       0.59      0.72      0.65      1671
           5       0.00      0.00      0.00      1683
           6       0.20      0.54      0.29      1681
           7       0.60      0.13      0.22      1687
           8       0.30      0.29      0.30      1684
           9       0.18      0.12      0.15      1681
          10       0.36      0.35      0.36      1683

    accuracy                           0.36     18498
   macro avg       0.35      0.36      0.33     18498
weighted avg       0.35      0.36      0.33     18498
```
```
[2025-03-23 01:45:21.108570] 0: trloss:2.23  tracc:0.31  valoss:2.22  vaacc:0.32
[2025-03-23 01:45:28.639175] 1: trloss:2.08  tracc:0.48  valoss:2.19  vaacc:0.35
[2025-03-23 01:45:35.960851] 2: trloss:2.00  tracc:0.56  valoss:2.21  vaacc:0.32
[2025-03-23 01:45:43.320653] 3: trloss:1.96  tracc:0.60  valoss:2.20  vaacc:0.34
[2025-03-23 01:45:50.746042] 4: trloss:1.93  tracc:0.62  valoss:2.20  vaacc:0.33
[2025-03-23 01:45:58.238338] 5: trloss:1.93  tracc:0.63  valoss:2.21  vaacc:0.33
[2025-03-23 01:46:05.835606] 6: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:46:13.306121] 7: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:46:20.799668] 8: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:46:28.339709] 9: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:46:35.898516] 10: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:46:43.514879] 11: trloss:1.92  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:46:50.919327] 12: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:46:58.637150] 13: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:47:06.113855] 14: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:47:13.612313] 15: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:47:21.010827] 16: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:47:28.489674] 17: trloss:1.92  tracc:0.64  valoss:2.19  vaacc:0.34
[2025-03-23 01:47:36.079186] 18: trloss:1.91  tracc:0.65  valoss:2.19  vaacc:0.34
[2025-03-23 01:47:43.738111] 19: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:47:51.338062] 20: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:47:58.884605] 21: trloss:1.92  tracc:0.64  valoss:2.19  vaacc:0.34
[2025-03-23 01:48:06.391599] 22: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:48:13.891244] 23: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:48:21.366687] 24: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:48:28.952160] 25: trloss:1.92  tracc:0.65  valoss:2.19  vaacc:0.34
[2025-03-23 01:48:36.694811] 26: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:48:44.198769] 27: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:48:51.666453] 28: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:48:59.339793] 29: trloss:1.92  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:49:06.883767] 30: trloss:1.91  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:49:14.455849] 31: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:49:21.904521] 32: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:49:29.449813] 33: trloss:1.91  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:49:36.955426] 34: trloss:1.92  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:49:44.471944] 35: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:49:52.007760] 36: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:49:59.601136] 37: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:50:07.414332] 38: trloss:1.92  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:50:14.980866] 39: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
[2025-03-23 01:50:22.512380] 40: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:50:30.123849] 41: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.34
[2025-03-23 01:50:37.744775] 42: trloss:1.92  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:50:45.434318] 43: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:50:53.107303] 44: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:51:00.754086] 45: trloss:1.92  tracc:0.64  valoss:2.20  vaacc:0.33
[2025-03-23 01:51:08.177686] 46: trloss:1.92  tracc:0.64  valoss:2.19  vaacc:0.34
[2025-03-23 01:51:15.688873] 47: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.33
[2025-03-23 01:51:23.295821] 48: trloss:1.91  tracc:0.65  valoss:2.19  vaacc:0.34
[2025-03-23 01:51:31.370679] 49: trloss:1.91  tracc:0.65  valoss:2.20  vaacc:0.34
```

# Change learning_rate
## subjects=12,num_epochs=20,Adam(learning_rate=0.001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3) - originally in example
```
0:  trloss:2.01  tracc:0.53  valoss:2.38  vaacc:0.16
19: trloss:1.61  tracc:0.94  valoss:2.38  vaacc:0.16

              precision    recall  f1-score   support
macro avg       0.37      0.37      0.34      9278
```

## subjects=12,num_epochs=20,Adam(learning_rate=0.0001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0:  trloss:2.25  tracc:0.29  valoss:2.35  vaacc:0.17
19: trloss:1.73  tracc:0.84  valoss:2.38  vaacc:0.15

              precision    recall  f1-score   support
   macro avg       0.41      0.38      0.36      9278
```

## subjects=12,num_epochs=20,Adam(learning_rate=1e-05,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0:  trloss:2.37  tracc:0.15  valoss:2.38  vaacc:0.14
19: trloss:2.04  tracc:0.55  valoss:2.35  vaacc:0.17

              precision    recall  f1-score   support
   macro avg       0.30      0.35      0.32      9278
```


# Change batch_size
## subjects=12,num_epochs=20,batch_size=8,Adam(learning_rate=0.001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0: trloss:2.25  tracc:0.28  valoss:2.34  vaacc:0.19
19: trloss:2.07  tracc:0.47  valoss:2.33  vaacc:0.20

              precision    recall  f1-score   support
   macro avg       0.23      0.29      0.24      9278
```

## subjects=12,num_epochs=20,batch_size=16,Adam(learning_rate=0.001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0: trloss:2.13  tracc:0.41  valoss:2.34  vaacc:0.20
19: trloss:1.86  tracc:0.69  valoss:2.36  vaacc:0.17

              precision    recall  f1-score   support
   macro avg       0.36      0.34      0.31      9278
```

### subjects=12,num_epochs=20,batch_size=32,Adam(learning_rate=0.001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0: trloss:2.07  tracc:0.48  valoss:2.39  vaacc:0.14
19: trloss:1.69  tracc:0.86  valoss:2.36  vaacc:0.18

              precision    recall  f1-score   support
   macro avg       0.42      0.37      0.34      9278
```

### subjects=12,num_epochs=20,batch_size=64,Adam(learning_rate=0.001,weight_decay=0),ReduceLROnPlateau(factor=0.7,patience=3)
```
0: trloss:2.01  tracc:0.53  valoss:2.38  vaacc:0.16
19: trloss:1.61  tracc:0.94  valoss:2.38  vaacc:0.16

              precision    recall  f1-score   support
   macro avg       0.37      0.37      0.34      9278
```
