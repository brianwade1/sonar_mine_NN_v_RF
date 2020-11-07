# Fully Connected Feed-Forward Neural Network and a Random Forest Model for Predictions of Sonar Returns

This project trains two models: a fully connected feed-forward neural network and a random forest to classify the returns from a sonar. The sonar target objects are a metal cylinder (simulated mines) and standard rocks (false mines). The features of the data are the returns from a frequency-modulated chirps rising in frequency.

The hyperparameters of both the neural net and random forest models were optimized with a grid search. The results show that the random forest and neural net achieved comparable accuracy. Examining the receiver operating characteristics (ROC) curve of each model show they both achieve an area under the curve (AUC) over 0.9. Additionally, both models achieved an accuracy over 88%.

---

## Folders and Files

This repo contains the following folders and files:

Folders and Files within:

* [Data](Data) : Raw data and description
  * sonar.dat - Raw data file from data website
  * sonar_coded.csv - Raw data with labels changed to 0 and 1
  * sonar_data_description.txt - Data description

* [Images](Images): Images used in the readme file

* [Models](Models): Trained models
  * NN_model.mat - Neural Network model
  * RF_model.mat - Random Forest model
  * NN_model_parameters.txt - The hyperparameters of the optimal model from the grid search.
  * RF_model_parameters.txt - The hyperparameters of the optimal model from the grid search.

Main Files:

* [sonar_NN_model.m](sonar_NN_model.m) - Grid search of neural net hyperparameters and training of the final neural network.
* [sonar_RF_model.m](sonar_RF_model.m) - Grid search of random forest hyperparameters and training of the final random forest.

---

## Dataset

The data for this project is from the UCI Machine Learning Repository titled [Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)). It is composed of 111 samples of data collected by Terry Sejnowski, now at the Salk Institute and the University of California at San Deigo, and developed in collaboration with R. Paul Gorman of Allied-Signal Aerospace Technology Center. [[1]](#References) [[2]](#References). Each sample contains 60 input features that represent the energy of the returned signal. The features represent rising frequencies with higher frequencies returned later in time during the single chirp. Each sample is coded with the actual target as either "R" for rock or "M" for simulated mine.

---

## Grid Search Optimization of the Hyperparameters

### Neural Network Hyperparameter Grid Search

The hyperparameters of both the neural net and random forest were optimized with a grid search over a discrete space. The neural net hyperparameter optimization searched over a discrete set of increasing number of nodes in the first and second hidden layers, a discrete set of batch sizes, and two different training function (Adam and RMSProp). All search iterations included several fixed hyperparameters to include a dropout layer (alpha = 0.1) between each layer to increase generalization, an initial learning rate of 0.001 with a learning rate reduction of 50% that occurred every 30 epochs, and a maximum training time of 300 epochs. Additionally, the input data was scaled with the standard scaler (maps each feature to a mean of 0 and standard deviation of 1) to increase training efficiency. For all cases, the same 60% of the data was chosen at random to be the test set, 20% was used as a validation set for early stopping with 5 epochs patience, and 20% of the data was used as a holdout test set.

### Random Forest Hyperparameter Grid Search

The random forest model's hyperparameters were also optimized in a grid search. Similarly to the neural net, a random sample of 80% of the data was used for training and 20% of teh data was withheld as a holdout test set. Initially, a random forest was built on all the training data and the out-of-bag error was used to assess each feature's (frequency's) importance. To increase training efficiency, the grid search and full model was built only using the top 50% most important features. Using the top predictive features and the training data, a grid search was conducted over a discrete set of minimum leaf sizes. Then for each grid search, the out-of-bag error was calculated on using only a subset of all the trees starting from 1 to the maximum number of trees (500). The final model was created using the minimum leaf size and number of trees that resulted in the lowest out-of-bag error.

---
## Optimized Models

### Optimized Neural Net Model

The grid search yielded the following architecture for the fully-connected feed-forward neural network consisting of two hidden layers.

| Number of Nodes in 1st Hidden Layer | Number of Nodes in 2nd Hidden Layer| Batch Size | Training Function |
| --- | --- | ---| ---|
| 20 | 15 | 4 | adam |

### Optimized Random Forest Model

The grid search yielded the following architecture for the random forest model.

| Number of Trees | Minimum Leaf Size |
| --- | --- |
| 240 | 1 |

---

## Results

### Neural Network

The final neural network was able to correctly predict 83% of the test set samples correctly. This accuracy is comparable across the training, validation, and test sets indicating the model was not over or under trained as seen in the below confusion charts.

![NN_Confusion_chart](/Images/confusion_chart_NN.png)

Additionally, the ROC curve shows strong classification abilities of the model across a wide range of classification thresholds. This is also evident from the neural net model's AUC which was over 0.92.

![NN_ROC_chart](/Images/ROC_chart_NN.png)

### Random Forest

The random forest model achieved similar results compared to the neural net model. The final model achieved an accuracy of 83% on the test set (however, it did achieve a 100% accuracy on the training set). This is shown in the confusion chart below.

![NN_Confusion_chart](/Images/confusion_chart_RF.png)

Similar to the neural net, the random forest's ROC curve showed strong classification abilities of the model across a wide range of classification thresholds. Agin, this was also evident from the neural net model's AUC which was over 0.91.

![NN_ROC_chart](/Images/ROC_chart_RF.png)

---

## MATLAB Libraries

This simulation was built using MATLAB 2020a with the following libraries:

* Statistics Toolbox
* Deep Learning Toolbox
* Optimization Toolbox

---

## References

[1]  Sejnowski, Terry and R. Paul Gorman, "Connectionist Bench (Sonar, Mines vs. Rocks) Data Set." UCI Machine Learning Repository, Center for Machine Learning and Intelligent Systems [https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)]

[2] Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89. [http://rexa.info/paper/7257d06678a052c7cb6f1d08d8eda2f5ac07f74a]
