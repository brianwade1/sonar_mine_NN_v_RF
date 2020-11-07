%% Documentation
%Program written by: Brian Wade
%Date 4 Apr 2020

%% Description and setup
%Input data from: 
% UIC Machine Learning Repository, Connectionist Bench 
%(Sonar, Mines vs. Rocks) Data Set, 
% https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

%% Program setup
clear
clc
close all
start_time=tic; %Timer
rng(123) % set random seed

%% User inputs
val_perc = 0.2;
test_perc = 0.2;

data_folder = 'Data';
image_folder = 'Images';

input_filename = 'sonar_coded.csv';
data_end_col = 60;
fit_col = 61; 

%Hyperparameters!
FC_1_set = [5, 10, 15, 20, 30];
FC_2_set = [3, 5, 10, 15];
batch_size_set = [4, 8, 16, 32, 64];
train_fun_set = {'adam', 'rmsprop'};

% Set Hyperparameters
dropoutrate = 0.1;
maxEpochs = 200;
InitialLearnRate = 0.001;
LearnRateDropFactor = 0.5;
GradientThreshold = 1;
ValidationPatience = 5;

%% Read input DoE
input = csvread(fullfile(data_folder,input_filename));
data = input(:,1:data_end_col);
target = input(:,fit_col);

nruns = size(data,1);

%% Prepare data for modeling

% Data is ordered so randomize it.
random_row_ind = randperm(nruns);
data = data(random_row_ind,:);
t = target(random_row_ind,:);

% Transpose becasue MATLAB likes features as rows for neural nets
X = data';
t = categorical(t)';

% Scale input data
[x_scaled,PX] = mapstd(X);

% Divide data into train, validation, and test sets
[trainInd,valInd,testInd] = dividerand(size(X,2), ...
    (1 - val_perc - test_perc), val_perc, test_perc);

x_train_scaled = x_scaled(:,trainInd);
t_train = t(:,trainInd);
x_val_scaled = x_scaled(:,valInd);
t_val = t(:,valInd);
x_test_scaled = x_scaled(:,testInd);
t_test = t(:,testInd);

%% Prepare data for training
nFeatures = size(x_train_scaled,1);
numClasses = size(categories(t_train),1);

nSamples = size(x_train_scaled,2);
nValSamples = size(x_val_scaled,2);
nTestSamples = size(x_test_scaled,2);

Xtrain = reshape(x_train_scaled, [1,1,nFeatures,nSamples]);
Xval = reshape(x_val_scaled, [1,1,nFeatures,nValSamples]);
Xtest = reshape(x_test_scaled, [1,1,nFeatures,nTestSamples]);

%% Perform grid search over hyperparameters
% Initialize the storage of AUC, parameters, and NN models
auc_all = zeros(length(FC_1_set) + length(FC_2_set) +...
    length(batch_size_set) + length(train_fun_set), 1);
parameter_set = zeros(size(auc_all,1), 4);
net_set = cell(size(auc_all,1));

disp('Starting Hyperparameter Search')
num = 1;

for i = 1:length(FC_1_set)
    FC_1 = FC_1_set(i);
    for j = 1:length(FC_2_set)
        FC_2 = FC_2_set(j);
        for k = 1:length(batch_size_set)
            batch_size = batch_size_set(k);
            for l = 1:length(train_fun_set)
                train_fun = train_fun_set{l};
                               
                layers = [ ...
                    imageInputLayer([1 1 nFeatures],'Name','Input')
                    fullyConnectedLayer(FC_1,'Name','FC1')
                    dropoutLayer(dropoutrate)
                    reluLayer
                    fullyConnectedLayer(FC_2,'Name','FC2')
                    dropoutLayer(dropoutrate)
                    reluLayer
                    fullyConnectedLayer(numClasses)
                    softmaxLayer
                    classificationLayer];
                
                validationFrequency = floor(size(t_train,2)/batch_size);
                LearnRateDropPeriod = floor(maxEpochs/10);
                
                options = trainingOptions(train_fun, ...
                    'MaxEpochs',maxEpochs, ...
                    'MiniBatchSize',batch_size, ...
                    'InitialLearnRate',InitialLearnRate, ...
                    'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',LearnRateDropPeriod, ...
                    'LearnRateDropFactor',LearnRateDropFactor, ...
                    'GradientThreshold',GradientThreshold, ...
                    'Shuffle','every-epoch', ...
                    'ValidationData',{Xval,t_val}, ...
                    'ValidationFrequency',validationFrequency, ...
                    'ValidationPatience',ValidationPatience, ...
                    'Verbose',0,...
                    'VerboseFrequency',validationFrequency);
                
                
                [net,info] = trainNetwork(Xtrain,t_train,layers,options);
                
                [~, scores_val] = classify(net,Xval);
                [~,~,~,auc] = perfcurve(t_val,scores_val(:,2),1);
                
                auc_all(num) = auc;
                parameter_set(num, :) = [FC_1, FC_2, batch_size, l];
                net_set{num} = net;
                
                num = num + 1;
                
                disp(['Completed iteration with FC_1 = ', num2str(FC_1),...
                ' and FC_2 = ', num2str(FC_2), ' and batch size = ', ...
                num2str(batch_size), ' and training function = ',...
                train_fun, ' || AUC of validation set = ', num2str(auc)])
                
            end
        end
    end
end

% find best model
[auc_max, ind_max] = max(auc_all);
net = net_set{ind_max};

% Get hyperparameters of best model
best_FC_1 = parameter_set(ind_max, 1);
best_FC_2 = parameter_set(ind_max, 2);
best_batch_size = parameter_set(ind_max, 3);
best_train_fun_num = parameter_set(ind_max, 4);
best_train_fun = train_fun_set{best_train_fun_num};

% Save best hyperparameters
fid = fopen('Models/NN_model_parameters.txt','wt');
fprintf(fid, ['Number of fully connected nodes, layer 1 = ',...
    num2str(best_FC_1), '\n']);
fprintf(fid, ['Number of fully connected nodes, layer 2 = ',...
    num2str(best_FC_2), '\n']);
fprintf(fid, ['Batch Size = ', num2str(best_batch_size), '\n']);
fprintf(fid, ['Training Function = ', best_train_fun, '\n']);
fclose(fid);

% Save best NN model
save('Models/NN_model','net')

% Predict results and check accuracy
[y_hat_train, scores_train] = classify(net,Xtrain);
[y_hat_val, scores_val] = classify(net,Xval);
[y_hat_test, scores_test] = classify(net,Xtest);

train_accuracy = sum(t_train == y_hat_train') / numel(t_train);
val_accuracy = sum(t_val == y_hat_val') / numel(t_val);
test_accuracy = sum(t_test == y_hat_test') / numel(t_test);

disp('')
disp('')
disp('********************************')
disp('***         NN Results       ***')
disp('********************************')
disp('')
disp('')
disp(['Accuracy of Training Set: ',num2str(train_accuracy)])
disp(['Accuracy of Validation Set: ',num2str(val_accuracy)])
disp(['Accuracy of Test Set: ',num2str(test_accuracy)])   
disp('')
disp('')

% Print confusion matrix and ROC curve
figure()
subplot(1,3,1)
confusionchart(t_train,y_hat_train);

subplot(1,3,2)
confusionchart(t_val,y_hat_val);

subplot(1,3,3)
confusionchart(t_test,y_hat_test);

% Save plots image to folder
image_file = 'confusion_chart_NN.png';
image_save_path = fullfile(image_folder,image_file);
saveas(gcf, image_save_path)

figure()
[xtest,ytest,~,auc] = perfcurve(t_test,scores_test(:,2),1);
plot(xtest,ytest)
xlabel('False positive rate')
ylabel('True positive rate')
text(0.5,0.25,strcat('AUC=',num2str(auc)),'EdgeColor','k')
title('ROC curve, predicted vs. actual rating')

image_file = 'ROC_chart_NN.png';
image_save_path = fullfile(image_folder,image_file);
saveas(gcf, image_save_path)


%% Finish program
%display total time to complete tasks
tElapsed = toc(start_time); 
hour=floor(tElapsed/3600);
tRemain = tElapsed - hour*3600;
minutes =floor(tRemain/60);
sec = tRemain - minutes*60;
 
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Program Complete!!!!!')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')

disp(['Time to complete: ',num2str(hour),' hours, ',num2str(minutes),...
    ' minutes, ',num2str(sec),' seconds'])


