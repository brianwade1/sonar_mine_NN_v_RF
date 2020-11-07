%% Documentation
%Program written by: Brian Wade
%Date 4 Apr 2020

%% Description and setup
%Input data from: 
%1. UIC Machine Learning Repository, Connectionist Bench (Sonar, Mines vs. Rocks) Data Set, https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

%% Program setup
clear
clc
close all
start_time=tic; %Timer
rng(42) % set random seed


%% User inputs
val_per = 0.0;
test_per = 0.2;

data_folder = 'Data';
image_folder = 'Images';

input_filename = 'sonar_coded.csv';

X_end_col = 60;
fit_col = 61; 
want_parallel = false;
num_trees = 500;
surrogate = 'off';


%% Read input DoE
input = csvread(fullfile(data_folder,input_filename));

% Divide into input and target
x = input(:,1:X_end_col);
target = input(:,fit_col);

% Number of samples
nruns = size(x,1);

%% Prepare data for modeling
% Randomize data points because they are ordered by target
order = randperm(length(target));
x_random = x(order, :);
target_random = target(order);

% Divide into train, test set
[trainInd,valInd,testInd] = dividerand(nruns,1-test_per-val_per,val_per,test_per);

x_train= x_random(trainInd,:);
x_val= x_random(valInd,:);
x_test= x_random(testInd,:);

target_train = target_random(trainInd,:);
target_val = target_random(valInd,:);
target_test = target_random(testInd,:);

%% Important features
disp('********************************')
disp('Starting to feature evaluation')

forest_import = TreeBagger(num_trees,x_train,target_train,...
    'Method','classification',...
    'Surrogate',surrogate,...
    'OOBPredictorImportance','On',...
    'OOBPrediction','On');

figure
bar(forest_import.OOBPermutedPredictorDeltaMeanMargin)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')

pct_50 = prctile(forest_import.OOBPermutedPredictorDeltaMeanMargin,50);
pct_50_ind = find(forest_import.OOBPermutedPredictorDeltaMeanMargin > pct_50);


%% Hyperparameter search
min_leaf_size = [1, 2, 3, 5, 10, 15, 20, 25];

%Prepare for loop
oobError_set = cell(length(min_leaf_size),1);
forest_set = cell(length(min_leaf_size),1);
val_error_set=zeros(length(min_leaf_size),1);
auc_set = zeros(length(min_leaf_size),1);

for i = 1:length(min_leaf_size)
    
    
    disp('********************************')
    disp(['Starting to train RF with max number of splits = ',...
        num2str(min_leaf_size(i))])
    
    
    forest = TreeBagger(num_trees,...
        x_train(:,pct_50_ind),target_train,...
        'MinLeafSize',min_leaf_size(i),...
        'Method','classification',...
        'Surrogate',surrogate,...
        'OOBPrediction','On');
    
    [Y_fit,Scores_fit] = oobPredict(forest);
    [xtest,ytest,~,auc] = perfcurve(target_train,Scores_fit(:,2),1);
    
    auc_set(i) = auc;
    
    oobError_set{i}= oobError(forest,'Mode','cumulative');
    forest_set{i} = forest;

end


ClassError_set = reshape(cell2mat(oobError_set),...
    [num_trees numel(min_leaf_size)]);

[ClassError_best, minErrIdxLin] = min(ClassError_set(:));
[idx_Num_trees, idx_max_splits] = ind2sub(size(ClassError_set),...
    minErrIdxLin);

num_trees_best = idx_Num_trees;
min_leaf_size_best = min_leaf_size(idx_max_splits);


forest_best = TreeBagger(num_trees_best,...
    x_train(:,pct_50_ind),target_train,...
    'MinLeafSize',min_leaf_size_best,...
    'Method','classification',...
    'OOBPrediction','On');

% Save RF model
save('Models/RF_model','forest_best')

% Save best hyperparameters
fid = fopen('Models/RF_model_parameters.txt','wt');
fprintf(fid, ['Number of trees = ',...
    num2str(num_trees_best), '\n']);
fprintf(fid, ['Minimum leaf size = ',...
    num2str(min_leaf_size_best), '\n']);
fprintf(fid, ['Feature index = ', num2str(pct_50_ind), '\n']);
fclose(fid);

% Predict results and get accuracy
[y_train_RF,scores_train] = (predict(forest_best,x_train(:,pct_50_ind)));
[y_val_RF,scores_val] = (predict(forest_best,x_val(:,pct_50_ind)));
[y_test_RF, scores_test] = (predict(forest_best,x_test(:,pct_50_ind)));

avg_OOB = mean(forest_best.oobError./num_trees_best);

train_accuracy = sum(target_train == str2double(y_train_RF)) / numel(target_train);
test_accuracy = sum(target_test == str2double(y_test_RF)) / numel(target_test);

disp('********************************')
disp('***         RF Results       ***')
disp('********************************')
disp('')
disp('Best model:')
disp(['Min leaf size: ',num2str(min_leaf_size_best)])
disp(['Number of trees: ',num2str(num_trees_best)])
disp(['Average out-of-bag error: ',num2str(avg_OOB)])
disp('')
disp(['Accuracy of Training Set: ',num2str(train_accuracy)])
disp(['Accuracy of Test Set: ',num2str(test_accuracy)])   


% Print confusion matrix and ROC curve
figure()
subplot(1,2,1)
confusionchart(target_train,str2double(y_train_RF));

subplot(1,2,2)
confusionchart(target_test,str2double(y_test_RF));

image_file = 'confusion_chart_RF.png';
image_save_path = fullfile(image_folder,image_file);
saveas(gcf, image_save_path)

figure()
[xtest,ytest,~,auc] = perfcurve(target_test,scores_test(:,2),1);
plot(xtest,ytest)
xlabel('False positive rate')
ylabel('True positive rate')
text(0.5,0.25,strcat('AUC=',num2str(auc)),'EdgeColor','k')
title('ROC curve, predicted vs. actual rating')

image_file = 'ROC_chart_RF.png';
image_save_path = fullfile(image_folder,image_file);
saveas(gcf, image_save_path)


%% Finish program
%display total time to complete tasks
tElapsed = toc(start_time); 
hour=floor(tElapsed/3600);
tRemain = tElapsed - hour*3600;
min=floor(tRemain/60);
sec = tRemain - min*60;
 
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Program Complete!!!!!')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')

disp(['Time to complete: ',num2str(hour),' hours, ',num2str(min),' minutes, ',num2str(sec),' seconds'])


