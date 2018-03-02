d=1;
i=3;
percent = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
d1 = [7,10,15,17,20];
%get the train data and train label
train_data = train(:,1:d);
train_label = train(:,d+1);
test_data = test(:,1:d);
test_label = test(:,d+1);
% train error and test error
% [w1_train,w2_train,b1_train,b2_train]=...
%     sigmoid_train(train_data,train_label,W1,b1,W2,b2,15);
[w1_train,w2_train,b1_train,b2_train]=...
    ReLU_train(train_data,train_label,W1,b1,W2,b2,d1(i));

train_error(i)=...
    neural_networks_sqerror...
    (train_data,train_label,w1_train,b1_train,w2_train,b2_train);
test_error(i)=...
    neural_networks_sqerror...
    (test_data,test_label,w1_train,b1_train,w2_train,b2_train);
% %% Cross validation
% sum_test_error = 0;
% for fold=1:5
%     cvtrain_data = cv_sub_train{fold}(:,1:d);
%     cvtrain_label = cv_sub_train{fold}(:,d+1);
%     cvtest_data = cv_data_all{fold}(:,1:d);
%     cvtest_label = cv_data_all{fold}(:,d+1);
%     [w1_cvtrain,w2_cvtrain,b1_cvtrain,b2_cvtrain]=...
%         ReLU_train(cvtrain_data,cvtrain_label,W1,b1,W2,b2,d1(i));
%     sum_test_error = sum_test_error+...
%         neural_networks_sqerror...
%         (cvtest_data,cvtest_label,w1_cvtrain,b1_cvtrain,w2_cvtrain,b2_cvtrain);
% end
% cv_error(i) = sum_test_error/5;

% Plot learned function
x = linspace(0,1,100)';
z = w1_train*x'+repmat(b1_train,1,100);% d1*m
%a1 = sigmoid(z); % d1*m
a1 = ReLU(z); % d1*m
f = w2_train * a1+repmat(b2_train,1,100);% 1*m


% Learning curve
for fold=1:10
    subtrain_data = subs{fold}(:,1:d);
    subtrain_label = subs{fold}(:,d+1);
    [w1_subtrain,w2_subtrain,b1_subtrain,b2_subtrain]=...
        sigmoid_train(subtrain_data,subtrain_label,W1,b1,W2,b2,15);
%     [w1_subtrain,w2_subtrain,b1_subtrain,b2_subtrain]=...
%         ReLU_train(subtrain_data,subtrain_label,W1,b1,W2,b2,15);
    sub_train_error(fold)=neural_networks_sqerror...
        (subtrain_data,subtrain_label,w1_subtrain,b1_subtrain,w2_subtrain,b2_subtrain);
    sub_test_error(fold)=neural_networks_sqerror...
        (test_data,test_label,w1_subtrain,b1_subtrain,w2_subtrain,b2_subtrain);
end
