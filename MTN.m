clc;clear;

t0=tic; % 对整个流程进行计时

% 设置学习率和训练轮数
testNum=1; % 重复实验的次数
numEpochs = 1000; % 每次实验的十折交叉中的每一折的最大迭代数
lr = 0.001; % 初始学习率
k=10;  % 十折交叉
Acti_type = 4; % 激活函数类型，1RELU，2LeakyRELU，3Logistic，4tanh
Power=2; % 设置MTN的最高展开次
class_num=2;%分类数目
shift_num=0.3;

%%%%%%%%%%%%%%%%%%% 数据导入 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load feature_selected
format long

total_num=size(feature,2);
len_data=size(feature,1);

feature = gpuArray(feature);

train_num=round(total_num*(k-1)/k); %训练集的大小（九折的数量）

% 设置网络结构
feature_size=size(feature,1); %特征向量的维度

expan_size=1;
for i=1:Power
    expan_size=expan_size + prod(feature_size:(feature_size+i-1))/prod(1:i); %多项式展开后的维度
end

hidden_size = feature_size;
output_size = class_num;

%%%%%%%%%%%%%%%%%%%%one-hot编码%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
onehot=zeros(train_num,class_num);%设置训练数据集对应标签one-hot编码需要的空矩阵大小

identity_matrix=[1,0;0,1];
for i=1:total_num
    switch Labels(i)
        case 1
            onehot(i,:)=identity_matrix(1,:);
        case 2
            onehot(i,:)=identity_matrix(2,:);
    end
end%根据分类类型不同增加或减少case情况
onehot=onehot';

figure; % 创建一个新的图形窗口
h1 = animatedline('Color','b'); % 创建一个动画线条用于测试集准确率
h2 = animatedline('Color','r'); % 创建一个动画线条用于测试集准确率
xlabel('Epoch');
ylabel('Accuracy');
xlim([0 numEpochs]); % 设置横轴的范围为0到最大迭代数
ylim([0 1]); % 设置纵轴的范围为0到1（因为准确率的范围是0到1）

Total_acc=0; %总准确率计数
Total_conf_acc=zeros(2,2); %总混淆矩阵计数
for testingNum=1:testNum
    accuracy = zeros(1, k);
    confusionMatrix=zeros(class_num,class_num,k);

    % cv实现的交叉验证会尽可能的保证每折中的各类别样本数相等，并随机打乱数据集
    cv = cvpartition(Labels', 'kfold', k); 
    for i=1:k
        res_Train_Data=feature(:,cv.training(i));
        res_Test_Data=feature(:,cv.test(i));

        % 训练集归一化
        Tmin=min(res_Train_Data,[],2);
        Tmax=max(res_Train_Data,[],2);
        res_Train_Data=(res_Train_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;

        % 测试集归一化
        res_Test_Data=(res_Test_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;

        %%%%%%%%%%%%%%%%%%%%%%% 多项式展开 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        TrainData=gpuArray(Taylor_expan(res_Train_Data,Power));
        TestData=gpuArray(Taylor_expan(res_Test_Data,Power));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        TrainLabels = gpuArray(onehot(:,cv.training(i)));
        TrainLabels_valid=gpuArray(Labels(:,cv.training(i)));
        TestLabels=gpuArray(Labels(:,cv.test(i)));
    
    
        TrainData_valid=TrainData; % 用于训练集验证
        res_Train_Data_valid=res_Train_Data;
        
        % 初始化权重矩阵
        W1 = gpuArray(Initialization(hidden_size, expan_size,7)); % 1随机初始化，2Gaussian初始化，3Xavier-logistic初始化，4Xavier-tanh初始化
        W2 = gpuArray(Initialization(output_size, hidden_size,2));

        train_accurancy=0;
        test_accurancy=0;
        
        epsilon=0.00001;  % 防除零
        %初始化梯度的均值(一阶矩):
        GW2=zeros(size(W2)); 
        GW1=zeros(size(W1));
        %初始化梯度的未减去均值的方差(二阶矩):
        MW2=zeros(size(W2)); 
        MW1=zeros(size(W1));
        
        % 训练网络
        for epoch = 1:numEpochs
            % 在训练集上进行训练
            % 正向传播
            z1 = W1*TrainData;
            hidden = Activate(z1,Acti_type); %Activate为激活函数
            z2 = W2*(hidden);
            output = softmax(z2);
            
            % 计算输出层误差
            d_softmax = output - TrainLabels;
            %W2和hidden层的导数：
            d_W2 = d_softmax * (hidden)'/train_num;
            d_hidden = W2' * d_softmax; 
            d_z1 = d_hidden .* Activate_grad(z1,Acti_type);
            %计算W1和batchData的导数：
            d_W1 = d_z1 * TrainData'/train_num;
    
            % 更新参数
            [W1,GW1,MW1]=Gradient_renewal(7,W1,d_W1,GW1,MW1,lr,epoch); % 使用学习率衰减和梯度方向优化：1可调衰减，2AdaGrad，
            [W2,GW2,MW2]=Gradient_renewal(7,W2,d_W2,GW2,MW2,lr,epoch); %     3RMSprop，4Momentum，5AdaM，6AdaM_2，7AdaM_3
            
            % 在训练集上进行测试
            z1 = W1*TrainData_valid;
            hidden = Activate(z1,Acti_type);
            z2 = W2*(hidden);
            output = softmax(z2);
            % 计算准确率和损失
            [~, predictedLabels] = max(output);
            train_accurancy = mean(predictedLabels == TrainLabels_valid);
            %addpoints(h1, epoch, train_accurancy); % 在图形上添加训练集准确率点
            
            pred=output(1,:);
            Labels_binary=TrainLabels_valid*(-1)+2;
            % 计算交叉熵损失
            cross_entropy_loss_train = -mean(Labels_binary .* log(pred) + (1 - Labels_binary) .* log(1 - pred));
            
            % 在测试集上进行测试
            z1 = W1*TestData;
            hidden = Activate(z1,Acti_type);
            z2 = W2*(hidden);
            output = softmax(z2);
            % 计算准确率和损失
            [~, predictedLabels] = max(output);
            test_accurancy = mean(predictedLabels == TestLabels);
            
            pred=output(1,:);
            Labels_binary=TestLabels*(-1)+2;
            % 计算交叉熵损失
            cross_entropy_loss_test = -mean(Labels_binary .* log(pred) + (1 - Labels_binary) .* log(1 - pred));

            fprintf('No.%d, K=%d, Epoch=%d, Train accurancy:%.4f, Test accurancy:%.4f, Train loss:%.4f, Test loss:%.4f\n',testingNum,i,epoch,train_accurancy*100,test_accurancy*100,cross_entropy_loss_train,cross_entropy_loss_test);
            
            if i == 1
                addpoints(h1, epoch, train_accurancy); % 在图形上添加测试集准确率点
                addpoints(h2, epoch, test_accurancy); % 在图形上添加测试集准确率点
            end
            drawnow; % 更新图形窗口
        end
        accuracy(i)=test_accurancy;
        confusionMatrix(:,:,i) = confusionmat(TestLabels, predictedLabels);
    end
    Total_acc=Total_acc+mean(accuracy);
    Total_conf_acc=Total_conf_acc+mean(confusionMatrix,3);

    %%%%%%%%%%%%%%%%% 记录和计算第一次结束后的结果 %%%%%%%%%%%%%%%%%%%%%%%%%%
%     if testingNum == 1
%         accur=zeros(10,1);
%         sensi=zeros(10,1);
%         speci=zeros(10,1);
%         for p=1:k
%             TP = confusionMatrix(1,1,p);
%             FP = confusionMatrix(1,2,p);
%             FN = confusionMatrix(2,1,p);
%             TN = confusionMatrix(2,2,p);
%             accur(p) = (TP + TN) / (TP + FP + FN + TN);
%             sensi(p) = TP / (TP + FN);
%             speci(p) = TN / (TN + FP);
%         end
%         results=[accur,sensi,speci];
%     end
%     keyboard;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
time=toc(t0);

Average_accuracy=Total_acc*100/testingNum;

Total_conf_acc=Total_conf_acc/testNum;

TP = Total_conf_acc(1,1);
FP = Total_conf_acc(1,2);
FN = Total_conf_acc(2,1);
TN = Total_conf_acc(2,2);

% 计算准确率、灵敏率和特异度
accuracy = (TP + TN) / (TP + FP + FN + TN);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

%输出结果
fprintf('\n\t Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, Average time: %.4f\n', accuracy, sensitivity, specificity,time/(testNum*k));