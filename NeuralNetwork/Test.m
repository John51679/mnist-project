T = readtable('mnist_train.csv');
D = table2array(T);

trainY = D(:,1);
trainX = D(:,2:size(D,2));

meanVector = mean(trainX);
stdVector = std(trainX);

for j = 1:size(trainX,2)
    for i = 1:size(trainX,1)
        if stdVector(j) ~= 0
            trainX(i,j) = (trainX(i,j) - meanVector(j))/stdVector(j);
        else
            break;
        end
    end
end

mean(trainX,'all')
std(trainX(:))