function [W] = Initialization(row_num,col_num,type)
    switch type
        case 1 % 随机初始化
            W = rand(row_num,col_num,'double');
        case 2 % Gaussian初始化
            W = sqrt(2/(row_num+col_num))*randn(row_num,col_num,'double');
        case 3 % Xavier-logistic初始化
            r = sqrt(6/(row_num+col_num));
            W = unifrnd(-r,r,[row_num,col_num]);
        case 4 % Xavier-tanh初始化
            r = 4*sqrt(6/(row_num+col_num));
            W = unifrnd(-r,r,[row_num,col_num]);
        case 5 % Xavier-tanh初始化（忽略下层节点数）
            r = 4*sqrt(6/(col_num));
            W = unifrnd(-r,r,[row_num,col_num]);
        case 6 % Xavier-tanh初始化，考虑expan
            r = 4*sqrt(3/(4*col_num*(1+((1-0.3).^2)/(3*1*0.3))));
            W = unifrnd(-r,r,[row_num,col_num]);
        case 7 % Xavier-tanh初始化，考虑expan2
            r = 4*sqrt(3/( col_num * (1+ 9*(1+0.3).^4 / (1^2 + 1*0.3 + 0.3^2).^2) ));
            W = unifrnd(-r,r,[row_num,col_num]);
    end
end
