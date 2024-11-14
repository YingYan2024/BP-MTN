function [X] = Activate_grad(x,type)
    switch type
        case 1 %RELU
            X=ones(size(x));
            X(x<=0)=0;
        case 2 %LeakyRELU
            X=ones(size(x));
            X(x<=0)=0.01;
        case 3 %Logistic
            tem = 1./(1+exp(-x));
            X = tem.*(1-tem);
        case 4 %tanh
            X = sech(x).^2;
    end
end

