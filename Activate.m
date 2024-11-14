function [X,Type] = Activate(x,type)
    switch type
        case 1 %RELU
            X = max(0,x);
        case 2 %LeakyRELU
            X = max(0.01*x,x);
        case 3 %Logistic
            X = 1./(1+exp(-x));
        case 4 %tanh
            X = tanh(x);
    end
    Type=type;
end