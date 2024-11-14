function [M,G,MT] = Gradient_renewal(type,m,grad,accumu_grad,Mt,lr,epoch)
    switch type
        case 1
            if epoch < 99
                M = m - grad * lr*(0.99^epoch);
            else
                M = m - grad * lr*(0.99^100);
            end
            G = 0;
            MT=0;
        case 2 % AdaGrad
            gg = grad.*grad;
            G = gg + accumu_grad;
            Theta = -lr*grad./sqrt(G + 0.00001);
            M = m + Theta;
            MT=0;
        case 3 % RMSprop
            G = 0.9*accumu_grad + 0.1*grad.*grad;
            Theta = -lr*grad./sqrt(G + 0.00001);
            M = m + Theta;
            MT=0;
        case 4 % Momentum
            G = 0.9*accumu_grad - lr*grad;
            M = m + G;
            MT=0;
        case 5 % AdaM
            MT = (0.9*Mt + 0.1*grad)/(1-0.9^epoch);
            G = (0.99*accumu_grad + 0.01*grad.*grad)/(1-0.99^epoch);
            Theta = -lr*MT./sqrt(G+0.00001);
            M = m + Theta;
        case 6 % AdaM_2
            G = accumu_grad * 0.999 + 0.001 * grad .* grad;
            MT = 0.9 * Mt + 0.1 * grad;
            alpha = lr * sqrt(1 - 0.999^epoch) / (1 - 0.9^epoch);
            M = m - alpha * MT ./ (sqrt(G) + 1e-8);
        case 7 % AdaM_3
            G = accumu_grad * 0.99 + 0.01 * grad .* grad;
            MT = 0.9 * Mt + 0.1 * grad;
            alpha = lr * sqrt(1 - 0.99^epoch) / (1 - 0.9^epoch);
            M = m - alpha * MT ./ (sqrt(G) + 1e-8);
        case 8 % Nothing
            G = 0;
            MT = 0;
            M = m - lr*grad;
    end
end

