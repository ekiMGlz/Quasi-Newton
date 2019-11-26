function [y] = dixmaan(x)
% DIXMAAAND - Solve Dixmaan's function for a given x with the following
% parameters alpha = 1, beta = gamma = delta = 0.26, k1 = k2 = k3 = k4 = 0
%   Input:
%       x: Function arguments
%   Output:
%       y: Dixmaand function evaluated at x
%
    % intialize parameters
    alpha = 1;
    beta = 0.26;
    gamma = 0.26;
    delta = 0.26;
    
    k1 = 0;
    k2 = 0;
    k3 = 0;
    k4 = 0;
    
    % get vector size
    n = length(x);
    m = floor(n/3);
    
    % initialize result as 1
    y = 1;
    
    for i=1:n
        i_n = i/n;
        
        y = y + alpha*x(i)^2*(i_n)^k1;
        
        if i <= n-1
            y = y + beta*x(i)^2*(x(i+1)+x(i+1)^2)^2*(i_n)^k2;
        end
    end
    
    for i=1:2*m
        i_n = i/n;
        
        y = y + gamma*x(i)^2*x(i+m)^4*i_n^k3;
        
        if i <= m
            y = y + delta*x(i)*x(i+2*m)*(i_n)^k4;
        end
    end
end