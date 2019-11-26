function [xf, iter] = lineBGFS(f, x0, maxiter, tol)
% lineBGFS - Find a local minima of f using line search with 
% BGFS updates for the  approximated Hessian
%   Input:
%       f: Function handle
%       x0: Initial point
%       maxIter: Maximum number of iterations
%       tol: Tolerance for the inf norm of the gradient
%   Output:
%       xf: Final approximation of x*, the local minima of f
%       iter: Number of iterations to reach xf

    % Set the initial values for xf, gk, and Hk
    iter = 0;
    xf = x0;
    gk = grad(f, xf);
    n = length(gk);
    Hk = speye(n);


    while norm(gk, 'inf') > tol && iter < maxiter
        % Define the descent direction
        dk = -Hk*gk;

        % Find an alpha for dk using lineSearch
        [alpha, gnew] = lineSearch(f, xf, dk, gk);

        % Update Hk
        s = alpha*dk;
        gamma = gnew - gk;
        rho = 1/dot(gamma, s);

        HkGamma = Hk*gamma*rho;
        sHkGammaT = s*HkGamma';
        Hk = Hk -(sHkGammaT + sHkGammaT') + (rho*(dot(gamma, HkGamma) + 1)*s)*s';

        % Advance towards xk + alpha*dk
        xf = xf + s;
        gk = gnew;
        
        iter  = iter + 1;
    end
end
