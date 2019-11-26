function [xf, iter] = TRSR1(f, x0, maxIter, tol)
% TRSR1 - Find a local minima of f using the trust region method with 
% Symmetric Rank 1 Updates for the  approximated Hessian
%   Input:
%       f: Function handle
%       x0: Initial point
%       maxIter: Maximum number of iterations
%       tol: Tolerance for the inf norm of the gradient
%   Output:
%       xf: Final approximation of x*, the local minima of f
%       iter: Number of iterations to reach xf

    % Initialize some parameters used in the alg.
    eta = 1e-1;
    delta_max = 1.5;
    delta = 1;
    n = length(x0);
    r = 1e-6;
    
    % Obtain the first values of x_k, g_k, B_k and H_k
    iter = 0;
    
    if iscolumn(x0)
        x_k = x0;
    else
        x_k = x0';
    end
    
    g_k = grad(f, x_k);
    H_k = speye(n);
    B_k = speye(n);
    
    while norm(g_k, inf) > tol && iter < maxIter
        
        % If s_k is not a descent direction, use pCauchy
        s_k = -H_k*g_k;
        norm_p = norm(s_k);
        if dot(s_k, g_k) < 0
            if  norm_p > delta
                s_k = ( delta/norm_p )* s_k;
            end
        else
            s_k = pCauchy(B_k, g_k, delta);
        end
        
        % Calculate the quality of the approximation
        quality = ( f(x_k + s_k) - f(x_k) ) ...
                  / ( dot(g_k, s_k) + 0.5 * s_k'*B_k*s_k);
        
        % Adjust the trust region radius based on the quality
        if quality < 0.1
            delta = 0.5 * delta;
        elseif quality > 0.75 &&  norm(s_k) > 0.8 * delta    
            delta = min(delta_max, 2*delta);
        end
        
        % If the quality is above the minimum quality, then advance towards
        % p_k, and calculate the new gradient
        if quality > eta
            
            x_k = x_k + s_k;
            g_new = grad(f, x_k);
            
            % If the secant equation is met, then update H_k and B_k
            gamma = (g_new - g_k);
            gamma_shift = gamma - B_k*s_k;
            dot_gammashift_s = dot(gamma_shift, s_k);
            if abs(dot_gammashift_s) >= r * norm(s_k) * norm(gamma_shift)

                B_k = B_k + (1/dot_gammashift_s) * (gamma_shift*gamma_shift');
                s_shift = s_k - H_k*gamma;
                H_k = H_k + (1/dot(s_shift, gamma)) * (s_shift)*(s_shift)';
                
            end
            
            g_k = g_new;
            iter = iter + 1;
        end
        
    end

    xf = x_k;
    
    
end

