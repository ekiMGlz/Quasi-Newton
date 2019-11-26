function [xf, iter] = limBGFS_cyclic(f, x0, maxiter, tol, m)
% lineBGFS - Find a local minima of f using line search with 
% limited memory BGFS updates for the  approximated Hessian
%   Input:
%       f: Function handle
%       x0: Initial point
%       maxIter: Maximum number of iterations
%       tol: Tolerance for the inf norm of the gradient
%       m: Number of previous steps used for the BGFS updates. Uses the last m steps
%   Output:
%       xf: Final approximation of x*, the local minima of f
%       iter: Number of iterations to reach xf
    
    % Set initial values for xf, gk and dk
    iter = 0;
    xf = x0;
    gk = grad(f, xf);
    n = length(gk);
    dk = -gk;

    % Initialize the memory arrays (saves info on previous m steps)
    S = zeros(n, m);
    G = S;
    % Index of the latest update in S and G
    startAt = 1;
    
    while norm(gk, 'inf') > tol && iter < maxiter
        
        % Find an alpha for dk using lineSearch
        [alpha, gnew] = lineSearch(f, xf, dk, gk);
        
        % Update S and G at startAt
        S(:, startAt) = alpha*dk;
        G(:, startAt) = gnew - gk;
        
        
        %Calculate a new dk
        iter  = iter + 1;
        if iter < m
            dk = -calcHg(S, G, gnew, startAt, iter);
        else
            dk = -calcHg(S, G, gnew, startAt, m);
        end
        
        % Advance towards xk + alpha*dk
        xf = xf + S(:, startAt);
        gk = gnew;

        % Update the latest index of S and G
        startAt = mod(startAt, m) + 1;

    end
end

function [q] = calcHg(S, G, gnew, startAt, len)
    % Initialize rhos and alphas
    q = gnew;
    m = len;
    rhos = zeros(m, 1);
    alphas = zeros(m, 1);
    
    % Parallel index to iterate through S and G starting at (startAt) and ending at (startAt - 1), increasing the index by one each step
    j = startAt;
    for i = 1:m
        
        rhos(i) = 1 / dot(S(:, j), G(:, j));
        alphas(i) = dot(S(:, j), q) * rhos(i);
        
        q = q - alphas(i)*G(:, j);
        j = mod(j, m) + 1;

    end

    % Calculate delta
    delta = 1 / (rhos(1) * dot(G(:, startAt), G(:, startAt)));
    
    q = delta * q;
    
    % Parallel index to iterate through S and G starting at (startAt - 1) and ending at (startAt), decreasing the index by 1 each step
    j = j - 1;
    if j == 0
        j = m;
    end

    for i = m:-1:1
        beta = dot(G(:, j), q)*rhos(i);
        q = q + (alphas(i) - beta) * S(:, j);
        
        j = j - 1;
        if j == 0
            j = m;
        end
    end

end