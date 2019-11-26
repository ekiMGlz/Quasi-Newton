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
        if iter <= m
            dk = -calcHg(S(:, 1:iter), G(:, 1:iter), gnew, startAt);
        else
            dk = -calcHg(S, G, gnew, startAt);
        end
        
        % Advance towards xk + alpha*dk
        xf = xf + S(:, startAt);
        gk = gnew;

        % Update the latest index of S and G
        startAt = mod(startAt, m) + 1;

    end
end

function [q] = calcHg(S, G, gnew, startAt)
    % Initialize rhos and alphas
    q = gnew;
    m = length(S(1, :));
    rhos = 1 ./ dot(S, G, 1);
    alphas = zeros(m, 1);
    
    % Indices from startAt+1 to startAt increasing by one and looping around
    indices = mod((0:m-1) + startAt, m) + 1;
    for i = fliplr(indices)
        
        alphas(i) = dot(S(:, i), q) * rhos(i);
        q = q - alphas(i)*G(:, i);

    end

    % Calculate delta
    delta = 1 / (rhos(startAt) * dot(G(:, startAt), G(:, startAt)));
    q = delta * q;
    
    for i = indices
        beta = dot(G(:, i), q) * rhos(i);
        q = q + (alphas(i) - beta) * S(:, i);
    end

end