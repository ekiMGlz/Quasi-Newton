function [alpha, gnew] = lineSearch(f, xk, dk, gk)
% lineSearch Find an alpha such that xk + alpha*dk satisfies both Wolfe conditions for f
% Throws an error if the found alpha does not satisfy both Wolfe conditions.
%   Input:
%       f: Function handle
%       xk: Initial point
%       dk: Descent direction
%       gk: Gradient of f at xk
%   Output:
%       alpha: Real coefficient such that xk + alpha*dk satisfies both Wolfe conditions
%       gnew: Gradient of f at xk + alpha*dk

    % Some algorithm parameters and constant values
    c1 = 1e-4;
    c2 = 0.99;
    maxIter = 50;
    i = 1;

    alpha_0 = 0;
    alpha_max = 10;

    gTd = dot(gk, dk);
    fk = f(xk);
    phi_0 = fk;

    while i <= maxIter
        % Pick an alpha in (alpha_0, alpha_max) and calculate some values
        alpha = (alpha_0+alpha_max)*0.5;
        phi_i = f(xk + alpha*dk);
        gnew = grad(f, xk + alpha*dk);
        d_phi = dot(gnew, dk);
        
        % Test Wolfe 1
        if ( phi_i > fk + alpha*c1*gTd ) || ( phi_i >= phi_0 )
            % If Wolfe 1 failed, search for an alpha with zoom(alpha_0, alpha)
            [alpha, gnew] = zoom(alpha_0, alpha, f, xk, dk, fk, gTd);
            break
        % Test Wolfe 2
        elseif (abs(d_phi) <= -c2*gTd)
            % Both conditions are met
            break
        elseif d_phi >= 0
            % Wolfe 2 failed. Search for an alpha with zoom(alpha, alpha_0)
            [alpha, gnew] = zoom(alpha, alpha_0, f, xk, dk, fk, gTd);
            break
        else
            % Wolfe 2 failed. Search for an alpha with (alpha, alpha_max)
            alpha_0 = alpha;
            phi_0 = phi_i;
            i = i + 1;
        end
    end

    % Check if Wolfe 2 was met
    assert(abs(dot(gnew, dk)) <= -c2*gTd, "No se satisface W2")
end

function [alpha, gnew] = zoom(a_low, a_high, f, xk, dk, fk, gTd)
    
    % Some algorithm parameters and initial values
    c1 = 1e-4;
    c2 = 0.99;
    
    maxIter = 50;
    i = 1;

    phi_low = f(xk + a_low*dk);

    while i <= maxIter
        % Pick an alpha in (a_low, a_high) and calculate some values
        alpha = (a_low + a_high)*0.5;
        phi_i = f(xk + alpha*dk);
        gnew = grad(f, xk + alpha*dk);
        d_phi = dot(gnew, dk);

        % Check Wolfe 1
        if ( phi_i > fk + alpha*c1*gTd ) || ( phi_i >= phi_low )
            % Wolfe 1 failed, search un (alpha_low, alpha)
            a_high = alpha;
            i = i + 1;
        % Check Wolfe 2
        elseif (abs(d_phi) <= -c2*gTd)
            % Both conditions met
            break
        else
            % Wolfe 2 failed, search in (alpha, alpha_high)
            if (d_phi*(a_high - a_low) >= 0)
                a_high = a_low;
            end
            a_low = alpha;
            phi_low = phi_i;
            i = i + 1;
        end
    end

    % Check if Wolfe 2 was met
     assert(abs(d_phi) <= -c2*gTd, "No se satisface W2")
end