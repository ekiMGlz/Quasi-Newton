
N = [200, 1000];
M = [1,3,5,17,29];

tol = 1e-5;
maxiter = 10000;

% Initialize results' vectors
results_bgfs = zeros(1,4);
results_sr = zeros(1,4);

results_lmbgfs = zeros(12,5);

% DixmaanD function
f = @(x) dixmaan(x);

%% Test lineBGFS and Symmetric Rank One Update for n = 200
x_0 = ones(1,200)*2;

% line BGFS
results_bgfs(1) = 200;

tic
[xf, iter] = lineBGFS(f, x_0, maxiter, tol);
results_bgfs(2) = toc; % time elapsed

results_bgfs(3) = iter;
results_bgfs(4) = norm(xf, 'inf');

% symmetric rank one
results_sr(1) = 200;

tic
[xf, iter] = TRSR1(f, x_0, maxiter, tol);
results_sr(2) = toc; % time elapsed

results_sr(3) = iter;
results_sr(4) = norm(xf, 'inf');

%% Test lineBGFS with limited memory for n = 200, 1000

% Test methods
k = 1;
for i=1:2
    n = N(i);
    
    x_0 = ones(1,n)*2;
    
    for j = 1:6
        m = M(j);
        
        results_lmbgfs(k,1) = n;
        results_lmbgfs(k,2) = m;
    
        % solve with limited memory BGFS
        tic
        [xf, iter] = limBGFS_cyclic(f, x_0, maxiter, tol, m);
        results_lmbgfs(k,3) = toc;
        
        results_lmbgfs(k,4) = iter;
        results_lmbgfs(k,5) = norm(xf, 'inf');
        
        k = k+1;
    end   
end
