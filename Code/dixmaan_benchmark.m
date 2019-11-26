%N represents the size of the x vector
N = [200, 1000];
%M represents the size of memory that will be used in limBGFS_cyclic
M = [1,3,5,17,29];

tol = 1e-5;
maxiter = 2000;

% Initialize results vectors
results_bgfs = zeros(1, 4);
results_sr = zeros(1, 4);

results_lmbgfs = zeros(10, 5);

% DixmaanD function
f = @(x) dixmaan(x);

%% Test lineBGFS and Symmetric Rank One Update for n = 200
x_0 = ones(200, 1)*2;

% line BGFS
results_bgfs(1) = 200;

tic
[xf, iter] = lineBGFS(f, x_0, maxiter, tol);
results_bgfs(2) = toc; % time elapsed

results_bgfs(3) = iter;
results_bgfs(4) = norm(grad(f, xf), 'inf');

% symmetric rank one
results_sr(1) = 200;

tic
[xf, iter] = TRSR1(f, x_0, maxiter, tol);
results_sr(2) = toc; % time elapsed

results_sr(3) = iter;
results_sr(4) = norm(grad(f, xf), 'inf');

%% Test lineBGFS with limited memory for n = 200, 1000

% Test methods
k = 1;
for n = N
    
    x_0 = ones(n, 1)*2;
    
    for m = M
        
        results_lmbgfs(k,1) = n;
        results_lmbgfs(k,2) = m;
    
        % solve with limited memory BGFS
        tic
        [xf, iter] = limBGFS_cyclic(f, x_0, maxiter, tol, m);
        results_lmbgfs(k,3) = toc;
        
        results_lmbgfs(k,4) = iter;
        results_lmbgfs(k,5) = norm(grad(f, xf), 'inf');
        
        k = k+1;
    end   
end

%% Displaying the results as tables
header = {'n', 'Tiempo', 'Iteraciones', 'Norm_grad'};
headerMem = {'n', 'Memoria', 'Tiempo', 'Iteraciones', 'Norm_grad'};

T1 = array2table(results_sr, "VariableNames", header);
T2 = array2table(results_bgfs, "VariableNames", header);
T3 = array2table(results_lmbgfs, "VariableNames", headerMem);

fprintf("Resultados de SR1:\n");
disp(T1);
fprintf("Resultados de BGFS:\n");
disp(T2);
fprintf("Resultados de limBGFS:\n");
disp(T3);


%% Export Benchmarks

writetable(T1, "../Benchmarks/SR1_dixmaan_benchmark.csv");
writetable(T2, "../Benchmarks/BGFS_dixmaan_benchmark.csv");
writetable(T3, "../Benchmarks/limBGFS_dixmaan_benchmark.csv");
