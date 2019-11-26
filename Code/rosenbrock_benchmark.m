%% Some parameters for the benchmarks
N = [2, 10, 100, 200, 1000];
mem = [1, 3, 5, 17, 29];

leq200 = sum(N <= 200);
leq1000 = sum(N <= 1000);

bench_SR1 = zeros(leq200, 4);
bench_BGFS = zeros(leq200, 4);
bench_limBGFS = zeros(leq1000*length(mem), 5);

i1 = 1;
i2 = 1;

tol = 1e-5;
maxIters = 2000;
f = @(x) rosenbrock(x);


%% Calculate the benchmarks

for n = N
    x0 = repmat([-1.2; 1], floor(n/2), 1);

    if n <= 200
        tic;
        [xf, iters] = TRSR1(f, x0, maxIters, tol);
        t = toc;

        bench_SR1(i1, :) = [n, t, iters, norm(grad(f, xf), 'inf')];

        tic;
        [xf, iters] = lineBGFS(f, x0, maxIters, tol);
        t = toc;
        
        bench_BGFS(i1, :) = [n, t, iters, norm(grad(f, xf), 'inf')];

        i1 = i1 + 1;

    end
    
    if n <= 1000
        for m = mem
            tic;
            [xf, iters] = limBGFS_cyclic(f, x0, maxIters, tol, m);
            t = toc;
            
            bench_limBGFS(i2, :) = [n, m, t, iters, norm(grad(f, xf), 'inf')];
            
            i2 = i2 + 1;
        end
    end
end


%% Display results

header1 = {'n', 'Tiempo', 'Iteraciones', 'Norm_grad'};
header2 = {'n', 'Memoria', 'Tiempo', 'Iteraciones', 'Norm_grad'};

T1 = array2table(bench_SR1, "VariableNames", header1);
T2 = array2table(bench_BGFS, "VariableNames", header1);
T3 = array2table(bench_limBGFS, "VariableNames", header2);

fprintf("Resultados de SR1:\n");
disp(T1);
fprintf("Resultados de BGFS:\n");
disp(T2);
fprintf("Resultados de limBGFS:\n");
disp(T3);


%% Export Benchmarks

writetable(T1, "../Benchmarks/SR1_rosenbrock_benchmark.csv");
writetable(T2, "../Benchmarks/BGFS_rosenbrock_benchmark.csv");
writetable(T3, "../Benchmarks/limBGFS_rosenbrock_benchmark.csv");

%% Export Computer Benchmark

% com_benchmark = bench;
% csvwrite("../Benchmarks/computer_matlab_benchmark.csv", com_benchmark);
