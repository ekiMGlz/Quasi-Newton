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

for n = N
    x0 = repmat([-1.2; 1], n, 1);
    f = @(x) rosenbrock(x);

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

header1 = {'n', 'Tiempo', 'Iteraciones', 'Norm_gf'};
header2 = {'n', 'Memoria', 'Tiempo', 'Iteraciones', 'Norm_gf'};

T1 = array2table(bench_SR1, "VariableNames", header1);
T2 = array2table(bench_BGFS, "VariableNames", header1);
T3 = array2table(bench_limBGFS, "VariableNames", header2);

setHeading(T1, "SR1");
setHeading(T2, "BGFS");
setHeading(T3, "limBGFS");

disp(T1);
disp(T2);
disp(T3);

com_benchmark = bench;
csvwrite("../Benchmarks/computer_matlab_benchmark.csv", com_benchmark);
csvwrite("../Benchmarks/SR1_rosenbrock_benchmark.csv", bench_SR1);
csvwrite("../Benchmarks/BGFS_rosenbrock_benchmark.csv", bench_BGFS);
csvwrite("../Benchmarks/limBGFS_rosenbrock_benchmark.csv", bench_limBGFS);
