N = [2, 10, 100, 200, 1000];
mem = [1, 3, 5, 17, 29];

leq200 = sum(N <= 200);
leq1000 = sum(N <= 1000);

bench_SR1 = zeros(leq200, 4);
bench_BGFS = zeros(leq200, 4);
bench_limBGMS = zeros(leq1000*length(mem), 5);

i1 = 1;
i2 = 1;
i3 = 1;

tol = 1e-5;
maxIters = 10000;

for n = N
    x0 = repmat([-1.2; 1], n, 1)
    f = @(x) rosenbrock(x)

    if n <= 200
        tic;
        [xf, iters] = TRSR1(f, x0, maxIters, tol);
        t = toc;

        bench_SR1(i1, :) = [n, t, iters, norm(grad(f, xf), 'inf')];

        tic;
        [xf, iters] = lineBGFS(f, x0, maxIters, tol);

    elseif n <= 1000
        
    end
end