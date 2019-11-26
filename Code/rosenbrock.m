function f = rosenbrock(x)
    n = length(x);
    c = 100;
    f = 0;
    
    for i = 2:2:n
        f = f + c*(x(i) - x(i-1)^2)^2 + (1 - x(i-1))^2;
    end
end