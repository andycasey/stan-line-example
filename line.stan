

data {
    int<lower=1> N;
    real x[N];
    real y[N];
    real y_err[N];
}
parameters {
    real m;
    real b;
}

model {
    for (n in 1:N)
        y[n] ~ normal(m*x[n] + b, y_err[n]);
}