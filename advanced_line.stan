

data {
    int<lower=1> N;
    real x[N];
    real y[N];
    real x_err[N];
    real y_err[N];
}
parameters {
    real x_true[N];
    real m;
    real b;
}

model {
    for (n in 1:N) {
        x[n] ~ normal(x_true[n], x_err[n]);
        y[n] ~ normal(m*x_true[n] + b, y_err[n]);
    }
}