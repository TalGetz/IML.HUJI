from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools


def test_univariate_gaussian():
    MU = 10
    VAR = 1
    # Question 1 - Draw samples and print fitted model
    x = np.random.normal(MU, VAR, 1000)
    gauss = UnivariateGaussian()
    gauss.fit(x)
    print("({}, {})".format(gauss.mu_, gauss.var_))

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = [10 * i for i in range(1, 101)]
    models = [UnivariateGaussian().fit(x[:sample_sizes[i]]) for i in range(100)]
    expectations = [np.abs(m.mu_ - MU) for m in models]
    fig, ax = plt.subplots()
    ax.plot(sample_sizes, expectations)
    plt.title("error of gaussian fit")
    plt.xlabel("sample size")
    plt.ylabel("absolute error")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    y_estimate = gauss.pdf(x)
    fig, ax = plt.subplots()
    ax.scatter(x, y_estimate)
    plt.title("gaussian estimated pdf\n(expect to see gaussian around 10 with variance 1)")
    plt.xlabel("x")
    plt.ylabel("gaussian pdf")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    model = MultivariateGaussian().fit(X)
    print(model.mu_)
    print(model.cov_)

    # Question 5 - Likelihood evaluation
    AMOUNT_OF_POINTS = 200
    mu_s = [np.array(
        [f1, 0, f3, 0]) for f1, f3 in
        itertools.product(np.linspace(-10, 10, AMOUNT_OF_POINTS), np.linspace(-10, 10, AMOUNT_OF_POINTS))]
    mu_s = np.array(mu_s).reshape((AMOUNT_OF_POINTS, AMOUNT_OF_POINTS, 4))
    likelihood = np.zeros((AMOUNT_OF_POINTS, AMOUNT_OF_POINTS))
    for i in range(mu_s.shape[0]):
        print(i)
        for j in range(mu_s.shape[0]):
            likelihood[i, j] = (model.log_likelihood(mu_s[i, j], cov, X))

    extent = [-10, 10, -10, 10]
    plt.imshow(likelihood, cmap='hot', extent=extent, interpolation="nearest")
    plt.ylabel("value of f1")
    plt.xlabel("value of f3")
    plt.title("heatmap of likelihood based on expectancy\n(i can learn the highest likelihood point [a maximum])")
    plt.show()

    # Question 6 - Maximum likelihood
    index = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    print(np.round(mu_s[index], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
