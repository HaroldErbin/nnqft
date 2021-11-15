"""
Utility functions for the NN-QFT correspondence.
"""

import itertools

from joblib import Parallel, delayed

import numpy as np
from scipy.special import erf, gamma
from scipy.integrate import nquad, quad, solve_ivp

from sklearn.linear_model import LinearRegression, Lasso, Ridge

import matplotlib.pyplot as plt

from logger import Logger


# ---
# Partitions


def build_inputs(x, n, symmetric=True):
    """
    Build inputs to evaluate Green functions.

    Given M inputs, the inputs to the Green functions are all combinations (including identical
    entries) of these M inputs. There are M^n possible inputs.

    If the theory is symmetric (by default), one keeps ordered combinations only to speed up the
    computations.
    """

    points = [np.r_["0,2", v] for v in itertools.product(x, repeat=n)]

    if symmetric is True:
        results = []

        for array in [np.sort(a, axis=0) for a in points]:
            if array.tolist() not in results:
                results.append(array.tolist())

        return np.array(results)
    else:
        return np.array(points)


def partition(collection):
    # https://stackoverflow.com/questions/19368375/set-partitions-in-python

    collection = list(collection)

    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def wick(x):
    """
    Compute Wick contractions for the set x.

    Wick(x_1, ..., x_n) = {P in partitions(x_1, ..., x_n) | |p| = 2 for all p in P}
    """

    return [p for p in partition(x) if set(map(len, p)) == {2}]


# ---
# Kernels and neural network


def gauss_K(x1, x2, sigma_b, sigma_w):
    if isinstance(x1, float):
        d = 1
    else:
        d = len(x1)

    def inner(y1, y2):
        return sigma_w**2 / d * np.dot(y1, y2)

    factor = sigma_w**2

    return sigma_b**2 + factor * np.exp(- inner(x1 - x2, x1 - x2) / 2)


def erf_K(x1, x2, sigma_b, sigma_w):
    if isinstance(x1, float):
        d = 1
    else:
        d = len(x1)

    def inner(y1, y2):
        return sigma_b**2 + sigma_w**2 / d * np.dot(y1, y2)

    num = 2 * inner(x1, x2)
    denom = np.sqrt((1 + 2 * inner(x1, x1)) * (1 + 2 * inner(x2, x2)))

    factor = sigma_w**2 * 2 / np.pi

    return sigma_b**2 + factor * np.arcsin(num / denom)


def relu_K(x1, x2, sigma_b, sigma_w):
    if isinstance(x1, float):
        d = 1
    else:
        d = len(x1)

    def inner(y1, y2):
        return sigma_b**2 + sigma_w**2 / d * np.dot(y1, y2)

    num = inner(x1, x2)
    denom = np.sqrt(inner(x1, x1) * inner(x2, x2))
    factor = sigma_w**2 / (2 * np.pi)

    ratio = num / denom

    # needed because of numerical errors which can make the ratio slightly above 1
    if ratio > 1 or ratio < -1:
        theta = np.arccos(1)
    else:
        theta = np.arccos(ratio)

    return sigma_b**2 + factor * denom * (np.sin(theta) + (np.pi - theta) * np.cos(theta))


class NeuralNetwork:
    """
    Untrained neural network with one hidden layer.

    Since there is no training nor complicated layer, we don't need to use a package for
    neural networks.
    """

    kernels = {
        "gauss": gauss_K,
        "erf": erf_K,
        "relu": relu_K
    }

    layers_map = {
        "gauss": lambda z: np.exp(z),
        "erf": lambda z: erf(z),
        "relu": lambda z: np.maximum(z, 0)
    }

    def __init__(self, N, d_in=1, d_out=1, sigma_b=1., sigma_w=1., model="gauss",
                 bias=True):

        # width
        self.N = int(N)

        self.d_in = int(d_in)
        self.d_out = int(d_out)

        self.sigma_b = sigma_b
        self.sigma_w = sigma_w

        self.bias = bias

        mu_b = 0
        mu_W = 0
        self.mu_b = 0
        self.mu_W = 0

        self.model = model

        # initialize weights and bias with Gaussian distributions
        self.W = [
            np.random.normal(mu_W, sigma_w / np.sqrt(d_in), (N, d_in)),
            np.random.normal(mu_W, sigma_w / np.sqrt(N), (d_out, N))
        ]
        self.b = [
            np.random.normal(mu_b, sigma_b, (N,)),
            np.zeros((d_out,)) if self.bias is False
                else np.random.normal(mu_b, sigma_b, (d_out,))
        ]

    def __call__(self, x):
        z0 = self.layer_outputs(x, 0)
        x1 = self.layers_map[self.model](z0)

        if self.model == "gauss":
            # norm = np.dot(x, x)
            # norm = np.einsum('ijk,ijk->ij', x, x)
            norm = np.linalg.norm(x, axis=-1)**2
            arg = 2 * (self.sigma_b**2 + self.sigma_w**2 / self.d_in * norm)

            # x1 /= np.sqrt(np.exp(arg))
            x1 /= np.sqrt(np.exp(arg)).reshape(*arg.shape, 1)

        return self.layer_outputs(x1, 1)

    def layer_outputs(self, z, n):
        # return np.dot(self.W[n], z) + self.b[n]
        return np.einsum("il,jkl->jki", self.W[n], z) + self.b[n]

    @property
    def kernel(self):
        return self.kernels[self.model]


def build_nets(N, d_in=1, d_out=1, sigma_b=1., sigma_w=1., model="gauss",
               bias=True, n_nets=100, n_bags=1):

    # try to always reproduce the same weights
    # np.random.seed(seed)

    N = int(N)

    n_nets = int(n_nets)
    n_bags = int(n_bags)

    nets = [
        [NeuralNetwork(N, d_in, d_out, sigma_b, sigma_w, model, bias=bias)
         for _ in range(n_nets)]
        for _ in range(n_bags)
    ]

    if n_bags == 1:
        nets = nets[0]

    return nets


def build_kernel(model_params, bias=True):

    model = model_params["model"]
    sigma_b = model_params["sigma_b"]
    sigma_w = model_params["sigma_w"]

    shift = sigma_b**2 if bias is False else 0

    return lambda x, y: NeuralNetwork.kernels[model](x, y, sigma_b, sigma_w) - shift


# ---
# Green functions


def gp_green(points, K):
    """
    Compute the free n-point Green function from Wick theorem.

    It is necessary to give the order explicitly in case the Green function is evaluated at
    multiple sets of points. In general, x has the shape (n_sets, n, d_in).
    The order of the Green function is inferred from the shape of the data.

    Args:
        K (function): kernel function (with parameters defined)
        x (iterable[float]): inputs
    """

    shape = np.shape(points)

    # add a trivial first dimension when only one set of points is given
    if len(shape) == 2:
        points = np.reshape(points, (1, *shape))

    # infer the order of the Green function
    n = points.shape[1]

    if n % 2 == 1:
        results = [0 for x in points]
    else:
        # TODO: works only if d_out = 1
        results = [np.sum([np.prod([K(v[0], v[1]) for v in p]) for p in wick(x)])
                   for x in points]

    if len(results) == 1:
        return results[0]
    else:
        return np.array(results)


def exp_green(points, nets, n_jobs=-1):
    """
    Compute the experimental n-point Green functions for the data.

    Given a single network, the experimental value is obtained by multiplying the output of the
    network for each input. In a given bag, the experimental Green function is obtained by taking
    the mean value. If there is only one bag, this is what the function returns.
    If there are more than one bag, the function returns the mean and standard deviation.
    """

    def bag_result(values):
        """
        Combine predictions for a single bag.
        """

        return np.squeeze(np.mean(np.prod(values, axis=-2), axis=0), axis=-1)

    shape = np.shape(points)

    # add a trivial first dimension when only one set of points is given
    if len(shape) == 2:
        points = np.reshape(points, (1, *shape))

    # infer the order of the Green function
    n = points.shape[1]

    nets_shape = np.shape(nets)

    if len(nets_shape) == 2:
        # bagging

        # parallelizing is useful only for n > 4

        if n_jobs == 1 or n <= 4:
            results = [bag_result([net(points) for net in bag])
                       for bag in nets]
        else:
            parallel = Parallel(n_jobs=n_jobs, verbose=0)
            def compute(b): return bag_result([net(points) for net in b])
            results = parallel(delayed(compute)(bag) for bag in nets)

        return np.mean(results, axis=0), np.std(results, axis=0)
    else:
        # TODO: works only if d_out = 1
        # output shape : (n_sets, n, d_in)

        return bag_result([net(points) for net in nets])


def delta_green(x, nets, K, normed=False, n_jobs=-1):
    """
    Compute the difference between the experimental and free n-point Green functions.

    The result can be normalized by the free Green function.
    """

    gp_result = gp_green(x, K)
    exp_result = exp_green(x, nets, n_jobs=n_jobs)

    nets_shape = np.shape(nets)

    if len(nets_shape) == 2:
        mean_result, std_result = exp_result

        diff = mean_result - gp_result

        if normed is True:
            diff /= gp_result
            std_result /= gp_result

        return diff, std_result
    else:
        diff = exp_result - gp_result

        if normed is True:
            diff /= gp_result

        return diff


def truncate_legs_4pt(points, Kw, cutoff=None):

    shape = np.shape(points)

    # add a trivial first dimension when only one set of points is given
    if len(shape) == 2:
        points = np.reshape(points, (1, *shape))

    # infer the order of the Green function
    n = points.shape[1]
    d = points.shape[-1]

    if cutoff is None:
        cutoff = np.inf

    def integrand(*xy):
        if d == 1:
            # the definition of xy is different if using nquad, quad, etc.
            if len(xy) == 2:
                y = xy[0]
                xx = xy[1]
            else:
                y = xy[0]
                xx = xy[1:]

            return np.prod([Kw(x, y) for x in xx])
        else:
            return np.prod([Kw(x, y) for y, x in zip(xy[:d], xy[d:])])

    def integrate(p):
        # remove 0 from integration range for stability
        eps = 0
        intervals = [(-cutoff, eps), (eps, cutoff)]
        ranges = list(itertools.product(intervals, repeat=d))
        flat_ranges = [np.ravel(r) for r in ranges]

        # ranges = [(-cutoff, cutoff) for _ in range(d)]
        # result = np.array(nquad(integrand, ranges, p))

        # in some tests, nquad seemed to not converge; however it did in latter tests
        # it looks also 3 times faster
        if d == 1:
            result = np.array([quad(integrand, *r, p) for r in flat_ranges])
        else:
            result = np.array([nquad(integrand, r, p) for r in ranges])

        result = np.array([nquad(integrand, r, p) for r in ranges])

        error = np.sum(result[:, 1])
        if error > 1e-6:
            print(f"Warning: error is: {error}")

        return np.sum(result[:, 0])

    # TODO: parallelize?
    result = np.array([integrate(p) for p in points])

    return result


# ---
# Computation functions

def evaluate_network(data, params, N=2, n=2, normed=True, n_jobs=-1, verbose=False):

    n_nets = params.get("n_nets")
    n_bags = params.get("n_bags")
    model_params = params.get("model_params")

    model = params.get("model", "gauss")

    K = build_kernel(model_params)

    try:
        data = data[model]
    except KeyError:
        pass

    points = build_inputs(data, n)

    if isinstance(N, int):
        N = [N]

    def compute(i, n_jobs):
        nets = build_nets(i, n_nets=n_nets, n_bags=n_bags, **model_params)
        return delta_green(points, nets, K, normed=normed, n_jobs=n_jobs)

    N_enum = Logger.verbenum(reversed(N), text="N =", count=False, verbose=verbose)

    if n <= 4 or n_bags == 1:
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        results = parallel(delayed(compute)(i, n_jobs=1) for i in N_enum)
    else:
        results = [compute(i, n_jobs) for i in N_enum]

    if len(N) > 1:
        return dict(zip(N, reversed(results)))
    else:
        return results[0]


def plot_green(green_values, n=2):
    """
    Plot the mean value of Green functions as a function of N.

    The background is defined as the standard deviation
    """

    N = np.array(list(green_values.keys()))

    mean = np.mean(np.abs(np.c_[list(green_values.values())]), axis=-1)

    values = mean[:, 0]
    background = mean[:, 1]

    fig, ax = plt.subplots()

    ax.plot(N, values, label="experimental")
    ax.plot(N, background, label="background")

    ax.legend()

    ax.set_xlabel("$N$")
    ax.set_ylabel(f"$\langle |m_{n}| \\rangle$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig


def histogram_green(green_values, n=2):
    """
    Plot the histogram of Green functions as a function of N.
    """

    N_list = np.array(list(green_values.keys()))
    values = np.c_[list(green_values.values())][:, 0]

    fig, ax = plt.subplots()

    min_val, max_val = np.min(values), np.max(values)
    bins = np.histogram_bin_edges(values, bins=75, range=(min_val, max_val))

    for N, val in zip(N_list, values):
        ax.hist(val, bins=bins, histtype='step', stacked=False, label=f"${N}$")
        # sns.histplot(x=val, bins=bins, ax=ax)

    ax.legend()

    ax.set_xlabel(f"$m_{n}$")
    ax.set_ylabel("counts")

    plt.close()

    return fig


def predict_lambda(data, params, N=2, cutoff=None, bias=True, n_jobs=-1, verbose=False):

    n_nets = params.get("n_nets")
    n_bags = params.get("n_bags")
    model_params = params.get("model_params")

    model = model_params.get("model")

    K = build_kernel(model_params, bias=bias)
    Kw = build_kernel(model_params, bias=False)

    try:
        data = data[model]
    except KeyError:
        pass

    points = build_inputs(data, 4)

    integ = truncate_legs_4pt(points, Kw, cutoff=cutoff)

    if isinstance(N, int):
        N = [N]

    def compute(i):
        nets = build_nets(i, n_nets=n_nets, n_bags=n_bags, **model_params)

        # Halverson et al. (3.32) (without factor 4!)
        return - delta_green(points, nets, K, normed=False, n_jobs=1)[0] / integ

    text = f"(σ_w = {model_params['sigma_w']}) N ="
    N_enum = Logger.verbenum(reversed(N), text=text, count=False, verbose=verbose)

    parallel = Parallel(n_jobs=n_jobs, verbose=0)
    results = parallel(delayed(compute)(i) for i in N_enum)

    if len(N) > 1:
        return dict(zip(N, reversed(results)))
    else:
        return results[0]


def plot_lambda(lambda_values, log=True):

    N = np.array(list(lambda_values.keys()))

    values = np.c_[list(lambda_values.values())]

    if log is True:
        values = np.abs(values)

    if np.shape(values)[1] != 2:
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
    else:
        mean = values[:, 0]
        std = values[:, 1]

    fig, ax = plt.subplots()

    p1 = ax.plot(N, mean)
    ax.fill_between(N, mean - std, mean + std, color='blue', alpha=0.3)

    p2 = ax.fill(np.NaN, np.NaN, color='blue', alpha=0.3)

    if log is True:
        ytext = "$\langle |u_4| \\rangle \pm 1 \sigma$"
    else:
        ytext = "$\langle u_4 \\rangle \pm 1 \sigma$"

    ax.legend([(p2[0], p1[0]), ], [ytext])

    # ax.legend()

    ax.set_xlabel("$N$")
    ax.set_ylabel(ytext)

    ax.set_xscale("log")

    if log is True:
        ax.set_yscale("log")

    plt.close()

    return fig


def lambda_table(data, sigma_w, params, N=2, cutoff=None, bias=True,
                 n_jobs=-1, verbose=False):

    if isinstance(N, int):
        N = [N]

    lambda_values = {i: {} for i in N}

    sw_enum = Logger.verbenum(sigma_w, text="\nsigma_w = ", count=False, verbose=verbose)

    def compute(sw):
        active_params = params.copy()
        active_params["model_params"]["sigma_w"] = sw

        results = predict_lambda(data, active_params, N=N, cutoff=cutoff, bias=bias,
                                 n_jobs=1, verbose=verbose)

        return {i: (np.mean(val), np.std(val)) for i, val in results.items()}

    parallel = Parallel(n_jobs=n_jobs, verbose=0)
    results = parallel(delayed(compute)(i) for i in sw_enum)

    for sw, res in zip(sigma_w, results):
        for i, val in res.items():
            lambda_values[i][sw] = val

    return lambda_values


def plot_lambda_table_sw_N(lambda_values):
    # plot curve \lambda(\sigma_w) for each N

    N = np.array(list(lambda_values.keys()))
    sigma_w = np.array(list(lambda_values[N[0]].keys()))

    fig, ax = plt.subplots()

    for i in N:
        values = np.abs(np.c_[list(lambda_values[i].values())])

        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)

        ax.plot(sigma_w, mean, label=f"$N = {i}$")
        # ax.fill_between(sigma_w, mean - std, mean + std, alpha=0.3)
        ax.fill_between(sigma_w, mean, mean + std, alpha=0.3)

    ax.legend()

    ax.set_xlabel("$\sigma_W$")
    ax.set_ylabel("$\langle |u_4| \\rangle$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.close()

    return fig


def plot_lambda_table_N_sw(lambda_values):
    # plot curves \lambda(N) for each \sigma_w

    N = np.array(list(lambda_values.keys()))
    sigma_w = np.array(list(lambda_values[N[0]].keys()))

    figs = []

    for sw in sigma_w:
        values = {i: v[sw] for i, v in lambda_values.items()}

        title = f"$\sigma_W = {sw}$"

        fig_log = plot_lambda(values, log=True)
        fig_log.axes[0].set_title(title)
        figs.append(fig_log)

        fig = plot_lambda(values, log=False)
        fig.axes[0].set_title(title)
        figs.append(fig)

    plt.close()

    return figs


def nrg_flow_gauss_active_lambda(sw, d_in, sw0, u40):

    return u40 * (sw / sw0)**(4 - d_in)


def fit_lambda_sw(sw, u4, d_in=1):

    #specify sw0 and use index instead
    i0 = 0

    sw = np.log10(sw).reshape(-1, 1)
    u4 = np.log10(np.abs(u4))

    linreg = LinearRegression().fit(sw[i0:], u4[i0:])

    a = float(linreg.coef_)
    b = float(linreg.intercept_)

    pred = linreg.predict(sw)

    fig, ax = plt.subplots()

    ax.plot(sw, u4, label=f"$exp$")
    ax.plot(sw, pred, label=f"$fit$")

    ax.legend()

    ax.set_xlabel("$\log_{10} \sigma_W$")
    ax.set_ylabel("$\log_{10} u_4$")

    sw0, u40 = sw[i0][0], u4[i0]
    ath = d_in - 4
    bth = u40 - (4 - d_in) * sw0

    title = f"$\log_{{10}} |u_{{4,0}}| = {u4[i0]:.2f}$\n"
    title += f"$\log_{{10}} |u_4| = {a:.2f} \, \log_{{10}} \sigma_W {b:+.2f}$\n"
    title += f"theory: $\log_{{10}} |u_4| = {ath:.2f} \, \log_{{10}} \sigma_W {bth:+.2f}$"
    ax.set_title(title)

    plt.close()

    return (a, b), fig


def nrg_flow_gauss(d_in, sw0, u0, kr):
    """
    Integrate passive RG flow.

    Start with initial conditions $\sigma_{w,0}$, $u_{4,0}$ and $u_{6,0} = \kappa$
    and integrate over the range $k_r = [0, k_\text{max}]$. It returns the values of $\sigma_W$,
    $u_4$ and $u_6$ over the range $k_r$.
    """

    # below (86)
    # u20 = m0^2
    # m0 = sigma_w / np.sqrt(2 * d_in)
    u20 = 2 * sw0**2 / d_in

    # a0 = data spacing
    # equiv to Lambda dans Halverson et al.
    # L = 2 * a0 * N0
    # k_min = 2 * np.pi

    if kr[0] == 0:
        kr = np.r_[1e-10, kr[1:]]

    def floweq(k, u):

        # u2 := m(k)**2

        u2, u4, u6 = u

        # eq (62)
        Vd = 1 / (2**d_in * np.pi**(d_in/2) * gamma(d_in/2 + 1))

        d4 = d_in - 4
        d6 = 2 * d_in - 6

        # eq (63)
        u2b = u2 / k**2

        u4b = u4 * k**d4
        u6b = u6 * k**d6

        # eqs (64-66)
        beta2 = (- Vd * u4b / (1 + u2b)**2)
        beta4 = Vd / (1 + u2b)**2 * (- u6b + 6 * u4b**2 / (1 + u2b))
        beta6 = u4b * Vd / (1 + u2b)**3 * (30 * u6b - 90 * u4b**2 / (1 + u2b))

        beta2 = beta2 * k**2
        beta4 = beta4 / k**d4
        beta6 = beta6 / k**d6

        return (beta2 / k, beta4 / k, beta6 / k)

    results = solve_ivp(floweq, (kr[0], kr[-1]), (u20, *u0), t_eval=kr, method="Radau")

    if results.success is False:
        raise ValueError("Solution to flow equations not found.")

    u2, u4, u6 = results.y
    sw = np.sqrt(d_in * u2 / 2)

    return sw, u4, u6


def plot_flow_sol(kr, u):

    sw, u4, u6 = u

    fig, ax = plt.subplots()

    ax.plot(kr, sw, label=f"$\sigma_W$")
    ax.plot(kr, u4, label=f"$u_4$")
    ax.plot(kr, u6, label=f"$u_6$")

    ax.legend()

    ax.set_xlabel("$k$")

    ax.set_xscale("log")
    # ax.set_yscale("symlog")

    plt.close()

    return fig


def plot_flow_swu(u_flow, u_exp=None):

    sw, u4, u6 = u_flow

    fig_u4, ax = plt.subplots()

    ax.plot(sw, u4, label="flow")

    ax.set_xlabel("$\sigma_W$")
    ax.set_ylabel("$\langle u_4 \\rangle$")

    ax.set_xscale("log")
    # ax.set_yscale("symlog")

    if u_exp is not None:
        sw_exp, u4_exp = u_exp
        ax.plot(sw_exp, u4_exp, "+", label="experimental")

    fig_u6, ax = plt.subplots()

    ax.plot(sw, u6, label="flow")

    ax.set_xlabel("$\sigma_W$")
    ax.set_ylabel("$\langle u_6 \\rangle$")

    ax.set_xscale("log")

    plt.close()

    return fig_u4, fig_u6
