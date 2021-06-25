"""
NN-QFT correspondence.

This code computes correlation functions and RG flow for neural network using QFT.

References:
- Halverson-Maitri-Stoner (2008.08601)
- Erbin-Lahoche-Ousmane Samary (to appear)
"""

import os
import sys
import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

from logger import Logger
from utils import (build_nets, evaluate_network, plot_green, histogram_green,
                   predict_lambda, plot_lambda, lambda_table, plot_lambda_table_sw_N,
                   plot_lambda_table_N_sw, nrg_flow_gauss, plot_flow_sol, plot_flow_swu,
                   fit_lambda_sw)


# ---
# constants


# ---
# command line

# options: --log : time or jobid, folder or filename
# parameters: save time

argparser = argparse.ArgumentParser()

argparser.description = "Computations for the NN-QFT correspondence."

argparser.add_argument('-v', '--verbose', action='store_true',
                       help='increase output verbosity')

argparser.add_argument('-id', '--jobid', type=str, nargs="?", default="",
                       help="jobid for logging")

argparser.add_argument('-s', '--seed', type=int, nargs="?", default=42,
                       help="set random seed")

argparser.add_argument('-n_jobs', type=int, nargs='?', default=1,
                       help='number of parallel jobs')

argparser.add_argument('-d_in', type=int, nargs='?', default=1,
                       help='input data dimension')

argparser.add_argument('-d_out', type=int, nargs='?', default=1,
                       help='output data dimension')

argparser.add_argument('-n_bags', type=int, nargs='?', default=5,
                       help='number of bags')

argparser.add_argument('-n_nets', type=int, nargs='?', default=1e3,
                       help='number of neural networks per bag')

argparser.add_argument('-sb', '--sigma_b', type=float, nargs='?', default=None,
                       help='value of sigma_b (default: 0 (relunet), 1 (otherwise))')

argparser.add_argument('-sw', '--sigma_w', type=float, nargs='?', default=1.,
                       help='value of sigma_w (default: 1)')

argparser.add_argument('-n', type=int, nargs='?', default=0,
                       help='compute n-point function (default: 2, 4, 6)')

argparser.add_argument('-m', '--model', type=str, nargs="?", default="gauss",
                       choices=["gauss", "erf", "relu"],
                       help="neural network to use")

argparser.add_argument('mode', type=str, nargs="?", default="test",
                       choices=["test", "green", "green_hist", "lambda", "lambda_sw",
                                "combine_lambda", "nrg_active", "nrg_passive"],
                       help='computation to perform')

argparser.add_argument('-t', '--test', action='store_true', default=False,
                       help='use simpler computations')


# dirty trick for VS code
try:
    args = argparser.parse_args()
except SystemExit:
    sys.argv = ['']
    args = argparser.parse_args()

n_jobs = args.n_jobs

if args.mode == "test":
    model = "gauss"
else:
    model = args.model

d_in = args.d_in
d_out = args.d_out

sigma_w = args.sigma_w
if args.sigma_b is None:
    sigma_b = 0. if model == "relu" else 1
else:
    sigma_b = args.sigma_b

n_bags = int(args.n_bags)
n_nets = int(args.n_nets)

orders = int(args.n)
if orders == 0:
    orders = [2, 4, 6]
else:
    orders = [orders]


if d_in > 1:
    raise NotImplementedError("Cannot deal with d_in > 1.")

if d_out > 1:
    raise NotImplementedError("Cannot deal with d_out > 1.")


model_params = {"model": model, "sigma_b": sigma_b, "sigma_w": sigma_w,
                "d_in": d_in, "d_out": d_out}
all_params = {"n_bags": n_bags, "n_nets": n_nets, "seed": args.seed, "model_params": model_params}

np.random.seed(args.seed)


# ---
# Parameters

# 2008.08601v2
widths = [2, 3, 4, 5, 10, 20, 50, 100, 500, 1000]

# 2008.08601v2
inputs_hkm = {
    "gauss": np.array([-1e-2, -6e-3, -2e-3, 6e-3, 2e-3, 1e-2]).reshape(-1, 1),
    # "gaussnet": np.array([-1e-2, -6e-3, -2e-3]).reshape(-1, 1),
    # 2008.08601v1
    # "erf": np.array([-1, -6e-1, -2e-1, 1e-1, 6e-1, 2]).reshape(-1, 1),
    "erf": np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.012]).reshape(-1, 1),
    "relu": np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]).reshape(-1, 1)
}

inputs_large = {
    "gauss": np.c_[np.outer(np.arange(-0.1, -1, -0.4), 10.**np.arange(-2, 2)),
                   np.outer(np.arange(0.1, 1, 0.4), 10.**np.arange(-2, 2))].reshape(-1, 1),
    # 2008.08601v1
    # "erf": np.array([-1, -6e-1, -2e-1, 1e-1, 6e-1, 2]).reshape(-1, 1),
    "erf": np.outer(np.arange(0.1, 1, 0.2), 10.**np.arange(-3, 1)).reshape(-1, 1),
    "relu": np.outer(np.arange(0.1, 1, 0.2), 10.**np.arange(-2, 2)).reshape(-1, 1)
}


# TODO: improve accuracy for erf and relu
if model == "relu":
    cutoff = 50
elif model == "erf":
    cutoff = 1e5
else:
    cutoff = None

logtime = "" if args.jobid != "" else "folder"

suffix = ""

if args.mode in ["green", "green_hist", "lambda"]:
    suffix += f"_sw={sigma_w}"

if args.seed != 42:
    suffix += f"_seed={args.seed}"

prefix = "test_" if args.test is True else ""

logger = Logger(path=f"./results/{model}/", logtime=logtime, args=args,
                prefix=prefix, suffix=suffix)

if args.test is True:
    N = [2, 5, 10, 50]
else:
    N = widths

inputs = inputs_hkm


# ---
# computations: Green functions: plot mean value


if args.mode == "green":
    if args.verbose is True:
        print(f"\n# Experiment: Green function - mean value (model: {model})")

    mean_figs = []
    hist_figs = []
    all_green_values = {}

    for n in orders:
        if args.verbose is True:
            print(f"\nGreen function: n = {n}\n")

        logger.timer()

        green_values = evaluate_network(inputs, all_params, N=N, n=n, normed=True,
                                        n_jobs=n_jobs, verbose=args.verbose)

        logger.timer(show=True)

        mean_figs.append(plot_green(green_values, n=n))
        hist_figs.append(histogram_green(green_values, n=n))

        all_green_values[n] = {i: {"mean": values[0].tolist(), "std": values[1].tolist()}
                               for i, values in green_values.items()}

    logger.save_figs(mean_figs, filename=f"{model}_green_exp_free_mean_{orders}.pdf")
    logger.save_figs(hist_figs, filename=f"{model}_green_exp_free_hist_{orders}.pdf")

    logger.save_json(all_green_values, filename=f"{model}_green_values_{orders}.json")


# ---
# computations: Green functions: plot histogram of errors


if args.mode == "green_hist":
    if args.verbose is True:
        print(f"\n# Experiment: Green function - error histogram (model: {model})")

    if args.test is False:
        N = [5, 10, 50, 100, 1000]

    figs = []
    all_green_values = {}

    for n in (2,):
        if args.verbose is True:
            print(f"\nGreen function: n = {n}\n")

        # inputs = inputs_large if n == 2 else inputs_hkm
        inputs = inputs_large

        green_values = evaluate_network(inputs, all_params, N=N, n=n, normed=True,
                                        n_jobs=n_jobs, verbose=args.verbose)

        figs.append(histogram_green(green_values, n=n))

        all_green_values[n] = {i: {"mean": values[0].tolist(), "std": values[1].tolist()}
                               for i, values in green_values.items()}

    logger.save_figs(figs, filename=f"{model}_green_exp_free_hist.pdf")

    logger.save_json(all_green_values, filename=f"{model}_green_values.json")


# ---
# computations: Compute quartic coupling


if args.mode == "lambda":
    if args.verbose is True:
        print(f"\n# Experiment: Quartic coupling (model: {model})")

    logger.timer()

    lambda_values = predict_lambda(inputs, all_params, N=N, cutoff=cutoff,
                                   n_jobs=n_jobs, verbose=args.verbose)

    logger.timer(show=True)

    fig_log = plot_lambda(lambda_values, log=True)
    fig = plot_lambda(lambda_values, log=False)

    logger.save_figs([fig_log, fig], filename=f"{model}_lambda_mean.pdf")

    logger.save_json({i: values.tolist() for i, values in lambda_values.items()},
                     filename=f"{model}_lambda.json")


# ---
# computations: Compine lambda computations for different \sigma_w


if args.mode == "combine_lambda":
    if args.jobid == "":
        raise ValueError("Must specify `jobid` to find the folder with the values to combine.")

    values = {}
    sigma_w = []

    for f in glob.glob(os.path.join(logger.path, args.jobid, "*gauss_lambda_sw=*.json")):
        sw = float(re.search(r'sw=(\d+\.\d+)', os.path.basename(f)).group(1))
        sigma_w.append(sw)

        values[sw] = logger.load_json(f)

    sigma_w.sort()
    N = list(values[sigma_w[0]].keys())

    lambda_values = {int(i): {sw: (np.mean(values[sw][i]), np.std(values[sw][i])) for sw in sigma_w}
                     for i in N}

    fig = plot_lambda_table_sw_N(lambda_values)
    logger.save_fig(fig, filename=f"{model}_lambda_mean_sw_N.pdf")

    figs = plot_lambda_table_N_sw(lambda_values)
    logger.save_figs(figs, filename=f"{model}_lambda_mean_N_sw.pdf")

    logger.save_json({str(i): {str(sw): v for sw, v in d.items()}
                      for i, d in lambda_values.items()},
                     filename=f"{model}_lambda_mean_sw_N.json")


# ---
# computations: \lambda(\sigma_w)


if args.mode == "lambda_sw":
    if args.verbose is True:
        print(f"\n# Experiment: compute λ(σ_w) (model: {model})")

    if args.test is True:
        sigma_w = np.r_[np.arange(1, 2.1, 0.5), 10]
    else:
        sigma_w = np.r_[np.arange(1, 10.1, 0.5), np.arange(10, 101, 10)]
        N = [5, 10, 50, 100, 500, 1000]

    logger.timer()

    lambda_values = lambda_table(inputs, sigma_w, all_params, N=N, cutoff=cutoff,
                                 bias=False, n_jobs=n_jobs, verbose=args.verbose)

    logger.timer(show=True)

    fig = plot_lambda_table_sw_N(lambda_values)
    logger.save_fig(fig, filename=f"{model}_lambda_mean_sw_N.pdf")

    figs = plot_lambda_table_N_sw(lambda_values)
    logger.save_figs(figs, filename=f"{model}_lambda_mean_N_sw.pdf")

    logger.save_json({str(i): {str(sw): v for sw, v in d.items()}
                      for i, d in lambda_values.items()},
                     filename=f"{model}_lambda_mean_sw_N.json")


# ---
# computations: Active NRG flow


if args.mode == "nrg_active":
    if args.verbose is True:
        print(f"\n# Experiment: Active NRG flow (model: {model})")

    data = logger.load_json(f"{model}_lambda_mean_sw_N.json", logid=True)
    data = {int(i): v for i, v in data.items()}

    u_exp = {i: (np.array(list(map(float, list(v.keys())))), np.array(list(v.values()))[:, 0])
             for i, v in data.items()}

    all_coeffs = {}
    figs = []

    for i, val in u_exp.items():
        sw, u4 = val
        coeffs, fig = fit_lambda_sw(sw, u4)

        all_coeffs[i] = coeffs

        title = f"$N = {i}$, " + fig.axes[0].get_title()
        fig.axes[0].set_title(title)
        figs.append(fig)

    logger.save_figs(figs, filename=f"{model}_lambda_sw_fit.pdf")

    logger.save_json(all_coeffs, filename=f"{model}_lambda_sw_fit.json")


# ---
# computations: Passive NRG flow


if args.mode == "nrg_passive":
    if args.verbose is True:
        print(f"\n# Experiment: Passive NRG flow (model: {model})")

    u60 = 50
    sw0 = 1.

    # u40_list = np.r_[-50, -np.power(10., np.arange(1, -4, -1)), 0.1]

    data = logger.load_json(f"{model}_lambda_mean_sw_N.json", logid=True)
    data = {int(i): v for i, v in data.items()}

    # make a list of all sw and u4
    u_exp = {i: (np.array(list(map(float, list(v.keys())))), np.array(list(v.values()))[:, 0])
             for i, v in data.items()}

    kr = np.arange(0, 100, 0.01)

    # recover u4 for all N with sigma_w = sw0
    u40_list = {i: v[str(sw0)][0] for i, v in data.items()}

    flow_figs = []
    swu4_figs = []
    swu6_figs = []

    results = {}

    logger.timer()

    for i, u40 in u40_list.items():
        u = nrg_flow_gauss(d_in, sw0, (u40, u60), kr)

        results[i] = u

        title = f"$N = {i}, u_{{4,0}} = {np.round(u40, 2)}$"

        fig = plot_flow_sol(kr, u)
        fig.axes[0].set_title(title)

        flow_figs.append(fig)

        fig_u4, fig_u6 = plot_flow_swu(u, u_exp[i])
        fig_u4.axes[0].set_title(title)
        fig_u6.axes[0].set_title(title)

        swu4_figs.append(fig_u4)
        swu6_figs.append(fig_u6)

    logger.timer(show=True)

    logger.save_figs(flow_figs, filename=f"{model}_nrg_passive_sol.pdf")
    logger.save_figs(swu4_figs, filename=f"{model}_nrg_passive_sw_u4.pdf")
    logger.save_figs(swu6_figs, filename=f"{model}_nrg_passive_sw_u6.pdf")

    logger.save_json({str(i): [v.tolist() for v in val] for i, val in results.items()},
                     filename=f"{model}_nrg_passive_sol.json")


# save parameters
if args.test is False and args.mode in ["green", "green_hist", "lambda", "lambda_sw"]:
    logger.save_json({"points": inputs[model].squeeze().tolist(), **all_params},
                    filename=f"{model}_params.json")
