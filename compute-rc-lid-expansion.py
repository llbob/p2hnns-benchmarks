import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from p2hnns_benchmarks.distance import euc_p2hdist, ang_p2hdist

DATASETS = [
    "glove-100-euclidean",
    "music-100-euclidean"
]

def compute_lid(file_path, k=100):
    f = h5py.File(file_path)
    distances = np.array(f['distances'])
    estimates = []
    for vec in distances:
        vec.sort()
        w = vec[min(len(vec) - 1, k)]
        half_w = 0.5 * w
        vec = vec[:k+1]
        vec = vec[vec > 1e-5]
        small = vec[vec < half_w]
        large = vec[vec >= half_w]
        s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
        valid = small.size + large.size
        estimates.append(-valid / s if valid > 0 and s != 0 else np.nan)
    return np.array(estimates)

def compute_expansion(file_path, k1=10, k2=20):
    f = h5py.File(file_path)
    expansions = []
    for vec in f['distances']:
        vec.sort()
        if k2 - 1 < len(vec) and k1 - 1 < len(vec) and vec[k1 - 1] > 1e-6:
            expansions.append(vec[k2 - 1] / vec[k1 - 1])
        else:
            expansions.append(np.nan)
    return np.array(expansions)

def compute_rc(file_path, k=10, samples=1000):
    f = h5py.File(file_path)
    distances = np.array(f['distances'])
    normals = np.array(f['normals'])
    biases = np.array(f['biases'])
    dataset = np.array(f['points'])
    m = len(normals)
    estimates = np.zeros(m, dtype=float)
    random_matrix = np.array([random.choice(dataset) for _ in range(samples)])
    hyperplane_avg_distances = np.zeros(m, dtype=float)
    for i in range(m):
        hyperplane = (normals[i], biases[i])
        distances_to_hyperplane = np.array([euc_p2hdist(x, hyperplane) for x in random_matrix])
        hyperplane_avg_distances[i] = np.average(distances_to_hyperplane)
    for i in range(m):
        for j in range(k - 1, 100):
            if j < len(distances[i]) and distances[i][j] > 1e-6:
                estimates[i] = hyperplane_avg_distances[i] / distances[i][j]
                break
    print("rc average:", np.mean(estimates))
    print("rc median:", np.median(estimates))
    print("rc std:", np.std(estimates))
    return estimates

def plot_distribution_box_grid(data_dict, metric_name, clip_max=None, auto_clip=True):
    sorted_datasets = sorted(
        data_dict.keys(),
        key=lambda k: np.nanmedian(data_dict[k])
    )

    all_data = []
    for k in sorted_datasets:
        vals = np.asarray(data_dict[k])
        vals = vals[np.isfinite(vals)]
        if clip_max is not None:
            vals = vals[vals <= clip_max]
        all_data.append(vals)

    flat_data = np.concatenate(all_data)
    if auto_clip and clip_max is None:
        clip_max = np.percentile(flat_data, 99.5)

    all_data = []
    for k in sorted_datasets:
        vals = np.asarray(data_dict[k])
        vals = vals[np.isfinite(vals)]
        if clip_max is not None:
            vals = vals[vals <= clip_max]
        all_data.append(vals)

    flat_data = np.concatenate(all_data)
    lo, hi = np.percentile(flat_data, [0.5, 99.5])
    margin = (hi - lo) * 0.05
    xlim = (max(lo - margin, 0), hi + margin)

    fig, axs = plt.subplots(len(sorted_datasets), 1, figsize=(10, len(sorted_datasets) * 1.3), sharex=True)
    if len(sorted_datasets) == 1:
        axs = [axs]

    for i, dataset in enumerate(sorted_datasets):
        ax = axs[i]
        data = np.asarray(data_dict[dataset])
        data = data[np.isfinite(data)]
        if clip_max is not None:
            data = data[data <= clip_max]

        if len(data) == 0:
            ax.set_visible(False)
            continue

        mean, std = np.mean(data), np.std(data)
        x = np.linspace(xlim[0], xlim[1], 300)
        y = norm.pdf(x, mean, std)

        ax.plot(x, y, color='black', lw=1)
        ax.fill_between(x, 0, y, color='black', alpha=0.1)

        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        for p in [p25, p50, p75]:
            ax.axvline(p, color='black', lw=1)

        ax.text(xlim[1]*0.97, max(y)*0.8, dataset,
                ha='right', va='center', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black'))

        ax.set_yticks([])
        
        ax.set_xlim(xlim)
        
        if i == len(sorted_datasets) - 1:
            ax.set_xlabel(metric_name)
        else:
            ax.tick_params(axis='x', labelbottom=True)
            
        ax.set_ylabel("")

    fig.suptitle(metric_name, fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    lid_results = {}
    exp_results = {}
    rc_results = {}

    for dataset in DATASETS:
        try:
            file_path = f"data/{dataset}.hdf5"
            lid_values = compute_lid(file_path)
            lid_results[dataset] = lid_values
            exp_values = compute_expansion(file_path)
            exp_results[dataset] = exp_values
            rc_values = compute_rc(file_path)
            rc_results[dataset] = rc_values
        except Exception as e:
            print(f"error processing {dataset}: {e}")

    plot_distribution_box_grid(
        lid_results,
        metric_name="Local Intrinsic Dimensionality",
        clip_max=15,
    )

    plot_distribution_box_grid(
        exp_results,
        metric_name="Expansion",
        clip_max=1.5
    )

    plot_distribution_box_grid(
        rc_results,
        metric_name="Relative Contrast",
        clip_max=50000
    )

if __name__ == "__main__":
    main()