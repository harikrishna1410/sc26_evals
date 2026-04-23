import scienceplots  # noqa: F401
from matplotlib import pyplot as plt

# Colorblind-safe palette (Wong, Nature Methods 2011)
FRAMEWORK_COLORS = {
    "EL(1)": "#000000",  # black
    "EL(102)": "#000000",  # black
    "Processpool": "#009E73",  # bluish green
    "Parsl": "#0072B2",  # blue
    "Dask": "#CC79A7",  # reddish purple
}

FRAMEWORK_MARKERS = {
    "EL(1)": "o",  # circle
    "EL(102)": "D",  # diamond
    "Processpool": "^",  # triangle up
    "Parsl": "X",  # x (filled)ed)
    "Dask": "d",  # thin diamond
}

FRAMEWORK_LINESTYLES = {
    "EL(1)": "-",
    "EL(102)": "--",
    "Processpool": "-",
    "Parsl": "-",
    "Dask": "-",
}

LABELS_DIRNAMES = {
    "../el": "EL",
    "../el_cluster": "EL(Cluster)",
    "../el_cluster_1level": "EL(Cluster, Depth = 1)",
    "../mpi": "MPI",
    "../parsl_with_trigger": "Parsl",
    "../dask_2level_processpool_with_trigger": "Dask(Depth=1)",
    "../dask_1level_with_trigger": "Dask(Depth=0)",
}

plt.style.use(["science", "ieee", "no-latex"])

# IEEE two-column format dimensions
# Full column width: ~3.5 in, half column: ~1.75 in
_HALF_COLUMN = {
    "figsize": (1.75, 1.5),
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.0,
    "errorbar.capsize": 1.5,
}


_FULL_COLUMN = {
    "figsize": (3.5, 2.5),
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "lines.markersize": 2.0,
    "errorbar.capsize": 1.5,
}

# Default to full column
plt.rcParams.update({k: v for k, v in _FULL_COLUMN.items() if k != "figsize"})


def get_half_column_fig(**kwargs):
    """Create a figure sized for half an IEEE two-column column."""
    plt.rcParams.update({k: v for k, v in _HALF_COLUMN.items() if k != "figsize"})
    figsize = kwargs.pop("figsize", _HALF_COLUMN["figsize"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def get_full_column_fig(**kwargs):
    """Create a figure sized for a full IEEE two-column column."""
    plt.rcParams.update({k: v for k, v in _FULL_COLUMN.items() if k != "figsize"})
    figsize = kwargs.pop("figsize", _FULL_COLUMN["figsize"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax
