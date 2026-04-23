import scienceplots  # noqa: F401
from matplotlib import pyplot as plt

# Colorblind-safe palette (Wong, Nature Methods 2011)
FRAMEWORK_COLORS = {
    "EL": "#000000",  # black
    "EL(Cluster)": "#000000",  # black
    "EL(Pull)": "#000000",  # black
    "EL(Cluster, Depth = 1)": "#E41A1C",  # red
    "MPI": "#009E73",  # bluish green
    "FLUX": "#F0E442",  # yellow
    "Parsl": "#0072B2",  # blue
    "Dask(Depth=0)": "#56B4E9",  # sky blue
    "Dask(Depth=1)": "#CC79A7",  # reddish purple
}

FRAMEWORK_MARKERS = {
    "EL": "o",  # circle
    "EL(Pull)": "o",  # circle
    "EL(Cluster)": "s",  # square
    "EL(Cluster, Depth = 1)": "D",  # diamond
    "MPI": "^",  # triangle up
    "FLUX": "v",  # triangle down
    "Parsl": "X",  # x (filled)
    "Dask(Depth=0)": "P",  # plus (filled)
    "Dask(Depth=1)": "d",  # thin diamond
}

FRAMEWORK_LINESTYLES = {
    "EL": "-",
    "EL(Cluster)": "--",
    "EL(Cluster, Depth = 1)": "-.",
    "EL(Pull)": ":",
    "MPI": "-",
    "FLUX": "-",
    "Parsl": "-",
    "Dask(Depth=0)": "-",
    "Dask(Depth=1)": "-",
}

LABELS_DIRNAMES = {
    "../el": "EL",
    "../el_cluster": "EL(Cluster)",
    "../el_cluster_1level": "EL(Cluster, Depth = 1)",
    "../el_cluster_2level_ws": "EL(Pull)",
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
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.25,
    "lines.markersize": 3,
    "errorbar.capsize": 1.5,
}


_THREE_QUARTER_COLUMN = {
    "figsize": (2.625, 2.25),
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


def get_three_quarter_fig(**kwargs):
    """Create a figure sized for three-quarters of an IEEE two-column page."""
    plt.rcParams.update(
        {k: v for k, v in _THREE_QUARTER_COLUMN.items() if k != "figsize"}
    )
    figsize = kwargs.pop("figsize", _THREE_QUARTER_COLUMN["figsize"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


_TWO_COLUMN = {
    "figsize": (7.0, 2.8),
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.25,
    "lines.markersize": 3,
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
    "lines.linewidth": 1.25,
    "lines.markersize": 3,
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


def get_two_column_fig(ncols=3, nrows=1, **kwargs):
    """Create a two-column-wide figure with `ncols` subplots."""
    plt.rcParams.update({k: v for k, v in _TWO_COLUMN.items() if k != "figsize"})
    figsize = kwargs.pop("figsize", _TWO_COLUMN["figsize"])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def get_full_column_fig(**kwargs):
    """Create a figure sized for a full IEEE two-column column."""
    plt.rcParams.update({k: v for k, v in _FULL_COLUMN.items() if k != "figsize"})
    figsize = kwargs.pop("figsize", _FULL_COLUMN["figsize"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax
