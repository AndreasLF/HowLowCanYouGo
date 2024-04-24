import matplotlib.pyplot as plt
import matplotlib as mpl
import contextlib
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import contextlib

class PaperStylePlotter:
    def __init__(self):
        self.default_settings = {
            # Styling is inspired by: https://github.com/garrettj403/SciencePlots/tree/master

            # Vibrant qualitative color scheme (color blind friendly) from https://personal.sron.nl/~pault/
            'axes.prop_cycle': (cycler('color', ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']) + 
                                cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':'])),
            # Set line style as well for black and white graphs
            # 'axes.prop_cycle' : (cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.'])),

            # Default figure size
            'figure.figsize': (5, 3),

            # dpi setting
            'figure.dpi': 600,

            # X-axis settings
            'xtick.direction': 'in',
            'xtick.major.size': 3,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 1.5,
            'xtick.minor.width': 0.5,
            'xtick.minor.visible': True,
            'xtick.top': True,

            # Y-axis settings
            'ytick.direction': 'in',
            'ytick.major.size': 3,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 1.5,
            'ytick.minor.width': 0.5,
            'ytick.minor.visible': True,
            'ytick.right': True,

            # Line widths
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.0,

            # Legend frame
            'legend.frameon': False,

            # Savefig settings
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,

            # Font settings
            'font.family': 'serif',
            'mathtext.fontset': 'dejavuserif',
            'font.size': 8,
            'font.serif': 'Times'

            # # LaTeX settings
            # 'text.usetex': True,
            # 'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        }
        self.previous_settings = None


    @contextlib.contextmanager
    def apply(self):
        self.previous_settings = mpl.rcParams.copy()
        mpl.rcParams.update(self.default_settings)
        try:
            yield
        finally:
            mpl.rcParams.update(self.previous_settings)


    def save_fig(self, fig, filename, format='pdf', save_folder='figures'):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        fig.savefig(f"{save_folder}/{filename}.{format}", format=format, bbox_inches='tight')


if __name__ == "__main__":
    # Usage example
    with PaperStylePlotter().apply():
        fig, ax = plt.subplots()

        # make some data
        ax.plot([0, 1, 2, 3], [0, 1, 4, 9], label='Line 1')
        ax.plot([0, 1, 2, 3], [0, 3, 6, 9], label='Line 2')
        ax.plot([0, 1, 2, 3], [0, 5, 2, 7], label='Line 3')
        ax.plot([0, 1, 2, 3], [0, 7, 8, 9], label='Line 4')
        ax.plot([0, 1, 2, 3], [0, 9, 1, 5], label='Line 5')
        ax.plot([0, 1, 2, 3], [0, 2, 3, 4], label='Line 6')
        ax.set_title("Paper Style Plot")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()
        # Save using the custom save function
        PaperStylePlotter().save_fig(fig, "paper_style_plot")