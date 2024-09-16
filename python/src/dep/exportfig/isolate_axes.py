import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

def isolate_axes(ah, vis=False):
    """
    Isolate the specified axes in a figure on their own.

    Parameters:
    ah (list): A list of axes handles, which must come from the same figure.
    vis (bool): A boolean indicating whether the new figure should be visible. Default: False.

    Returns:
    Figure: The handle of the created figure.
    """
    if not all(isinstance(ax, plt.Axes) for ax in ah):
        raise ValueError('ah must be an array of axes handles')

    # Check that the handles are all in the same figure
    fh = ah[0].figure
    for ax in ah:
        if ax.figure != fh:
            raise ValueError('Axes must all come from the same figure')

    # Create a new figure
    new_fh = plt.figure()
    if not vis:
        plt.close(new_fh)

    new_axes = []
    for ax in ah:
        new_ax = new_fh.add_subplot(111)
        new_ax.set_position(ax.get_position())
        for line in ax.get_lines():
            new_ax.add_line(line)
        for image in ax.get_images():
            new_ax.add_image(image)
        new_axes.append(new_ax)

    return new_fh

# Example usage:
# fig, axs = plt.subplots(2, 2)
# isolated_fig = isolate_axes([axs[0, 0], axs[1, 1]], vis=True)
# plt.show()
