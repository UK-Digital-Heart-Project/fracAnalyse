import matplotlib.pyplot as plt

def copyfig(oldfig):
    """
    Copy a figure to a new figure.
    
    Parameters:
    oldfig (matplotlib.figure.Figure): The original figure to copy.
    
    Returns:
    matplotlib.figure.Figure: The new figure.
    """
    # Create new figure
    newfig = plt.figure()
    
    # Copy content
    for ax in oldfig.get_axes():
        new_ax = newfig.add_subplot(111)
        for line in ax.get_lines():
            new_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
        new_ax.set_title(ax.get_title())
        new_ax.set_xlabel(ax.get_xlabel())
        new_ax.set_ylabel(ax.get_ylabel())
        new_ax.legend()
    
    # Copy axis properties
    newfig.set_size_inches(oldfig.get_size_inches())
    newfig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    return newfig

# Example usage:
# fig = plt.figure()
# plt.plot([1, 2, 3], [4, 5, 6])
# new_fig = copyfig(fig)
# plt.show(new_fig)
