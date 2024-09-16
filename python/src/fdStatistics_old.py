import numpy as np

def fd_statistics_old(fd_data, discard_mode):
    eval_slices = len(fd_data)
    quot, rem = divmod(eval_slices, 2)

    if not discard_mode:
        first = 1
        last = eval_slices
        used_slices = eval_slices
    else:
        first = 2
        last = eval_slices - 1
        used_slices = eval_slices - 2
        quot -= 1

    global_fd = np.nanmean(fd_data) if fd_data else np.nan

    basal_slices = range(first, first + quot)
    apical_slices = range(first + quot + rem, last + 1)

    mean_apical_fd = np.nanmean(fd_data[apical_slices]) if fd_data[apical_slices] else np.nan
    max_apical_fd = np.nanmax(fd_data[apical_slices]) if fd_data[apical_slices] else np.nan

    mean_basal_fd = np.nanmean(fd_data[basal_slices]) if fd_data[basal_slices] else np.nan
    max_basal_fd = np.nanmax(fd_data[basal_slices]) if fd_data[basal_slices] else np.nan

    return {
        "eval_slices": eval_slices,
        "used_slices": used_slices,
        "global_fd": global_fd,
        "mean_apical_fd": mean_apical_fd,
        "max_apical_fd": max_apical_fd,
        "mean_basal_fd": mean_basal_fd,
        "max_basal_fd": max_basal_fd,
    }
