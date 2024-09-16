import numpy as np

def fdStatistics(fd_data, discard_mode):
    # Create some default outputs in case of an early return
    fd_stats = {
        'evalSlices': 0,
        'usedSlices': 0,
        'globalFD': 0.0,
        'meanBasalFD': 0.0,
        'maxBasalFD': 0.0,
        'meanApicalFD': 0.0,
        'maxApicalFD': 0.0
    }
    
    print("Initialized fdStats structure.")

    # Quit if the input array is actually empty
    if len(fd_data) == 0:
        print("Empty input array. Returning early.")
        return fd_stats

    # Note the first and last processed slices
    Lower = 0
    Upper = len(fd_data) - 1

    # Calculate the number of "evaluated" slices
    fd_stats['evalSlices'] = Upper - Lower + 1

    # Return if there are too few values to yield both basal and apical statistics
    N = Upper - Lower + 1

    if not discard_mode and N < 2:
        print("Not enough slices to calculate basal and apical statistics. Returning early.")
        return fd_stats
    elif discard_mode and N < 4:
        print("Not enough slices after discard mode. Returning early.")
        return fd_stats

    # Trim the data further if end slices are being discarded
    if discard_mode:
        Lower += 1
        Upper -= 1
        N -= 2

    # Trim the working array and assign safely to a new variable
    fd = fd_data[Lower:Upper+1]

    Q = N // 2
    R = N % 2

    if R == 0:
        A, B = 0, Q - 1
        C, D = Q, N - 1
    elif R == 1:
        A, B = 0, Q - 1
        C, D = Q + 1, N - 1

    # Now create some arrays and calculate the summary statistics, discarding 0.0's and NaN's
    GlobalFD = np.array(fd)
    BasalFD = np.array(fd[A:B+1])
    ApicalFD = np.array(fd[C:D+1])

    THRESHOLD = 0.1  # Avoid comparing a floating-point value to 0.0

    GlobalFD = GlobalFD[np.logical_not(np.isnan(GlobalFD)) & (GlobalFD >= THRESHOLD)]
    BasalFD = BasalFD[np.logical_not(np.isnan(BasalFD)) & (BasalFD >= THRESHOLD)]
    ApicalFD = ApicalFD[np.logical_not(np.isnan(ApicalFD)) & (ApicalFD >= THRESHOLD)]

    fd_stats['usedSlices'] = len(GlobalFD)

    if len(GlobalFD) == 0:
        fd_stats['globalFD'] = np.nan
        print("GlobalFD is empty. Assigned NaN to fdStats.globalFD.")
    else:
        fd_stats['globalFD'] = np.nanmean(GlobalFD)
        print(f"GlobalFD mean: {fd_stats['globalFD']:.4f}")

    if len(BasalFD) == 0:
        fd_stats['meanBasalFD'] = np.nan
        fd_stats['maxBasalFD'] = np.nan
        print("BasalFD is empty. Assigned NaN to meanBasalFD and maxBasalFD.")
    else:
        fd_stats['meanBasalFD'] = np.nanmean(BasalFD)
        fd_stats['maxBasalFD'] = np.nanmax(BasalFD)
        print(f"meanBasalFD: {fd_stats['meanBasalFD']:.4f}, maxBasalFD: {fd_stats['maxBasalFD']:.4f}")

    if len(ApicalFD) == 0:
        fd_stats['meanApicalFD'] = np.nan
        fd_stats['maxApicalFD'] = np.nan
        print("ApicalFD is empty. Assigned NaN to meanApicalFD and maxApicalFD.")
    else:
        fd_stats['meanApicalFD'] = np.nanmean(ApicalFD)
        fd_stats['maxApicalFD'] = np.nanmax(ApicalFD)
        print(f"meanApicalFD: {fd_stats['meanApicalFD']:.4f}, maxApicalFD: {fd_stats['maxApicalFD']:.4f}")

    return fd_stats
