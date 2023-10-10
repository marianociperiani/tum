import numpy as np


def calculate_phase_calibration(data, range_calibration_index):
    """
        Phase calibration function on the ordered data.
        range_calibration_index: is the range index for the given range bin to calibrate on
    """
    nEz, nAz, nDoppler, nRange = np.shape(data)
    ant_vec = data[:, :, nDoppler >> 2, range_calibration_index].flatten()
    ant_vec = data[:, :, 2, range_calibration_index].flatten()
    shape = np.shape(ant_vec)
    # ant_vec = ant_vec[ant_vec != 0] # remove zeroed elements
    with np.errstate(divide='ignore', invalid='ignore'):  # prevent error of zero division
        calib_mag = np.nan_to_num(np.max(abs(ant_vec)) / abs(ant_vec))
    calib_vec = calib_mag * np.exp(-1j * np.angle(ant_vec))
    return calib_vec.reshape((nEz, nAz))

