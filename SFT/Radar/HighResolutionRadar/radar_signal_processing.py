import numpy as np
from scipy import io
from scipy import signal
from scipy import constants



def rangeFft(data, config=None):
    nTx, nRx, nRamps, nSamples = np.shape(data)

    # Precalculate the window
    window = np.hanning(nSamples)
    window = window * nSamples / np.sum(window)

    # Appy Window
    data = np.einsum('tcdr,r->tcdr', data, window)

    # Apply Range FFT only positive frequencies
    data = np.fft.fft(data, axis=-1)
    data = data[:, :, :, :nSamples >> 1]

    # Normalise FFT
    y_data = data / nSamples * 2

    if config is not None:
        f_ramp_eff = config['bandwidth'] / (config['t_ramp'] + config['t_ramp_delay']) * config['t_ramp']

        max_range = constants.c * config['n_samples'] / 4 / np.abs(f_ramp_eff)
        x_data = np.linspace(0, max_range, int(config['n_samples'] // 2), endpoint=False)
    else:
        x_data = np.arange(0, len(y_data))

    return x_data, y_data


def dopplerFft(data, config=None):
    nTx, nRx, nRamps, nRange = np.shape(data)

    # Precalculate the window
    window = np.hanning(nRamps)
    window = window * nRamps / np.sum(window)

    # Apply Window
    data = np.einsum('tcdr,d->tcdr', data, window)

    # Apply Doppler FFT
    data = np.fft.fft(data, axis=-2)
    data = np.fft.fftshift(data, axes=-2)

    # Normalise FFT
    y_data = data / nRamps * 2

    if config is not None:
        t_rep = config['t_wait'] + config['t_ramp'] + config['t_ramp_delay'] + config['t_jump']
        max_vel = constants.c / config['f0'] / t_rep / config['n_tx'] / 4
        # Warning: expects Time-Division-Multiplexing
        x_data = np.linspace(-max_vel, +max_vel, config['n_ramps'], endpoint=False)

    else:
        x_data = np.arange(0, len(y_data))

    return x_data, y_data


def azimuthFFT(data, config=None):
    nEl, nAz, nDoppler, nRange = np.shape(data)

    # Precalculate the window
    window = np.hanning(nAz)
    window = window * nAz / np.sum(window)

    # Apply Window
    data = np.einsum('sdcr,d->sdcr', data, window)

    # Apply Azimuth FFT
    data = np.fft.fft(np.conjugate(data),
                      axis=1)  # conjugating inverts polarity of AoA as negative angle in left, positive angle in right.
    data = np.fft.fftshift(data, axes=1)

    # Normalise FFT
    y_data = data / nAz * 2

    if config is not None:
        # Calculate axis
        nEl = config['n_elevation']
        nAz = config['n_azimuth']

        axis_elevation = np.degrees(np.arcsin((np.arange(nEl) - nEl / 2) / nEl / config['antenna_distance_elevation']))
        axis_azimuth = np.degrees(np.arcsin((np.arange(nAz) - nAz / 2) / nAz / config['antenna_distance_azimuth']))
    else:
        axis_elevation = np.arange(0, np.shape(y_data)[0])
        axis_azimuth = np.arange(0, np.shape(y_data)[1])


    return [axis_elevation, axis_azimuth], y_data


def NciRdm(data, nVx=(16*8)):
    return np.sum(np.sum(data, axis=0), axis=0) / nVx

def NciRangeFft(data):
    return np.sum(np.sum(np.sum(data, axis=0), axis=0), axis=0) / np.prod(np.shape(data)[0:-1])


def convPower(data):
    return 20 * np.log10(np.abs(data))


def order_antenna_basic2():
    nEl, nAz = 4, 32

    ant_distance_az = 0.5
    ant_distance_el = 1

    fov_azimuth = 90  # +- (degree)

    ant = np.array([
        6, 7, 4, 5,  # MMIC B
        9, 8, 11, 10,  # MMIC C
        13, 12, 15, 14,  # MMIC D
        2, 3, 0, 1  # MMIC A
    ])

    idx = []
    idx.append(np.concatenate([ant + 2 * 16, ant + 0 * 16]))  # tx2 - tx0
    idx.append(np.concatenate([ant + 3 * 16, ant + 1 * 16]))  # tx3 - tx1
    idx.append(np.concatenate([ant + 4 * 16, ant + 6 * 16]))  # tx4 - tx6
    idx.append(np.concatenate([ant + 5 * 16, ant + 7 * 16]))  # tx5 - tx7

    return idx, {'nEl': nEl, 'nAz': nAz, 'ant_distance_az': ant_distance_az, 'ant_distance_el': ant_distance_el, 'fov_azimuth': fov_azimuth}


def order_antenna_siw_mid():
    nEl, nAz = 3, 16

    ant_distance_az = 0.7
    ant_distance_el = 2

    fov_azimuth = 45  # +- (degree)

    ant = np.array([
        9, 8, 11, 10,  # MMIC C
        13, 12, 15, 14,  # MMIC D
    ])

    idx = []
    idx.append(np.concatenate([ant + 5 * 16, ant + 6 * 16]))  # tx5 - tx6
    idx.append(np.concatenate([ant + 4 * 16, ant + 7 * 16]))  # tx4 - tx7
    idx.append(np.concatenate([ant + 3 * 16]))  # tx3

    return idx, {'nEl': nEl, 'nAz': nAz, 'ant_distance_az': ant_distance_az, 'ant_distance_el': ant_distance_el, 'fov_azimuth': fov_azimuth}


def order_antenna_siw_long():
    nEl, nAz = 2, 16

    ant_distance_az = 2
    ant_distance_el = 2

    fov_azimuth = 15 # +- (degree)

    ant = np.array([
        6, 7, 4, 5,  # MMIC B
        2, 3, 0, 1,  # MMIC A
    ])

    idx = []
    idx.append(np.concatenate([ant + 2 * 16, ant + 1 * 16]))  # tx2 - tx1
    idx.append(np.concatenate([ant + 0 * 16]))  # tx0

    return idx, {'nEl': nEl, 'nAz': nAz, 'ant_distance_az': ant_distance_az, 'ant_distance_el': ant_distance_el, 'fov_azimuth': fov_azimuth}


def order_antenna_siw_long_180():
    """Antenna Sheet mounted 180 degree rotated"""
    nEl, nAz = 2, 16

    ant_distance_az = 2
    ant_distance_el = 2

    fov_azimuth = 15  # +- (degree)

    ant = np.array([
        9, 8, 11, 10,  # MMIC C
        13, 12, 15, 14,  # MMIC D
    ])

    idx = []
    idx.append(np.concatenate([ant + 5 * 16, ant + 7 * 16]))  # tx5 - tx7
    idx.append(np.concatenate([ant + 6 * 16]))  # tx6

    return idx, {'nEl': nEl, 'nAz': nAz, 'ant_distance_az': ant_distance_az, 'ant_distance_el': ant_distance_el, 'fov_azimuth': fov_azimuth}


def order_antenna_siw_mid_180():
    """Antenna Sheet mounted 180 degree rotated"""
    nEl, nAz = 3, 16

    ant_distance_az = 0.7
    ant_distance_el = 2

    fov_azimuth = 45  # +- (degree)

    ant = np.array([
        6, 7, 4, 5,  # MMIC B
        2, 3, 0, 1,  # MMIC A
    ])

    idx = []
    idx.append(np.concatenate([ant + 2 * 16, ant + 0 * 16]))  # tx2 - tx0
    idx.append(np.concatenate([ant + 3 * 16, ant + 1 * 16]))  # tx3 - tx1
    idx.append(np.concatenate([ant + 2 * 16]))  # tx2

    return idx, {'nEl': nEl, 'nAz': nAz, 'ant_distance_az': ant_distance_az, 'ant_distance_el': ant_distance_el, 'fov_azimuth': fov_azimuth}


def phase_calibration(data, calibration_file=None, calibration_data=None):
    if calibration_file is not None:
        calib_vec = io.loadmat(calibration_file)["calib_vec"].flatten()
        return data * calib_vec[:, None, None]
    elif calibration_data is not None:
        return data * calibration_data[:, None, None]
    else:
        return data





def antenna_reorder(data, antenna_sheet='basic2', axis={}, calibration_file=None, calibration_data=None):
    if antenna_sheet in ['basic2']:
        idx, settings = order_antenna_basic2()
    elif antenna_sheet in ['siw_mid', 'SiwMid']:
        idx, settings = order_antenna_siw_mid()
    elif antenna_sheet in ['siw_long', 'SiwLong', 'SiwLRR']:
        idx, settings = order_antenna_siw_long()
    elif antenna_sheet in ['siw_mid_180', 'SiwMid_180']:
        idx, settings = order_antenna_siw_mid_180()
    elif antenna_sheet in ['siw_long_180', 'SiwLong_180', 'SiwLRR_180']:
        idx, settings = order_antenna_siw_long_180()
    else:
        raise ValueError(f'ERROR:  <{antenna_sheet}> antenna sheet not supported. '
                         f'Select one of basic2, siw_long_range, siw_mid_range')

    # Apply reordering
    nTx, nRx, nDoppler, nRange = np.shape(data)
    data = np.reshape(data, (nTx * nRx, nDoppler, nRange))

    idx = np.concatenate(idx).ravel()
    data = data[idx]

    # Apply here phase correction
    data = phase_calibration(data, calibration_file=calibration_file, calibration_data=calibration_data)

    # Apply zero padding for simpler processing
    dst = np.zeros((settings['nEl'] * settings['nAz'], nDoppler, nRange), dtype=np.csingle)
    shape = np.shape(data)
    dst[:shape[0], :shape[1], :shape[2]] = data
    dst = np.reshape(dst, (settings['nEl'], settings['nAz'], nDoppler, nRange))

    # Calculate axis
    nAz = settings['nAz']
    nEl = settings['nEl']

    axis['azimuth'] = np.degrees(np.arcsin((np.arange(nAz) - nAz / 2) / nAz / settings['ant_distance_az']))
    axis['elevation'] = np.degrees(np.arcsin((np.arange(nEl) - nEl / 2) / nEl / settings['ant_distance_el']))

    return dst, settings


def ca_cfar(data, win_parameter, threshold):
    win_width = win_parameter[0]
    win_height = win_parameter[1]
    guard_width = win_parameter[2]
    guard_height = win_parameter[3]

    # Create window mask with guard cells
    mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
    mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0

    # Convert threshold value
    threshold = 10 ** (threshold / 10)

    num_valid_cells_in_window = signal.convolve2d(np.ones(np.shape(data), dtype=float), mask, mode='same')

    # Convert range-Doppler map values to power
    data = np.abs(data) ** 2

    # Perform detection
    rd_windowed_sum = signal.convolve2d(data, mask, mode='same')
    rd_avg_noise_power = rd_windowed_sum / num_valid_cells_in_window
    rd_snr = data / rd_avg_noise_power

    return rd_snr > threshold


def music(data: np.array, search: np.array, spacing: np.array, n_incidents: int, fc: float) -> np.array:
    """

    :param data: shape: (antennas, range)
    :param search: 1-D axis e.g. np.arange(-90, 90, 0.1)
    :param spacing: physical spacing between antenna elements e.g. 0.5 lambda
    :param n_incidents: expected number of reflectors
    :param fc: center frequency e.g. (f_stop + f_start) / 2
    :return:

    https://github.com/vslobody/MUSIC
    https://www.sciencedirect.com/science/article/pii/S209099771300031X
    """
    c0 = 299792458  # speed of light
    (M, k) = np.shape(data)  # M: number of antenna elements, k: number of samples
    r = np.stack([np.zeros(M), spacing, np.zeros(M)], axis=1)
    # r_xx  = E[x, x^H] -> The correlation matrix
    # where H = “Hermitian” means conjugate transpose
    r_xx = np.matmul(data, np.matrix.getH(data)) / k
    # Eigen-decompose
    D, E = np.linalg.eig(r_xx)
    idx = D.argsort()[::-1]
    # lmbd = D[idx]  # Vector of sorted eigenvalues
    E = E[:, idx]  # Sort eigenvectors accordingly
    En = E[:, n_incidents:len(E)]  # Noise eigenvectors (ASSUMPTION: M IS KNOWN)
    # ========= (4a) RECEIVED SIGNAL ========= #
    # Wave number vectors (in units of wavelength/2)
    X1 = np.cos(np.multiply(search, np.pi / 180.))
    X2 = np.sin(np.multiply(search, np.pi / 180.))
    X3 = np.sin(np.multiply(search, 0.))
    kSearch = np.multiply([X1, X2, X3], 2 * np.pi / (c0 / fc))
    ku = np.dot(r, kSearch)
    ASearch = np.exp(np.multiply(ku, -1j))
    chemodan = np.dot(np.transpose(ASearch), En)
    aac = np.absolute(chemodan)
    aad = np.square(aac)
    return np.sum(aad, 1)


def processing_pipeline(data, antenna_sheet, axis, calibration_file=None, calibration_data=None):
    # Calculate FFTs
    rFft = rangeFft(data)
    rPower = convPower(rFft)

    dFft = dopplerFft(rFft)
    dPower = convPower(dFft)

    dat, ant_settings = antenna_reorder(dFft, antenna_sheet=antenna_sheet, axis=axis, calibration_file=calibration_file, calibration_data=calibration_data)
    aFft = azimuthFFT(dat)
    aPower = convPower(aFft)

    return rPower, dPower, aPower, ant_settings

