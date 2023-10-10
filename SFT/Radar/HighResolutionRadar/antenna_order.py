import numpy as np


def reorder_antenna(data, config, custom_tx_index=None, tx_rotated=None, zero_padding=True):
    if config['name'] == 'lrr':
        # lrr
        rx_index = np.array([6, 7, 4, 5, 2, 3, 0, 1])
        tx_index = np.array([2, 1, 0])
        if tx_rotated == True:
            tx_index = np.array([0, 1, 2])
    elif config['name'] == 'mrr':
        # mrr
        rx_index = np.array([1, 0, 3, 2, 5, 4, 7, 6])
        tx_index = np.array([1, 2, 0, 3])
        if tx_rotated == True:
            tx_index = np.array([4, 2, 0, 3, 1])
            tx_index = np.array([3, 1, 2, 0])
    elif config['name'] == 'basic2':
        # basic2
        rx_index = np.array([ 6, 7, 4, 5, 9, 8, 11, 10, 13, 12, 15, 14, 2, 3, 0, 1])
        tx_index = np.array([2, 0, 3, 1, 4, 6, 5, 7])
    else:
        antenna_sheet_name = config['name']
        raise NameError(f'Unsupported configuration name \"{antenna_sheet_name}\". It should be one of the supported antenna sheets')

    if custom_tx_index is not None:
        tx_index = custom_tx_index

    idx = np.concatenate([rx_index + idx * len(rx_index) for idx in tx_index])

    # Apply reordering
    nTx, nRx, nDoppler, nRange = np.shape(data)
    data = np.reshape(data, (nTx * nRx, nDoppler, nRange))
    data = data[idx]

    if zero_padding:
        # Apply zero padding for simpler processing
        dst = np.zeros((config['n_elevation'] * config['n_azimuth'], nDoppler, nRange), dtype=np.csingle)
        shape = np.shape(data)
        dst[:shape[0], :shape[1], :shape[2]] = data
        data = np.reshape(dst, (config['n_elevation'], config['n_azimuth'], nDoppler, nRange))

    return data

