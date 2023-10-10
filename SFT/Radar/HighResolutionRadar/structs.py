""" Collection of c_type structs"""
import ctypes


class CalibStruct(ctypes.Structure):
    _fields_ = [
        ('nMin', ctypes.c_uint32),
        ('nMax', ctypes.c_uint32),
    ]


class TrcvStruct(ctypes.Structure):
    """ TRCV settings structure """
    _fields_ = [
        ('enableMonitoring', ctypes.c_bool),
        ('txEnable', ctypes.c_uint8),
        ('txPower1', ctypes.c_uint8),
        ('txPower2', ctypes.c_uint8),
        ('txPower3', ctypes.c_uint8),
        ('nStaticFreq', ctypes.c_uint32),
        ('calib', CalibStruct),
        ('rxEnable', ctypes.c_uint8),
        ('mixerGain', ctypes.c_uint8),
        ('lpGain', ctypes.c_int8),
        ('dcocEnable', ctypes.c_bool),
        ('dcocShift', ctypes.c_uint8),
        ('adcSettings', ctypes.c_uint8),
        ('lvdsRiseDelay', ctypes.c_uint16),
        ('lvdsFallDelay', ctypes.c_uint16),
        ]


class SpuConfiguration(ctypes.Structure):
    """
        SPU configuration:
        Important to note: match between SPU config and ramp definition (check both)

        Parameters:
            datawidth: supported 12 or 14 bit
            nTx: number of TX antennas on the board
            nSpuRx: number of RX per SPU
            nSpuVrx: number of virtual antenna combinations
            nSamples: Number of samples per payload ramp
            nRamps: number of ramps per scenario (more than in ramp scenario defined not allowed)
            nFft1: number of range-fft bins (bigger than nSamples -> zeropadding)
            nFft2: number of doppler-fft bins
            exp1: range-fft exponent
            exp2: doppler-fft exponent
            nRange: number of range bins (half of nFft1)
            nDoppler: number of doppler bins (=nFft2)
            process: level of signal processing (0=Time data, 1=Range-FFT, 2=Range-Doppler-Map)
            type_window1: range-fft windowing function (1=noWindow, 2:Hanning, 3:Hamming, 4:BlackmanHarris)
            type_window2: doppler-fft windowing function (1=noWindow, 2:Hanning, 3:Hamming, 4:BlackmanHarris)
    """
    _fields_=[
        ('dataWidth', ctypes.c_uint8),
        ('nTx', ctypes.c_uint8),
        ('nSpuRx', ctypes.c_uint8),
        ('nSpuVrx', ctypes.c_uint8),
        ('nSamples', ctypes.c_uint16),
        ('nRamps', ctypes.c_uint16),
        ('nFft1', ctypes.c_uint16),
        ('nFft2', ctypes.c_uint16),
        ('exp1', ctypes.c_uint8),
        ('exp2', ctypes.c_uint8),
        ('nRange', ctypes.c_uint16),
        ('nDoppler', ctypes.c_uint16),
        ('process', ctypes.c_uint8),
        ('type_window1', ctypes.c_uint8),
        ('type_window2', ctypes.c_uint8),
    ]

class TxPhaseConfig(ctypes.Structure):
    """
    TX phase settings. Eight supported phase settings which can be used during
    the ramp definition by indexing the corresponding phase. The phase is represented
    by a fixed point radian number. Hence, we use the function convert_deg2phase().
    """
    _fields_=[
        ('phases0', ctypes.c_uint16),
        ('phases1', ctypes.c_uint16),
        ('phases2', ctypes.c_uint16),
        ('phases3', ctypes.c_uint16),
        ('phases4', ctypes.c_uint16),
        ('phases5', ctypes.c_uint16),
        ('phases6', ctypes.c_uint16),
        ('phases7', ctypes.c_uint16),
    ]


class ScheduleItem(ctypes.Structure):
    """
    define the active Ramp definition and trigger SPU
    -> used to switch between different modes with a multi mode antenna
    """

    _fields_ = [
        ('ramp_slot', ctypes.c_uint8 * 5),     # for 5 MMICs # [MMIC_A, MMIC_B, MMIC_C, MMIC_D, MMIC_M]
        ('activate_spu', ctypes.c_uint8 * 4),  # for 4 SPUs - MMIC master not used # [MMIC_A, MMIC_B, MMIC_C, MMIC_D]
    ]
