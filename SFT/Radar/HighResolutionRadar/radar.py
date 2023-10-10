import ctypes
import math
from typing import Union, Tuple, Iterable
from .defines import *
from .structs import *
from .RamperSequenceDefinition import generateSequence
import numpy as np
import socket
from scipy import constants
import ctypes

def convert_deg2phase(deg):
    return int(deg / 360 * 0x10000)


def convertTime2Tick(t):
    return int(t / 20e-9)


def convertFreq2Tick(f):
    return int(f / 200e6 * 262144.0)


def convert_mmic_bits(setting):
    return 14 if (setting <= 8) else 12


def countBits(number):
    # log function in base 2
    # take only integer part
    return int((math.log(number) / math.log(2)) + 1)


class Configuration(object):
    def __init__(self):
        self.spu = {
            MMIC_a: SpuConfiguration(),
            MMIC_b: SpuConfiguration(),
            MMIC_m: SpuConfiguration(),
            MMIC_c: SpuConfiguration(),
            MMIC_d: SpuConfiguration(),
        }
        self.trcv = {
            MMIC_a: TrcvStruct(),
            MMIC_b: TrcvStruct(),
            MMIC_m: TrcvStruct(),
            MMIC_c: TrcvStruct(),
            MMIC_d: TrcvStruct(),
        }
        self.ramp = {  # only change ramp content through function not direct (SPU reasons)
            MMIC_a: [b''],
            MMIC_b: [b''],
            MMIC_m: [b''],
            MMIC_c: [b''],
            MMIC_d: [b''],
        }

        self._trcv_adc_setting = {2: 0, 3: 1, 4: 2, 6: 4, 8: 5, 12: 7}  # 12 bits currently not supported
        self._decFactor = 4
        self.nTx = 8
        self.nSpus = 4
        self.process = 0
        self.nSamples = 256
        self.tramp = self.nSamples[0] * self.decFactor / 50e6  # 4 comes from bit_size and sampling frequency, 50e6
        self.nRamps = 32
        self.f0 = 76.0e9
        self.bw = 1e9
        self.trampdly = 8.64e-6
        self.twait = 6e-6
        self.tjump = 0.6e-6

        self.phase = TxPhaseConfig()
        self.__default_phase__()
        self.__default_spu__()
        self.__default_trcv__()

    def __default_phase__(self):
        self.phase.phases0 = convert_deg2phase(0.0)
        self.phase.phases1 = convert_deg2phase(0.0)
        self.phase.phases2 = convert_deg2phase(0.0)
        self.phase.phases3 = convert_deg2phase(0.0)
        self.phase.phases4 = convert_deg2phase(0.0)
        self.phase.phases5 = convert_deg2phase(0.0)
        self.phase.phases6 = convert_deg2phase(0.0)
        self.phase.phases7 = convert_deg2phase(0.0)

    def __default_spu__(self):
        """ Single SPU configuration"""
        for spu in self.spu.values():
            spu.dataWidth = 14
            spu.nSpus = self.nSpus
            spu.nTx = self.nTx[0]
            spu.nSpuRx = 4
            spu.nSpuVrx = spu.nSpuRx * spu.nTx
            spu.nSamples = self.nSamples[0]
            spu.nRamps = self.nRamps[0]
            spu.nFft1 = self.nSamples[0]
            spu.nFft2 = self.nRamps[0]
            spu.exp1 = 4
            spu.exp2 = 2
            spu.nRange = int(spu.nFft1 / 2)
            spu.nDoppler = int(spu.nFft2)
            spu.process = 0
            spu.type_window1 = 2
            spu.type_window2 = 2

    def __default_trcv__(self):
        """ MMIC_a """
        self.trcv[0].enableMonitoring = 0
        self.trcv[0].txEnable = 1 + 2
        self.trcv[0].txPower1 = 0x3F
        self.trcv[0].txPower2 = 0x3F
        self.trcv[0].txPower3 = 0  # LO input
        self.trcv[0].nStaticFreq = convertFreq2Tick(self.f0)
        self.trcv[0].calib.nMin = self.trcv[0].nStaticFreq
        self.trcv[0].calib.nMax = convertFreq2Tick(self.f0 + self.bw[0])
        self.trcv[0].rxEnable = 15
        self.trcv[0].mixerGain = 1
        self.trcv[0].lpGain = 6
        self.trcv[0].dcocEnable = 0
        self.trcv[0].dcocShift = 0
        self.trcv[0].adcSettings = self._trcv_adc_setting[self.decFactor]
        self.trcv[0].lvdsRiseDelay = convertTime2Tick(self.trampdly)
        self.trcv[0].lvdsFallDelay = convertTime2Tick(0)

        """ MMIC_b """
        self.trcv[1].enableMonitoring = 0
        self.trcv[1].txEnable = 2 + 4
        self.trcv[1].txPower1 = 0  # LO input
        self.trcv[1].txPower2 = 0x3F
        self.trcv[1].txPower3 = 0x3F
        self.trcv[1].nStaticFreq = convertFreq2Tick(self.f0)
        self.trcv[1].calib.nMin = self.trcv[1].nStaticFreq
        self.trcv[1].calib.nMax = convertFreq2Tick(self.f0 + self.bw[1])
        self.trcv[1].rxEnable = 15
        self.trcv[1].mixerGain = 1
        self.trcv[1].lpGain = 6
        self.trcv[1].dcocEnable = 0
        self.trcv[1].dcocShift = 0
        self.trcv[1].adcSettings = self._trcv_adc_setting[self.decFactor]
        self.trcv[1].lvdsRiseDelay = convertTime2Tick(self.trampdly)
        self.trcv[1].lvdsFallDelay = convertTime2Tick(0)

        """ MMIC_m """
        self.trcv[2].enableMonitoring = 0
        self.trcv[2].txEnable = 2 + 4
        self.trcv[2].txPower1 = 0  # LO input
        self.trcv[2].txPower2 = 0x3F
        self.trcv[2].txPower3 = 0x3F
        self.trcv[2].nStaticFreq = convertFreq2Tick(self.f0)
        self.trcv[2].calib.nMin = self.trcv[2].nStaticFreq
        self.trcv[2].calib.nMax = convertFreq2Tick(self.f0 + self.bw[2])
        self.trcv[2].rxEnable = 15
        self.trcv[2].mixerGain = 1
        self.trcv[2].lpGain = 6
        self.trcv[2].dcocEnable = 0
        self.trcv[2].dcocShift = 0
        self.trcv[2].adcSettings = self._trcv_adc_setting[self.decFactor]
        self.trcv[2].lvdsRiseDelay = convertTime2Tick(self.trampdly)
        self.trcv[2].lvdsFallDelay = convertTime2Tick(0)

        """ MMIC_c """
        self.trcv[3] = self.trcv[0]

        """ MMIC_d """
        self.trcv[4] = self.trcv[1]

    @property
    def f0(self):
        return self._f0

    @f0.setter
    def f0(self, value):
        self._f0 = value
        for trcv in self.trcv.values():
            trcv.nStaticFreq = convertFreq2Tick(self.f0)
            trcv.calib.nMin = trcv.nStaticFreq

    @property
    def decFactor(self):
        return self._decFactor

    @decFactor.setter
    def decFactor(self, value):
        self._decFactor = value
        self._tramp = [samples * self._decFactor / 50e6 for samples in
                       self._nSamples]  # just adjust tramp in case of setting nSamples
        for trcv in self.trcv.values():
            trcv.adcSettings = self._trcv_adc_setting[value]

    @property
    def bw(self):
        return self._bw

    @bw.setter
    def bw(self, value: Union[float, Tuple[float, Tuple]]):
        if isinstance(value, float):
            self._bw = [value] * 5
            for trcv in self.trcv.values():
                trcv.calib.nMax = convertFreq2Tick(self._f0 + value)
        if isinstance(value, tuple):
            bw = value[0]
            for mmic in value[1]:
                self._bw[mmic] = bw
                trcv = self.trcv[mmic]
                trcv.calib.nMax = convertFreq2Tick(self._f0 + bw)

    @property
    def tramp(self):
        return self._tramp

    @tramp.setter
    def tramp(self, value: Union[float, Tuple[int, Tuple]]):
        if isinstance(value, float):
            self._tramp = [value] * 5
        if isinstance(value, tuple):
            tramp = value[0]
            for mmic in value[1]:
                self._tramp[mmic] = tramp

    @property
    def trampdly(self):
        return self._trampdly

    @trampdly.setter
    def trampdly(self, value):
        self._trampdly = value
        for trcv in self.trcv.values():
            trcv.lvdsRiseDelay = convertTime2Tick(self._trampdly)

    @property
    def tjump(self):
        return self._tjump

    @tjump.setter
    def tjump(self, value):
        self._tjump = value

    @property
    def twait(self):
        return self._twait

    @twait.setter
    def twait(self, value):
        self._twait = value

    @property
    def nSamples(self):
        return self._nSamples

    @nSamples.setter
    def nSamples(self, value: Union[int, Tuple[int, Tuple]]):
        if isinstance(value, int):
            self._nSamples = [value] * 5
            # TODO: decimation factor/TRCV setting
            self._tramp = [samples * self._decFactor / 50e6 for samples in self._nSamples]  # just adjust tramp in case of setting nSamples
            for spu in self.spu.values():
                spu.nSamples = value
                spu.Fft1 = value
                spu.nRange = int(value / 2)
        if isinstance(value, tuple):
            nsamples = value[0]
            for mmic in value[1]:
                self._nSamples[mmic] = nsamples
                self._tramp[mmic] = nsamples * self._decFactor / 50e6
                spu = self.spu[mmic]
                spu.nSamples = nsamples
                spu.Fft1 = nsamples
                spu.nRange = int(nsamples / 2)

    @property
    def nRamps(self):
        return self._nRamps

    @nRamps.setter
    def nRamps(self, value):
        if isinstance(value, int):
            self._nRamps = [value] * 5
            for spu in self.spu.values():
                spu.nRamps = value
                spu.Fft2 = value
                spu.nDoppler = int(value)
        if isinstance(value, tuple):
            nramps = value[0]
            for mmic in value[1]:
                self._nRamps[mmic] = nramps
                spu = self.spu[mmic]
                spu.nRamps = nramps
                spu.Fft2 = nramps
                spu.nDoppler = int(nramps)

    @property
    def process(self):
        """
        0: time-data
        1: range-fft
        2: range-doppler-fft
        :return:
        """
        return self._process

    @process.setter
    def process(self, value: int):
        self._process = value
        for spu in self.spu.values():
            spu.process = value

    @property
    def nTx(self):
        """
        number of TX Antennas
        :return:
        """
        return self._nTx

    @nTx.setter
    def nTx(self, value: Union[int, Tuple[int, Tuple]]):
        """
        int: set value for all spu's
        tuple: (value, (mmics))
        """
        if isinstance(value, int):
            self._nTx = [value] * 5
            for spu in self.spu.values():
                spu.nTx = value
                spu.nSpuVrx = spu.nSpuRx * spu.nTx
        if isinstance(value, tuple):
            if isinstance(value[0], int):
                ntx = value[0]
                for mmic in value[1]:
                    self._nTx[mmic] = ntx
                    spu = self.spu[mmic]
                    spu.nTx = ntx
                    spu.nSpuVrx = spu.nSpuRx * spu.nTx
            else:
                raise ValueError('Unexpected Datatype for number of Antennas')

    def to_config(self, mmic_id):
        config= {
            'name': None,
            'n_ramps': self.spu[mmic_id].nRamps,
            'n_mmic': None,
            'n_rx': None,
            'n_tx': self.nTx[mmic_id],
            'bandwidth': self.bw[mmic_id],
            'f0': self.f0,
            't_ramp': self.tramp,
            't_jump': self.tjump,
            't_wait': self.twait,
            't_ramp_delay': self.trampdly,
            'n_elevation': None, 
            'n_azimuth': None, 
            'antenna_distance_elevation': None, 
            'antenna_distance_azimuth': None,
            'field_of_view_azimuth': None,
        }
        return config


def load_preset(scenario) -> Configuration:
    """
    Preset Ramp Scenarios and SPU configs
    :param scenario: e.g. BASIC2, LRR, MRR
    :return:
    """
    parameter = Configuration()  # default values
    if scenario == BASIC2:
        parameter.nTx = 8
        parameter.nSpus = 4
        return parameter
    if scenario == SIW:
        parameter.nSpus = 2  # will be set fort all SPUs
        parameter.decFactor = 2

        # LRR
        parameter.nTx = (3, (MMIC_a, MMIC_b))
        parameter.nSamples = (512, (MMIC_a, MMIC_b))
        parameter.nRamps = (64, (MMIC_a, MMIC_b))
        parameter.bw = (212.7e6, (MMIC_a, MMIC_b))

        # MRR
        parameter.nTx = (4, (MMIC_c, MMIC_d))
        parameter.nSamples = (512, (MMIC_c, MMIC_d))
        parameter.nRamps = (128, (MMIC_c, MMIC_d))
        parameter.bw = (607.7e6, (MMIC_m, MMIC_c, MMIC_d))
        return parameter

    if scenario == SIW_180:
        parameter.nSpus = 2  # will be set fort all SPUs
        parameter.decFactor = 2

        # LRR
        parameter.nTx = (3, (MMIC_c, MMIC_d))
        parameter.nSamples = (512, (MMIC_c, MMIC_d))
        parameter.nRamps = (64, (MMIC_c, MMIC_d))
        parameter.bw = (212.7e6, (MMIC_c, MMIC_d))

        # MRR
        parameter.nTx = (4, (MMIC_a, MMIC_b))
        parameter.nSamples = (512, (MMIC_m, MMIC_a, MMIC_b))
        parameter.nRamps = (128, (MMIC_a, MMIC_b))
        parameter.bw = (607.7e6, (MMIC_m, MMIC_a, MMIC_b))
        return parameter

    raise NotImplementedError(f'No preset found with ID {scenario}')


class Control(object):
    """
     Configuration Class for the HighResolution Radar device from infineon

     Radar consists of 5 MMICs and 2 Aurix devices. Hence, we use the following indexing of the MMICs
     MMIC_a = 0, MMIC_b = 1, MMIC_m = 2, MMIC_c = 3, MMIC_d = 4.
    """

    def __init__(self, HRR_IP, HRR_CTRL_PORT, enable_dual_ethernet=False):
        self.__init_ctrl__(HRR_IP, HRR_CTRL_PORT, enable_dual_ethernet)

        self.enable_dual_ethernet = enable_dual_ethernet

    def __init_ctrl__(self, HRR_IP, HRR_CTRL_PORT, enable_dual_ethernet=False):
        if HRR_IP is None:
            print('Neglect Connection')
            return

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.remote_device = (HRR_IP, HRR_CTRL_PORT)
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_CONNECTION]), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Init connection: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

        self.sock.sendto(bytearray([161, 8, enable_dual_ethernet]), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Dual Ethernet: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def __update_trcv(self, config: Configuration):
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_TRCV_UPDATE,
                                        mmic_id]) + config.trcv[mmic_id], self.remote_device)
            data, _ = self.sock.recvfrom(1024)
            print(f"Update Trcv of MMIC {mmic_id}: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def __update_spu(self, config: Configuration):
        for mmic_id in [MMIC_a, MMIC_b, MMIC_c, MMIC_d]:
            self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_SPU_UPDATE])
                             + ctypes.c_uint8(mmic_id) + config.spu[mmic_id], self.remote_device)
            data, _ = self.sock.recvfrom(1024)
            print(f"Update SPU of MMIC {mmic_id}: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def __update_phase(self, config: Configuration):
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_PHASE_UPDATE]) + config.phase,
                         self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Update Phase: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def __update_ramper(self, config: Configuration):
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            for slot, ramp in enumerate(config.ramp[mmic_id]):
                if len(ramp) == 0:
                    raise ValueError(f"No ramp scenario defined for MMIC: {mmic_id}")
                ramp_size = len(ramp).to_bytes(2, 'little')
                self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_SCENARIO_UPDATE, mmic_id]) +
                                 ramp_size + ctypes.c_uint8(slot) + ramp
                                 , self.remote_device)
                data, _ = self.sock.recvfrom(1024)
                print(f"Update Ramp slot {slot} of MMIC {mmic_id}: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def update(self, mode, config: Configuration):

        generateSequence.define_ramps(mode, config)

        self.__update_trcv(config)
        self.__update_phase(config)
        self.__update_spu(config)
        self.__update_ramper(config)

        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_REINIT]), self.remote_device)
        # Master and Slave Aurix both send a status code
        data, _ = self.sock.recvfrom(1024)
        print(f"Reinit of Master Aurix: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")
        data, _ = self.sock.recvfrom(1024)
        print(f"Reinit of Slave Aurix: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def start(self):
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_RADAR_START, 0]) + ctypes.c_longlong(
            0x7FFFFFFFFFFFFFFF), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Start Measurement: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def stop(self):
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_RADAR_STOP]), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Stop Measurement: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def get_axis_ticks(self):
        c0 = constants.c

        max_range = c0 * self.spu.nSamples / (4 * self.BW)
        axis_range = np.linspace(0, max_range, self.spu.nRange)

        max_velocity = (c0 / self.F0) / (4 * self.tramp)
        axis_doppler = np.linspace(-max_velocity, +max_velocity, self.spu.nDoppler)

        return {'axis_range': axis_range, 'axis_doppler': axis_doppler}

    def get_temperature(self):
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_TEMPERATURE]), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        temperature = np.frombuffer(data[2:22], dtype='single')
        data, _ = self.sock.recvfrom(1024)
        print(f"Get MMIC temperatures: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")
        return temperature

    def get_firmware(self):
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_FIRMWARE]), self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        firmware = np.frombuffer(data[2:22], dtype='single')
        data, _ = self.sock.recvfrom(1024)
        print(f"Get Firmware Version: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")
        return firmware

    def define_schedule(self, schedule: Iterable[ScheduleItem]):
        count = 0
        payload = bytearray()
        for item in schedule:
            count += 1
            payload += item
        count = ctypes.c_uint8(count)
        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_DEFINE_SCHEDULE]) + count + payload, self.remote_device)
        data, _ = self.sock.recvfrom(1024)
        print(f"Get Firmware Version: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")

    def set_cw_mode(self, mmic_id, tx1=False, tx2=False, tx3=False):
        """
            Use this mode if you apply the sensor as aggressor for an interference scenario case.
            Attention: adjust the carrier frequency not to the edges of the victim ramp (decreases likelihood of
            interference). Thus, we have to set a new parameter.f0 or self.f0 in the parameter struct to set the
            static frequency for all MMIC.
            MMIC_A with ID 0 uses tx1 and tx2 equal MMIC_C
            MMIC_B with ID 1 uses tx2 and tx3 equal MMIC_D
            Important not to set the wrong tx path because it is used as input. Return message will provide you
            information about the status of the device and is not zero for any problem. When using the continuous mode
            it does not make sense to record any data so use the exit function or wait some time and stop the cw mode by
            sending ctrl.set_cw_mode(0)
            This function is supported by high resolution images after 2022-07-20.
        """

        assert mmic_id != 0, 'Master will always be in continuous mode'


        payload = np.array(0, dtype=np.uint8) | (tx1 << 0) | (tx2 << 1) | (tx3 << 2)
        payload = ctypes.c_byte(payload)

        self.sock.sendto(bytearray([GUI_COMMUNICATION_CODE, ETHERNET_REQUEST_CW_MODE,mmic_id]) + payload, self.remote_device)

        data, _ = self.sock.recvfrom(1024)
        print(f"Status CW Mode: {ErrorCodes[np.frombuffer(data, dtype='uint8')[2]]}")
