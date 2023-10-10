from enum import Enum

""" Ethernet defines"""
GUI_COMMUNICATION_CODE = 161            # necessary communication code for all control requests

# 0 = empty
ETHERNET_REQUEST_CONNECTION = 1         # Connect host to hrr
ETHERNET_REQUEST_PROTOCOL = 2           # select data output protocol (UDP/TCP)
ETHERNET_REQUEST_RADAR_START = 3        # start single/multi frame measurement
ETHERNET_REQUEST_RADAR_STOP = 4         # finish up the current measurement and stop radar
ETHERNET_UNDEFINED = 5                  # empty
ETHERNET_REQUEST_FIRMWARE = 6           # Request the HRR firmware version
ETHERNET_REQUEST_TEMPERATURE = 7        # Request the MMIC temperatures
ETHERN_REQUEST_DUALETHERNET = 8         # enable or disable the dual ethernet mode (0,1)
ETHERNET_UNDEFINED = 9                  # empty
ETHERNET_DEFINE_SCHEDULE = 10           # transmit a sequence that is defining the SPU and ramp config schedule
ETHERNET_REQUEST_REINIT = 11            # reinit the device
ETHERNET_REQUEST_TRCV_UPDATE = 12       # update the transreceiver settings
ETHERNET_REQUEST_SCENARIO_UPDATE = 13   # update the ramper scenario
ETHERNET_REQUEST_PHASE_UPDATE = 14      # update the predefined phase configuration of TX
ETHERNET_REQUEST_SPU_UPDATE = 15        # update the signal processing unit (SPU)
ETHERNET_REQUEST_CW_MODE = 16           # enable/disable the continuous wave mode with settings
# more not supported

""" MMIC defines"""
MMIC_a = 0
MMIC_b = 1
MMIC_m = 2
MMIC_c = 3
MMIC_d = 4

""" Scenarios"""
BASIC2 = 0
LRR = 1
MRR = 2
LRR_180 = 3
MRR_180 = 4
SIW = 5
SIW_180 = 6


ErrorCodes = {
    0: 'GuiStatusOk',
    1: 'GuiNotImplementedError',
    2: 'GuiEthSentFailed',
    3: 'GuiUDPProtocol',
    4: 'GuiTCPProtocol',
    5: 'GuiRawEthernet',
    6: 'GuiInvalidProtocol',
    7: 'GuiSetProtocolError',
    8: 'GuiMasterAurixFailed',
    9: 'GuiSlaveEthEnable',
    10: 'GuiSlaveEthDisable',
    11: 'GuiStartProcessing',
    12: 'GuiStopProcessing',
    13: 'GuiStreamingTestEnable',
    14: 'GuiStreamingTestDisable',
    15: 'GuiInvalidGUICode',
    16: 'GuiInvalidCommandCode',
    17: 'GuiInvalidProcessMode',
    18: 'GuiInvalidMmicId',
    19: 'GuiInvalidScenarioId',
    20: 'GuiHsslTransferCode',
    21: 'GuiRampMemExceeded'}
