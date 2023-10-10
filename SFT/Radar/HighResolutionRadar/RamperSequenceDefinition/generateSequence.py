import numpy as np
from . import generateSequence_cfg
from ..defines import *
#import generateSequence_cfg as generateSequence_cfg


def get_bit(value, bit_index):
    return value & (1 << bit_index)


def set_bit(word, value, length):
    word = word << length
    for bit_index in range(length):
        bit = get_bit(value, bit_index)
        if bit:
            word = word | (1 << bit_index)
    return word


def define_config_set(configSet):
    config = configSet['cfg']
    tx = configSet['tx']
    lvds = configSet['lvds']
    txiq = configSet['txiq']

    header = 0b0
    word_low = 0b0
    word_high = 0b0

    header = set_bit(header, 2, 2)                                  # hdr type
    header = set_bit(header, 1, 1)                                  # ch_ld_conf_high
    header = set_bit(header, 1, 1)                                  # ch_ld_conf_low
    header = set_bit(header, 2, 6)                                  # num_conf
    header = set_bit(header, 0, 1)                                  # set Res_5 bit to 0
    header = set_bit(header, config, 5).to_bytes(2, 'little')       # conf_idx_start

    word_low = set_bit(word_low, txiq, 3)                           # tx3_iqm_sel
    word_low = set_bit(word_low, txiq, 3)                           # tx2_iqm_sel
    word_low = set_bit(word_low, txiq, 3)                           # tx1_iqm_sel
    word_low = set_bit(word_low, tx, 3).to_bytes(2, 'little')       # tx_out_en

    word_high = set_bit(word_high, lvds, 1).to_bytes(2, 'little')   # data_enable

    return header + word_low + word_high


def segment(time, start, step, cfg, num_ramps=None, frame_end=False):
    header = 0b0            # header = word 1
    ntime = 0b0             # word 3-4
    nstart = 0b0            # word 5-6
    nstep = 0b0             # word 7-8
    conf_set_set = 0b0      # word 9

    header = set_bit(header, 1, 2)                                      # hdr type
    header = set_bit(header, 1, 1)                                      # hdr CH_LD_CONF_SET
    header = set_bit(header, 1, 1)                                      # CH_LD_NSTEP
    header = set_bit(header, 1, 1)                                      # CH_LD_NSTART
    header = set_bit(header, 1, 1)                                      # CH_LD_NTIME
    if frame_end:
        header = set_bit(header, 1, 1)                                  # LOOP_END
    else:
        header = set_bit(header, 0, 1)                                  # LOOP_END
    if num_ramps:
        header = set_bit(header, 1, 1)                                  # LOOP_BEGIN
    else:
        header = set_bit(header, 0, 1)                                  # LOOP_BEGIN
    if frame_end:
        header = set_bit(header, 1, 1)                                  # LAST
    else:
        header = set_bit(header, 0, 1)                                  # LAST
    header = set_bit(header, 1, 1)                                      # OP
    header = set_bit(header, 0, 4)                                      # RES_2
    header = set_bit(header, 0, 2).to_bytes(2, 'little')                # SEG_SEL

    ntime = set_bit(ntime, time, 19).to_bytes(4, 'little')              # time

    nstart = set_bit(nstart, start, 29).to_bytes(4, 'little')           # start

    nstep = set_bit(nstep, step, 23).to_bytes(4, 'little')              # step

    conf_set_set = set_bit(conf_set_set, cfg, 5).to_bytes(2, 'little')  # conf_idx

    if num_ramps:
        loop = 0b0

        loop = set_bit(loop, num_ramps, 10).to_bytes(2, 'little')       # num ramps

        return header + loop + ntime + nstart + nstep + conf_set_set

    return header + ntime + nstart + nstep + conf_set_set


def segment_ramp(cfg, time, start, step, num_ramps=None):
    if num_ramps:
        return segment(time, start, step,
                       cfg, num_ramps)
    else:
        return segment(time, start, step, cfg)


def segment_jump(time, start, step):
    return segment(time, start, -step, 1)


def segment_wait(time, start, frame_end=False):
    if frame_end:
        return segment(time, start, 0, 1, frame_end=frame_end)
    else:
        return segment(time, start, 0, 1)


def ramp(cfg, timings, num_ramps=None, frame_end=False):
    nstart1, ntime1, nstep1, nstart2, ntime2, nstep2, ntime3 = timings
    seq = b''
    seq += segment_ramp(cfg, ntime1, nstart1, nstep1, num_ramps)
    seq += segment_jump(ntime2, nstart2, nstep2)
    seq += segment_wait(ntime3, nstart1, frame_end)
    return seq


def define_ramps(mode, config):

    if mode == BASIC2:

        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            timings, mmicCfgs, configSet = generateSequence_cfg.get_parameter(mode, mmic_id, config)
            config.ramp[mmic_id] = []
            config.ramp[mmic_id].append(define_sequence(timings, mmicCfgs, configSet))

    if mode == SIW:
        # LONG Range Ramps
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            timings, mmicCfgs, configSet = generateSequence_cfg.get_parameter(LRR, mmic_id, config)
            config.ramp[mmic_id] = []
            config.ramp[mmic_id].append(define_sequence(timings, mmicCfgs, configSet))
        # MID Range Ramps
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            timings, mmicCfgs, configSet = generateSequence_cfg.get_parameter(MRR, mmic_id, config)
            config.ramp[mmic_id].append(define_sequence(timings, mmicCfgs, configSet))

    if mode == SIW_180:
        # LONG Range Ramps
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            timings, mmicCfgs, configSet = generateSequence_cfg.get_parameter(LRR_180, mmic_id, config)
            config.ramp[mmic_id] = []
            config.ramp[mmic_id].append(define_sequence(timings, mmicCfgs, configSet))
        # MID Range Ramps
        for mmic_id in [MMIC_a, MMIC_b, MMIC_m, MMIC_c, MMIC_d]:
            timings, mmicCfgs, configSet = generateSequence_cfg.get_parameter(MRR_180, mmic_id, config)
            config.ramp[mmic_id].append(define_sequence(timings, mmicCfgs, configSet))



def define_sequence(timings, mmicCfgs, configSet):
    num_ant = len(mmicCfgs)

    sequence = b''
    sequence += define_config_set(configSet['conf0'])
    sequence += define_config_set(configSet['conf1'])
    sequence += define_config_set(configSet['conf2'])
    sequence += define_config_set(configSet['conf3'])
    sequence += define_config_set(configSet['conf4'])

    sequence += ramp(cfg=mmicCfgs[0], timings=timings, num_ramps=configSet['num_ramps'])

    for i in range(1, num_ant-1):
        sequence += ramp(cfg=mmicCfgs[i], timings=timings)

    sequence += ramp(cfg=mmicCfgs[-1], timings=timings, frame_end=True)

    return sequence

