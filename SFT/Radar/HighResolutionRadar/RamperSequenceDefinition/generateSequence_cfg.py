import numpy as np
from ..defines import BASIC2, LRR, MRR, LRR_180, MRR_180

def f2n(f):
    return (f / 200e6) * 262144.0


def t2n(t):
    return t / 20e-9


def get_parameter(mode, mmic_id, config):

    nRamps = config.nRamps[mmic_id]  # default
    bw = config.bw[mmic_id]
    tramp = config.tramp[mmic_id]

    if mode == BASIC2:
        mmicCfgs = mmicCfg_basic2()
    elif mode == LRR:
        mmicCfgs = mmicCfg_lrr()
        # if LRR, use settings of MMIC_A for all
        nRamps = config.nRamps[0]
        bw = config.bw[0]
        tramp = config.tramp[0]
    elif mode == MRR:
        mmicCfgs = mmicCfg_mrr()
        # if MRR, use settings of MMIC_C for all
        nRamps = config.nRamps[3]
        bw = config.bw[3]
        tramp = config.tramp[3]
    elif mode == LRR_180:
        mmicCfgs = mmicCfg_lrr_180()
        nRamps = config.nRamps[3]
        bw = config.bw[3]
        tramp = config.tramp[3]
    elif mode == MRR_180:
        mmicCfgs = mmicCfg_mrr_180()
        nRamps = config.nRamps[0]
        bw = config.bw[0]
        tramp = config.tramp[0]
    else:
        raise(NotImplementedError("Please specify a valid mode!"))

    f0 = config.f0
    trampdly = config.trampdly
    twait = config.twait
    tjump = config.tjump

    nstart1 = np.uint32(f2n(f0))
    ntime1 = np.uint32(t2n(tramp + trampdly))
    nstep1 = np.uint32(f2n(bw) / ntime1 * 64)

    nstart2 = np.uint32(f2n(f0 + bw))
    ntime2 = np.uint32(t2n(tjump))
    nstep2 = np.uint32(f2n(bw) / ntime2 * 64)

    ntime3 = np.int32(t2n(twait))

    timings = [nstart1, ntime1, nstep1, nstart2, ntime2, nstep2, ntime3]

    # Master MMIC0 -> MMIC 0
    configSet0 = {'conf0': {'cfg': 0, 'tx': 2, 'lvds': 1, 'txiq': 0},
                  'conf1': {'cfg': 1, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf2': {'cfg': 2, 'tx': 1, 'lvds': 1, 'txiq': 0},
                  'conf3': {'cfg': 3, 'tx': 0, 'lvds': 1, 'txiq': 0},
                  'conf4': {'cfg': 4, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'num_ramps': nRamps}

    # MasterMMIC1 -> MMIC 1
    configSet1 = {'conf0': {'cfg': 0, 'tx': 2, 'lvds': 1, 'txiq': 0},
                  'conf1': {'cfg': 1, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf2': {'cfg': 2, 'tx': 4, 'lvds': 1, 'txiq': 0},
                  'conf3': {'cfg': 3, 'tx': 0, 'lvds': 1, 'txiq': 0},
                  'conf4': {'cfg': 4, 'tx': 2, 'lvds': 0, 'txiq': 0},
                  'num_ramps': nRamps}

    # Master MMIC-M -> MMIC 2
    configSetM = {'conf0': {'cfg': 0, 'tx': 2, 'lvds': 0, 'txiq': 0},
                  'conf1': {'cfg': 1, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf2': {'cfg': 2, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf3': {'cfg': 3, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf4': {'cfg': 4, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'num_ramps': nRamps}

    # Slave MMIC0 -> MMIC 3
    configSet3 = {'conf0': {'cfg': 0, 'tx': 2, 'lvds': 1, 'txiq': 0},
                  'conf1': {'cfg': 1, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf2': {'cfg': 2, 'tx': 1, 'lvds': 1, 'txiq': 0},
                  'conf3': {'cfg': 3, 'tx': 0, 'lvds': 1, 'txiq': 0},
                  'conf4': {'cfg': 4, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'num_ramps': nRamps}

    # Slave MMIC1 -> MMIC 4
    configSet4 = {'conf0': {'cfg': 0, 'tx': 2, 'lvds': 1, 'txiq': 0},
                  'conf1': {'cfg': 1, 'tx': 0, 'lvds': 0, 'txiq': 0},
                  'conf2': {'cfg': 2, 'tx': 4, 'lvds': 1, 'txiq': 0},
                  'conf3': {'cfg': 3, 'tx': 0, 'lvds': 1, 'txiq': 0},
                  'conf4': {'cfg': 4, 'tx': 2, 'lvds': 0, 'txiq': 0},
                  'num_ramps': nRamps}

    # combine the configuration sets
    configSet = [configSet0, configSet1, configSetM, configSet3, configSet4]

    return timings, mmicCfgs[mmic_id], configSet[mmic_id]


def mmicCfg_basic2():

    cfg0, cfg1, cfg2, cfg3 = 0, 1, 2, 3

    mmic_a_cfg = [cfg0, cfg2, cfg3, cfg3, cfg3, cfg3, cfg3, cfg3]
    mmic_b_cfg = [cfg3, cfg3, cfg2, cfg0, cfg3, cfg3, cfg3, cfg3]
    mmic_m_cfg = [cfg0, cfg0, cfg0, cfg0, cfg0, cfg0, cfg0, cfg0]
    mmic_c_cfg = [cfg3, cfg3, cfg3, cfg3, cfg0, cfg2, cfg3, cfg3]
    mmic_d_cfg = [cfg3, cfg3, cfg3, cfg3, cfg3, cfg3, cfg2, cfg0]

    mmicCfgs = {0: mmic_a_cfg, 1: mmic_b_cfg, 2: mmic_m_cfg, 3: mmic_c_cfg, 4: mmic_d_cfg}

    return mmicCfgs


def mmicCfg_lrr():
    cfg0, cfg1, cfg2, cfg3 = 0, 1, 2, 3

    mmic_a_cfg = [cfg0, cfg2, cfg3]
    mmic_b_cfg = [cfg3, cfg3, cfg2]
    mmic_m_cfg = [cfg0, cfg0, cfg0]
    mmic_c_cfg = [cfg1, cfg1, cfg1]
    mmic_d_cfg = [cfg1, cfg1, cfg1]

    mmicCfgs = {0: mmic_a_cfg, 1: mmic_b_cfg, 2: mmic_m_cfg, 3: mmic_c_cfg, 4: mmic_d_cfg}

    return mmicCfgs


def mmicCfg_mrr():
    cfg0, cfg1, cfg2, cfg3, cfg4 = 0, 1, 2, 3, 4

    mmic_a_cfg = [cfg1, cfg1, cfg1, cfg1]
    mmic_b_cfg = [cfg1, cfg1, cfg1, cfg1]
    mmic_m_cfg = [cfg0, cfg0, cfg0, cfg0]
    mmic_c_cfg = [cfg0, cfg2, cfg3, cfg3]
    mmic_d_cfg = [cfg3, cfg3, cfg2, cfg0]

    mmicCfgs = {0: mmic_a_cfg, 1: mmic_b_cfg, 2: mmic_m_cfg, 3: mmic_c_cfg, 4: mmic_d_cfg}

    return mmicCfgs

def mmicCfg_lrr_180():
    cfg0, cfg1, cfg2, cfg3 = 0, 1, 2, 3

    mmic_c_cfg = [cfg0, cfg2, cfg3]
    mmic_d_cfg = [cfg3, cfg3, cfg2]
    mmic_m_cfg = [cfg0, cfg0, cfg0]
    mmic_a_cfg = [cfg1, cfg1, cfg1]
    mmic_b_cfg = [cfg1, cfg1, cfg1]

    mmicCfgs = {0: mmic_a_cfg, 1: mmic_b_cfg, 2: mmic_m_cfg, 3: mmic_c_cfg, 4: mmic_d_cfg}

    return mmicCfgs


def mmicCfg_mrr_180():
    cfg0, cfg1, cfg2, cfg3, cfg4 = 0, 1, 2, 3, 4

    mmic_c_cfg = [cfg1, cfg1, cfg1, cfg1]
    mmic_d_cfg = [cfg1, cfg1, cfg1, cfg1]
    mmic_m_cfg = [cfg0, cfg0, cfg0, cfg0]
    mmic_a_cfg = [cfg0, cfg2, cfg3, cfg3]
    mmic_b_cfg = [cfg3, cfg3, cfg2, cfg0]

    mmicCfgs = {0: mmic_a_cfg, 1: mmic_b_cfg, 2: mmic_m_cfg, 3: mmic_c_cfg, 4: mmic_d_cfg}

    return mmicCfgs
