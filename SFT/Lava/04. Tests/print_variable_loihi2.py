#!/usr/bin/env python
# coding: utf-8

#import modules

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import logging

#lava modules
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.nc.model import NcProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU, NeuroCore, Loihi2NeuroCore
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var
from lava.magma.core.model.nc.var import NcVar



class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (10,))
        bias = kwargs.pop("bias", 0)
        vth = kwargs.pop("vth", 10)
        self.shape = shape
        
        self.u = Var(shape=(1,), init=0)
	


#NcProcessModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(NeuroCore)
class Model1(NcProcessModel):

    u: np.ndarray =   LavaNcType(np.ndarray, np.int32, precision=24)

    def run_spk(self):
        print(f"Running time step: {self.time_step}")

        self.u=self.u+10
        print(f"Value of u: {self.u}")


from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps



num_steps_per_input = 1
vth=10
step=1
process1 = P1(shape=(10,), bias=step,num_steps=num_steps_per_input, threshold=vth)


run_cfg = Loihi2HwCfg(exception_proc_model_map={P1: Model1})






"""

#PyProcessModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class Model1(PyLoihiProcessModel):
    
    u: np.ndarray = LavaPyType(np.ndarray, float)
    
    def run_spk(self):
        print(f"Running time step: {self.time_step}")
        
        self.u=self.u+10
        print(f"Value of u: {self.u}")


from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps



num_steps_per_input = 3
vth=10
step=1
process1 = P1(shape=(10,), bias=step,num_steps=num_steps_per_input, threshold=vth)

run_cfg = Loihi2SimCfg(exception_proc_model_map={P1: Model1})

"""





print("Starting network")

process1.run(condition=RunSteps(num_steps=num_steps_per_input), run_cfg=run_cfg)


print(f"\nProcess model used: {process1.model_class}")


print(f"Final value of u: {process1.u}")


