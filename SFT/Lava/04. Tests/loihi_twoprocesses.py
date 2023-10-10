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
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU, NeuroCore, Loihi2NeuroCore
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var
from lava.magma.core.model.nc.var import NcVar

# Neuro core model interface
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.model.nc.model import NcProcessModel

class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (10,))
        self.shape = shape
        self.s_out = OutPort(shape=(1,))
        self.u = Var(shape=(1,), init=0)
	

class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inp2 = InPort(shape=(1,))
        self.v = Var(shape=(1,), init=0)



#NcProcessModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(NeuroCore)
class Model1(NcProcessModel):
    u: np.ndarray = LavaNcType(int, np.uint16, precision=12)
    #s_out: PyOutPort = LavaPyType(PyOutPort.SCALAR_DENSE, float)
    s_out: PyOutPort =  LavaNcType(NcOutPort, int)

    def run_spk(self):
        print(f"Running time step: {self.time_step}")
        self.u=self.u+10
        self.s_out.send(self.u)

#PyProcessModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
class Model2(PyLoihiProcessModel):

    v: np.ndarray =   LavaPyType(np.ndarray, int)
    inp2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self):
        print("Receiving data")
        self.v = self.inp2.recv()
        print(f"v: {self.v}")
        print(f"Running time step: {self.time_step}")



from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps



num_steps_per_input = 1
vth=10
step=1
process1 = P1(shape=(1,), bias=step,num_steps=num_steps_per_input, threshold=vth)
process2 = P2()




run_cfg = Loihi2HwCfg(exception_proc_model_map={P1: Model1, P2:Model2})





"""
#PyProcessModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class Model1(PyLoihiProcessModel):
    
    u: np.ndarray = LavaPyType(np.ndarray, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=24)
    def run_spk(self):
        print(f"Running time step: {self.time_step}")
        
        self.u=self.u+10
        print(f"Value of u: {self.u}")
        self.s_out.send(self.u)



#PyProcessModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
class Model2(PyLoihiProcessModel):

    v: np.ndarray =   LavaPyType(np.ndarray, int)

    inp2: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self):
        print("Hola")
        self.v = self.inp2.recv()
        print(f"Running time step: {self.time_step}")

        print(f"v: {self.v}")



from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps



num_steps_per_input = 3
vth=10
step=1
process1 = P1(shape=(10,), bias=step,num_steps=num_steps_per_input, threshold=vth)
process2 = P2 ()
run_cfg = Loihi2SimCfg(exception_proc_model_map={P1: Model1})


"""



print("Starting network")
process1.s_out.connect(process2.inp2)
process2.run(condition=RunSteps(num_steps=num_steps_per_input), run_cfg=run_cfg)


print(f"\nProcess model used: {process1.model_class}")



