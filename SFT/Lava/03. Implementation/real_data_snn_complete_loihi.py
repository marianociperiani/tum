#!/usr/bin/env python
# coding: utf-8

# # **1-Layer-SNN (Loihi)**
# Encodes the data in a process, then sends it to a Dense Layer where all the processing layers will be contained.

# ### Importing modules


#import modules

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import logging

#lava modules
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.resources import NeuroCore
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var
from lava.proc.dense.process import Dense

#spikingFT https://github.com/KI-ASIC-TUM/time-coded-SFT
import spikingFT.models.snn
import spikingFT.utils.ft_utils




#Nc Libraries
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.resources import NeuroCore
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.model.nc.var import NcVar
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.net import NetL2



# ### Creating processes
# Two Processes: one with an OutPort, one with an InPort and OutPort

# In[ ]:


class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1024,))
        bias = kwargs.pop("bias", 0)
        vth = kwargs.pop("vth", 10)
        data = kwargs.get("data",)
        nsamples = kwargs.get("nsamples",1024)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.spiked = Var(shape=shape, init=0)
        self.acc_spikes = Var(shape=shape, init=0)
        self.time_spiked = Var(shape=shape, init=0)
        self.tspk = Var(shape=shape, init=0)
        self.k = Var(shape=shape, init=0)
        self.bias = Var(shape=shape, init=bias)
        self.vth = Var(shape=(1,), init=vth)
        self.data = Var(shape=(4,1024),init=data)
        self.nsamples = Var(shape=(1,),init=nsamples)

class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inp2 = InPort(shape=shape)
        self.s_out2 = OutPort(shape=(1022,))

        self.time_spiked2 = Var(shape=(1022,), init=0)
        self.v_membrane = Var(shape=shape, init=0)
        self.re_weights = Var(shape=shape, init=0)
        self.im_weights =  Var(shape=shape, init=0)
        self.i_real = Var(shape=shape, init=0)
        self.i_imag = Var(shape=shape, init=0)
        self.v_real = Var(shape=shape, init=0)
        self.v_imag = Var(shape=shape, init=0)
        self.stacked = Var(shape=(1022,), init=0)
        self.stacked2 = Var(shape=(1022,), init=0)
        self.spiked2 = Var(shape=(1022,), init=0)
        self.refractory =  Var(shape=(1022,), init=0)

class P3(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1022,))

        self.inp3 = InPort(shape=shape)


# ### Creating processes models

# In[ ]:


#PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(NeuroCore)
@tag('floating_pt')
class NcModel1(AbstractNcProcessModel):
    a_in: NcInPort = LavaNcType(NcInPort, np.int16, precision=16)
    s_out: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)
    
    u: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    spiked: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    acc_spikes: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    time_spiked: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    bias: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    tspk: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    k: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    vth: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    data: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    nsamples:NcVar = LavaNcType(NcVar, np.int32, precision=24)
    
    def run_spk(self):

        if(self.time_step==1):

            """     
            0: "case 1: one strong reflection and one weak reflection@25,201",   
            1:"case 2:one weak reflection far away@404",
            2:"two reflections close to each other@42,66",
            3:"multiple reflections@1,39,87,127
            """

            a_in_data=self.data[2,:]
            self.u=a_in_data
            #Calculates the slope
            if(self.time_step==1):
                m=(1-num_steps_per_input/2)/(np.max(self.u)-np.min(self.u))
                b=1-m*np.max(self.u)

                for index2 in range(len(self.u)):
                    self.tspk[index2] =m*self.u[index2]+b
                    self.k[index2] = (vth/self.tspk[index2])-self.u[index2]


        if(self.time_step<=num_steps_per_input/2):


            #Keeps adding up until the spiked is produced.
            for index in range(len(self.v)):
                if (self.acc_spikes[index]<1):
                    self.v[index] += self.u[index]+self.k[index]

            # Check if the threshold is reached.
            s_out = self.v>= self.vth
            self.spiked[:]=s_out

            #Variable to save spikes (to keep the voltage = 0 after neuron spiked)
            self.acc_spikes+=self.spiked
            
            #Saves time at the spike is produced.
            self.time_spiked[s_out]=self.time_step

            # Reset voltage to 0 (Refactory period).
            self.v[s_out] = 0 


            self.s_out.send(self.spiked)
            # Sends spike.



#PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(NeuroCore)
@tag('floating_pt')
class NcModel2(AbstractNcProcessModel):

    inp2: NcInPort = LavaNcType(NcInPort, np.int16, precision=16)
    s_out2: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)

    time_spiked2: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v_membrane: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    i_real: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    i_imag: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v_real: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v_imag: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    stacked2: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    stacked: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    refractory: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    spiked2: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    re_weights: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    im_weights: NcVar = LavaNcType(NcVar, np.int32, precision=24)


    def run_spk(self):
        if(self.time_step==1):
            #Getting weight arrays
            re_weights, im_weights = spikingFT.utils.ft_utils.dft_connection_matrix(
                            nsamples,
                            "numpy"
                            )
            self.re_weights=re_weights
            self.im_weights=im_weights

        #Add up currents (weights x spikes at each time step)
        if(self.time_step<=num_steps_per_input/2):
            in_data1 = self.inp2.recv()
        
            k_real = np.dot(self.re_weights, in_data1.transpose())
            k_imag = np.dot(self.im_weights, in_data1.transpose())

            self.i_real+=k_real
            self.i_imag+=k_imag

            self.v_real+=self.i_real
            self.v_imag+=self.i_imag

        #Simulation time reaches Ts
        if(self.time_step==num_steps_per_input/2):

            #Getting rid of offset and negative spectrum.
            sft_real = self.v_real[1:int(nsamples/2)]
            sft_imag = self.v_imag[1:int(nsamples/2)]

            # Max possible voltage during charging stage is the zero-mode intensity
            # for a wave containing a flat x_max divided by two.
            self.v_threshold =np.sum(self.re_weights[0,:]) * (num_steps_per_input/2) / 4

            #Calculating current to add up at each time step.
            i_spiking=2*self.v_threshold/(num_steps_per_input/2)
            self.stacked=np.hstack([sft_real, sft_imag])
            self.stacked2=np.hstack([sft_real, sft_imag])

        
            sft_max = np.max(np.abs(np.hstack([sft_real, sft_imag])))

            sft_real = np.divide(self.v_real,sft_max)
            sft_imag = np.divide(self.v_imag,sft_max)
            sft_modulus = np.sqrt(sft_real**2 + sft_imag**2)
            sft_modulus = np.log10(9*sft_modulus/sft_modulus.max()+1)

            print("SFT Modulus: \n{}\n".format(sft_modulus))
            fig, ax = plt.subplots(1, 1,figsize=(5, 5))
            ax.plot(sft_modulus[1:512])
            plt.show()


        #Until simulation time reaches 2*Ts
        if(self.time_step>num_steps_per_input/2 and self.time_step<=num_steps_per_input):
            #Calculating current to add up at each time step.
            i_spiking=2*self.v_threshold/(num_steps_per_input/2)
            #Keeps adding up until the spiked is produced.
            for index5 in range(len(self.stacked)):
                if (self.refractory[index5]<1):
                    self.stacked[index5] += i_spiking

            # Check if the threshold is reached.
            s_out2 = self.stacked>= self.v_threshold

            self.spiked2[:]=s_out2

            #Variable to save spikes (to keep the voltage = 0 after neuron spiked).
            self.refractory+=self.spiked2
            
            # Reset voltage to 0 (Refactory period).
            self.stacked[s_out2] = 0 

            #Time step at which the threshold is reached by each neuron.
            self.time_spiked2[s_out2]=self.time_step

            # Sends spike.
            self.s_out2.send(self.stacked)




            if(self.time_step==num_steps_per_input):
                print("\nInputs: \n{}\n".format(self.stacked2))
                print("\nSpiking times: \n{}\n".format(self.time_spiked2))
                plt.plot(self.time_spiked2,self.stacked2,'ro')
                plt.xlabel("Time (steps)")
                plt.ylabel("Voltage at end of Silent Stage)")
                plt.title('Spiking time')
                print("-")
                plt.show()





#PyProcModel implementing P3
@implements(proc=P3, protocol=LoihiProtocol)
@requires(NeuroCore)
@tag('floating_pt')
class NcModel3(AbstractNcProcessModel):

    inp3: NcInPort = LavaNcType(NcInPort, np.int16, precision=16)

    def run_spk(self):
        if(self.time_step>num_steps_per_input/2 and self.time_step<=num_steps_per_input):
            in_data1 = self.inp3.recv()



# ### Containing process
# It will be where all the layer-processes will be contained


class ContainingProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1024, ))
        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=(1022, ))


# ### Containing process model



@implements(proc=ContainingProcess, protocol=LoihiProtocol)
class ContainingProcessModel(AbstractSubProcessModel):
    def __init__(self, proc):

        self.p2 = P2()



        proc.in_ports.s_in.connect(self.p2.inp2)
        self.p2.s_out2.connect(proc.out_ports.s_out)

        """
        # connect in-port of the containing process to the in-port of the first process (inside containing process).
        proc.in_ports.s_in.connect(self.p2.inp2)
        # connect last-process output (inside containing process) to the output of the containing process.
        self.p2.s_out2.connect(proc.out_ports.s_out)
        """


# ### Running the SNN



from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_configs import Loihi1HwCfg
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps




num_steps_per_input = 100 #2*Ts
vth=5
step=1
bias=0
shape=(1024,) #shape
nsamples=1024

data=np.load('./data_tum._original.npy') #(4,1024), 4 cases, 1024 samples per case
sender1 = P1(shape=shape, bias=bias,num_steps=num_steps_per_input, vth=vth,data=data,nsamples=nsamples)
sender2 = ContainingProcess()
sender3 = P3()

# Connecting output port to an input port (of the containing process).
sender1.s_out.connect(sender2.s_in)
sender2.s_out.connect(sender3.inp3)

sender2.run(RunSteps(num_steps=num_steps_per_input), Loihi2HwCfg(select_tag='floating_pt', select_sub_proc_model=True))

