import sys
import scri
import sxs
import numpy as np
from matplotlib import pyplot as plt
from scri import flux
from scri import waveform_base


SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0023/"
Sim_path=""
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir"
md = sxs.metadata.Metadata.from_file(SA_path + Sim_path + "metadata.txt")    
h = scri.SpEC.read_from_h5(SA_path + Sim_path + Waveform_path)
M = md['remnant_mass']

x_1 = [0.5, 0.0, 0.0]

h1 = h.transform(space_translation=x_1)

A = scri.momentum_flux(h1);
B = scri.momentum_flux(h.interpolate(h1.t));

C = A - B
D = A + B 

x = h1.t
y = np.log(np.abs(C/D))[:,0]
plt.title("log fractional error of flux in SXS:BBH:0023") 
plt.xlabel("Time") 
plt.plot(x,y)
plt.show()

