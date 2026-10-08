import numpy as np
import spherical as sf
import functools
from .. import ModesTimeSeries
from ..asymptotic_bondi_data.transformations import _process_transformation_kwargs
import juliacall

Scri = juliacall.newmodule("Scri.jl")
Scri.seval("using Scri")
Scri.seval("using Quaternionic")

transform_bang = Scri.seval("transform!")
DataComponents = Scri.seval("Scri.DataComponents")

def transform(self, **kwargs):

    # Parse the input arguments, and define the basic parameters for this function
    frame_rotation, boost_velocity, supertranslation, working_ell_max, output_ell_max = _process_transformation_kwargs(self.ell_max, **kwargs)

    v = Scri.quatvec(boost_velocity)
    R = Scri.rotor(frame_rotation)
    α = Scri.Vector(supertranslation)

    data = np.array(self._raw_data.T, dtype=np.complex128, order="F", copy=True)
    t = self.t
    ell_max = self.ell_max
    fields_present = self.data_components

    times = Scri.Vector(t)
    data_julia = Scri.Array(data)
    data_components = DataComponents(*fields_present)

    data_p, t_p = transform_bang(data_julia, times, v, R, α, data_components)
    data_prime = np.array(data_p.to_numpy().T, dtype=np.complex128, order="C", copy=True) #No need to copy
    t_prime = t_p.to_numpy()

    ModesTS = functools.partial(ModesTimeSeries, ell_max=ell_max)

    spin_dict = {"psi0": 2, "psi1": 1, "psi2": 0,
                "psi3": -1, "psi4": -2, "sigma": 2}

    abd_prime = type(self)(t_prime, ell_max)

    for i, field in enumerate(fields_present):
        setattr(abd_prime, f"_{field}", ModesTS(data_prime[i], t_prime, spin_weight=spin_dict[field]))

    return abd_prime
