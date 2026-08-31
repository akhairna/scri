import numpy as np
import spherical as sf

import juliacall

Scri = juliacall.newmodule("Scri.jl")
Scri.seval("using Scri")
Scri.seval("using Quaternionic")

transform_bang = Scri.seval("transform!")


def _process_transformation_kwargs(**kwargs):
    # Build the supertranslation and spacetime_translation arrays
    supertranslation = np.zeros((4,), dtype=complex)  # For now; may be resized below
    ell_max_supertranslation = 1  # For now; may be increased below
    if "supertranslation" in kwargs:
        supertranslation = np.array(kwargs.pop("supertranslation"), dtype=complex)
        if supertranslation.dtype != "complex" and supertranslation.size > 0:
            # I don't actually think this can ever happen...
            raise TypeError(
                f"Input argument `supertranslation` should be a complex array with size>0. Got a {supertranslation.dtype} array of shape {supertranslation.shape}."
            )

        # Make sure the array has size at least 4, by padding with zeros
        if supertranslation.size <= 4:
            supertranslation = np.pad(
                supertranslation, (0, 4 - supertranslation.size), "constant", constant_values=(0.0,)
            )
        # Check that the shape is a possible array of scalar modes with complete (ell,m) data
        ell_max_supertranslation = int(np.sqrt(len(supertranslation))) - 1
        if (ell_max_supertranslation + 1) ** 2 != len(supertranslation):
            raise ValueError(
                f"Input supertranslation parameter must contain modes from ell=0 up to some ell_max, including all relevant m modes in standard order (see `spherical` documentation for details). Thus, it must be an array with length given by a perfect square; its length is {len(supertranslation)}."
            )

    spacetime_translation = np.zeros((4,), dtype=float)
    spacetime_translation[0] = sf.constant_from_ell_0_mode(supertranslation[0]).real
    spacetime_translation[1:4] = -sf.vector_from_ell_1_modes(supertranslation[1:4]).real

    if "spacetime_translation" in kwargs:
        st_trans = np.array(kwargs.pop("spacetime_translation"), dtype=float)
        if st_trans.shape != (4,) or st_trans.dtype != "float":
            raise TypeError(
                f"Input argument `spacetime_translation` should be a float array of shape (4,). Got a {st_trans.dtype} array of shape {st_trans.shape}."
            )

        spacetime_translation = st_trans[:]
        supertranslation[0] = sf.constant_as_ell_0_mode(spacetime_translation[0])
        supertranslation[1:4] = sf.vector_as_ell_1_modes(-spacetime_translation[1:4])
    if "space_translation" in kwargs:
        s_trans = np.array(kwargs.pop("space_translation"), dtype=float)
        if s_trans.shape != (3,) or s_trans.dtype != "float":
            raise TypeError(
                "\nInput argument `space_translation` should be an array of floats of shape (3,).\n"
                "Got a {} array of shape {}.".format(s_trans.dtype, s_trans.shape)
            )
        spacetime_translation[1:4] = s_trans[:]
        supertranslation[1:4] = sf.vector_as_ell_1_modes(-spacetime_translation[1:4])
    if "time_translation" in kwargs:
        t_trans = kwargs.pop("time_translation")
        if not isinstance(t_trans, float):
            raise TypeError("Input argument `time_translation` should be a single float.  " f"Got {t_trans}")
        spacetime_translation[0] = t_trans
        supertranslation[0] = sf.constant_as_ell_0_mode(spacetime_translation[0])

    # Get the rotor for the frame rotation
    frame_rotation = np.array(kwargs.pop("frame_rotation", [1, 0, 0, 0]), dtype=float)

    # Get the boost velocity vector
    boost_velocity = np.array(kwargs.pop("boost_velocity", [0.0] * 3), dtype=float)
    beta = np.linalg.norm(boost_velocity)
    if boost_velocity.dtype != float or boost_velocity.shape != (3,) or beta >= 1.0:
        raise ValueError(
            f"Input boost_velocity=`{boost_velocity}` should be a 3-vector with magnitude strictly less than 1.0."
        )

    return supertranslation, frame_rotation, boost_velocity


def transform(self, **kwargs):

    # Parse the input arguments, and define the basic parameters for this function
    supertranslation, frame_rotation, boost_velocity = _process_transformation_kwargs(**kwargs)

    v = Scri.quatvec(boost_velocity)
    R = Scri.rotor(frame_rotation)
    α = Scri.Vector(supertranslation)

    data = np.array(self._raw_data.T, dtype=np.complex128, order="F", copy=True)
    t = self.t
    ell_max = self.ell_max

    times = Scri.Vector(t)
    data_julia = Scri.Array(data)
    DataComponents = Scri.seval("Scri.DataComponents")
    data_components = DataComponents(*self.data_components)

    data_p, t_p = transform_bang(data_julia, times, v, R, α, data_components)
    data_prime = data_p.to_numpy()
    t_prime = t_p.to_numpy()

    abd_prime = type(self)(t_prime, ell_max)
    abd_prime._raw_data = data_prime

    return abd_prime
