import numpy as np
from spherical_functions import LM_total_size
from .. import WaveformModes
from .. import ModesTimeSeries
from .. import Inertial
from .. import h as h_DataType
import copy

class AsymptoticBondiData:
    """Class to store asymptotic Bondi data

    This class stores time data, along with the corresponding values of psi0 through psi4 and sigma.
    For simplicity, the data are stored as one contiguous array.  That is, *all* values are stored
    at all times, even if they are zero, and all Modes objects are stored with ell_min=0, even when
    their spins are not zero.

    The single contiguous array is then viewed as 6 separate ModesTimeSeries objects, which enables
    them to track their spin weights, and provides various convenient methods like `eth` and
    `ethbar`; `dot` and `ddot` for time-derivatives; `int` and `iint` for time-integrations; `norm`
    to take the norm of a function over the sphere; `bar` for conjugation of the functions (which is
    different from just conjugating the mode weights); etc.  It also handles algebra correctly --
    particularly addition (which is disallowed when the spin weights differ) and multiplication
    (which can be delicate with regards to the resulting ell values).

    This may lead to some headaches when the user tries to do things that are disabled by Modes
    objects.  The goal is to create headaches if and only if the user is trying to do things that
    really should never be done (like conjugating mode weights, rather than the underlying function;
    adding modes with different spin weights; etc.).  Please open issues for any situations that
    don't meet this standard.

    This class also provides various convenience methods for computing things like the mass aspect,
    the Bondi four-momentum, the Bianchi identities, etc.

    """

    def __init__(self, time, ell_max, multiplication_truncator=sum, frameType=Inertial):
        """Create new storage for asymptotic Bondi data

        Parameters
        ==========
        time: int or array_like
            Times at which the data will be stored.  If this is an int, an empty array of that size
            will be created.  Otherwise, this must be a 1-dimensional array of floats.
        ell_max: int
            Maximum ell value to be stored
        multiplication_truncator: callable [defaults to `sum`, even though `max` is nicer]
            Function to be used by default when multiplying Modes objects together.  See the
            documentation for spherical_functions.Modes.multiply for more details.  The default
            behavior with `sum` is the most correct one -- keeping all ell values that result -- but
            also the most wasteful, and very likely to be overkill.  The user should probably always
            use `max`.  (Unfortunately, this must remain an opt-in choice, to ensure that the user
            is aware of the situation.)

        """
        import functools

        if np.ndim(time) == 0:
            # Assume this is just the size of the time array; construct an empty array
            time = np.empty((time,), dtype=float)
        elif np.ndim(time) > 1:
            raise ValueError(f"Input `time` parameter must be an integer or a 1-d array; it has shape {time.shape}")
        if time.dtype != float:
            raise ValueError(f"Input `time` parameter must have dtype float; it has dtype {time.dtype}")
        ModesTS = functools.partial(ModesTimeSeries, ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        shape = [6, time.size, LM_total_size(0, ell_max)]
        self.frame = np.array([])
        self.frameType = frameType
        self._time = time.copy()
        self._raw_data = np.zeros(shape, dtype=complex)
        self._psi0 = ModesTS(self._raw_data[0], self._time, spin_weight=2)
        self._psi1 = ModesTS(self._raw_data[1], self._time, spin_weight=1)
        self._psi2 = ModesTS(self._raw_data[2], self._time, spin_weight=0)
        self._psi3 = ModesTS(self._raw_data[3], self._time, spin_weight=-1)
        self._psi4 = ModesTS(self._raw_data[4], self._time, spin_weight=-2)
        self._sigma = ModesTS(self._raw_data[5], self._time, spin_weight=2)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        self._time[:] = new_time
        return self._time

    u = time

    t = time

    @property
    def n_times(self):
        return self.time.size

    @property
    def n_modes(self):
        return self._raw_data.shape[-1]

    @property
    def ell_min(self):
        return self._psi2.ell_min

    @property
    def ell_max(self):
        return self._psi2.ell_max

    @property
    def LM(self):
        return self.psi2.LM

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigmaprm):
        self._sigma[:] = sigmaprm
        return self.sigma

    @property
    def has_sigma(self):
        return self._sigma is not None

    @property
    def h(self):
        h_mts = 2.0 * self._sigma.bar
        return WaveformModes(
            t=h_mts.t,
            data=np.array(h_mts)[:, h_mts.index(abs(h_mts.s), -abs(h_mts.s)) :],
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            frameType=Inertial,
            dataType=h_DataType,
            r_is_scaled_out=True,
            m_is_scaled_out=True,
        )

    @property
    def psi4(self):
        return self._psi4

    @psi4.setter
    def psi4(self, psi4prm):
        self._psi4[:] = psi4prm
        return self.psi4

    @property
    def has_psi4(self):
        return self._psi4 is not None

    @property
    def psi3(self):
        return self._psi3

    @psi3.setter
    def psi3(self, psi3prm):
        self._psi3[:] = psi3prm
        return self.psi3

    @property
    def has_psi3(self):
        return self._psi3 is not None

    @property
    def psi2(self):
        return self._psi2

    @psi2.setter
    def psi2(self, psi2prm):
        self._psi2[:] = psi2prm
        return self.psi2

    @property
    def has_psi2(self):
        return self._psi2 is not None

    @property
    def psi1(self):
        return self._psi1

    @psi1.setter
    def psi1(self, psi1prm):
        self._psi1[:] = psi1prm
        return self.psi1

    @property
    def has_psi1(self):
        return self._psi1 is not None

    @property
    def psi0(self):
        return self._psi0

    @psi0.setter
    def psi0(self, psi0prm):
        self._psi0[:] = psi0prm
        return self.psi0

    @property
    def has_psi0(self):
        return self._psi0 is not None

    @property
    def data_components(self):
        """Fields that are present in the instance."""

        fields = ["psi0", "psi1", "psi2", "psi3", "psi4", "sigma"]
        fields_present = [field for field in fields if getattr(self, f"has_{field}")]

        return fields_present

    #Slicing
    def __getitem__(self, key):
        """
        Extract time slices of the asymptotic Bondi data efficiently.
        """
        # If key is a valid time slice or index, extract the corresponding
        # sliced data
        if not isinstance(key, (slice, int)):
            raise ValueError(f"Invalid key `{key}` of type `{type(key)}`.")

        import functools

        ModesTS = functools.partial(ModesTimeSeries, ell_max=self.ell_max, multiplication_truncator=self.sigma._metadata['multiplication_truncator'])

        new_abd = self.copy()
        new_abd._raw_data = self._raw_data[:, key, :]
        new_abd._time = self.time[key]

        for i, component in enumerate(self.data_components):
            sliced_component = ModesTS(new_abd._raw_data[i], new_abd._time, spin_weight = getattr(new_abd, component).spin_weight)
            setattr(new_abd, f"_{component}", sliced_component)

        if self.frame.shape[0] == self.n_times:
            new_abd.frame = self.frame[key]
        return new_abd

    def copy(self):
        import copy

        new_abd = type(self)(self.t, self.ell_max)
        state = copy.deepcopy(self.__dict__)
        new_abd.__dict__.update(state)
        return new_abd

    def interpolate(self, new_times):
        new_abd = self.copy()
        new_abd._time = new_times
        # interpolate waveform data
        for field in new_abd.data_components:
            interpolated_fields = getattr(self, field).interpolate(new_times)
            setattr(new_abd, f"_{field}", interpolated_fields)

        # interpolate frame data if necessary
        if self.frame.shape[0] == self.n_times:
            import quaternion

            new_abd.frame = quaternion.squad(self.frame, self.t, new_times)
        return new_abd

    from .from_initial_values import from_initial_values

    from .constraints import (
        bondi_constraints,
        bondi_violations,
        bondi_violation_norms,
        bianchi_0,
        bianchi_1,
        bianchi_2,
        constraint_3,
        constraint_4,
        constraint_mass_aspect,
    )

    from .transformations import transform
    from .bms_charges import (
        mass_aspect,
        bondi_rest_mass,
        bondi_four_momentum,
        bondi_angular_momentum,
        CWWY_angular_momentum,
        bondi_dimensionless_spin,
        bondi_boost_charge,
        bondi_CoM_charge,
        supermomentum,
    )

    from .map_to_superrest_frame import map_to_superrest_frame
    from .map_to_abd_frame import map_to_abd_frame

class AsymptoticBondiData_v1(AsymptoticBondiData):
    """Asymptotic Bondi data with a field based constructor.
    """

    def __init__(self, time=None, ell_max=None, psi0=None, psi1=None, psi2=None, psi3=None, psi4=None, sigma=None, frameType=Inertial, multiplication_truncator=max, **kwargs):
        """
        Parameters
        ----------
        time: array_like, optional
            Times at which the data will be stored.
        ell_max: int, optional
            Maximum ell value to be stored

        psi0, psi1, psi2, psi3, psi4 : scri.ModesTimeSeries, optional
        Mode data for the Weyl scalars Ψ_0 ... Ψ_4. If Ψ_i is given, all
        higher-index scalars Ψ_j (j > i) must also be given.

        frameType : scri.FrameType, optional
            Frame in which the data is expressed. Default is ``Inertial``.
        multiplication_truncator : callable, optional
            Function that sets the ell_max of a product of two mode series, e.g.
            ``max`` or ``sum``. Default is ``max``.
        """
        self._time = time
        self._ell_max = ell_max
        fields_dict = {"psi0": (psi0, 2), "psi1": (psi1, 1), "psi2": (psi2, 0),
                  "psi3": (psi3, -1), "psi4": (psi4, -2), "sigma": (sigma, 2)}

        for field, (value, spin_weight) in fields_dict.items():
            setattr(self, f"_{field}",
                    ModesTimeSeries(value.data, time=value.time, ell_min=value.ell_min, ell_max=value.ell_max, spin_weight=spin_weight, multiplication_truncator=multiplication_truncator) if value is not None else None)

        self.frameType = frameType
        self.frame = kwargs.pop("frame", np.array([]))

        self.validate_fields()
        self.validate_times()
        self.construct_raw_data()

    @property
    def ell_max(self):
        """ell_max is either obtained from the kwarg or is set to the minimum
        value of ell_max among all the fields provided."""
        if self.data_components:
            self._ell_max = min([getattr(self, f"{field}").ell_max for field in self.data_components])
        return self._ell_max

    def validate_fields(self):
        """Check if the input fields are sensible."""
        fields = ["psi0", "psi1", "psi2", "psi3", "psi4"]
        fields_present = [True if field in self.data_components else False for field in fields]

        for i, (name, present) in enumerate(zip(fields, fields_present)):
            if present:
                missing = [f for f, p in zip(fields[i + 1 :], fields_present[i + 1 :]) if not p]
                if missing:
                    raise ValueError(f"{name} is present but higher-order Weyl scalars are missing: {missing}")

    def validate_times(self):
        """Check if the input time and/or the time arrays of the fields mutually
        agree with each other.
        """

        present_fields = self.data_components
        t_ref = self.t if self.t is not None else self._sigma.t

        for field in present_fields:
            if getattr(self, f"has_{field}"):
                if not np.array_equal(t_ref, getattr(self, field).t):
                    raise ValueError(
                        f"All fields i.e. Strain and Weyl scalar components must share the same set of times."
                        f"The data for {field} has a different set of times."
                    )

        self._time = t_ref

    def construct_raw_data(self):
        n_modes = LM_total_size(0, self.ell_max)
        n_times = self.n_times
        n_components = len(self.data_components)

        shape = [n_components, n_times, n_modes]
        self._raw_data = np.zeros(shape, dtype=complex)

        for i, field in enumerate(self.data_components):
            self._raw_data[i,:,:] = getattr(self, field).view(np.ndarray)
