import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy
import scipy.sparse
from scipy.integrate import romb
from scipy.special import binom
from scipy.stats.sampling import NumericalInversePolynomial

from kooplearn._src.utils import topk
from kooplearn.datasets.misc import (
    DataGenerator,
    DiscreteTimeDynamics,
    LinalgDecomposition,
)
import tqdm

from kooplearn.datasets.stochastic import LangevinTripleWell1D
class LangevinTripleWell1D(DiscreteTimeDynamics):
    def __init__(
        self,
        gamma: float = 0.1,
        kt: float = 1.0,
        dt: float = 1e-4,
        rng_seed: Optional[int] = None,
        height=0.1,
        sigma=0.05,
        freq=500
    ):
        self.gamma = gamma
        self.kt = kt
        self.rng = np.random.default_rng(rng_seed)
        self.dt = dt

        self._inv_gamma = (self.gamma) ** -1
        self.list_centers = []
        self.height = height
        self.sigma = sigma
        self.freq = freq
    def sample(self, X0: np.ndarray, T: int = 1, show_progress: bool = False):
        X0 = np.asarray(X0)
        if X0.ndim == 0:
            X0 = X0[None]

        memory = np.zeros((T + 1,) + X0.shape)
        meta_forces = np.zeros((T + 1,) + X0.shape)
        forces = np.zeros((T + 1,) + X0.shape)
        memory[0] = X0
        show_progress = True
        if show_progress:
            _iter = tqdm.tqdm(range(T), desc="Generating data", unit="step", leave=False)
        else:
            _iter = range(T)

        for t in _iter:
            if t% self.freq ==0 and t !=0:
                self.list_centers.append(memory[t])
            memory[t + 1], forces[t+1], meta_forces[t + 1] = self._step(memory[t])
        return memory, forces, meta_forces
    def eig(self):
        if not hasattr(self, "_ref_evd"):
            self._compute_ref_evd()
        return self._ref_evd
    def _compute_ref_evd(self):
        assets_path = Path(__file__).parent / "assets"

        lap_x = scipy.sparse.load_npz(assets_path / "1D_triple_well_lap_x.npz")
        grad_x = scipy.sparse.load_npz(assets_path / "1D_triple_well_grad_x.npz")
        cc = np.load(assets_path / "1D_triple_well_cc.npz")

        force = scipy.sparse.diags(self.force_fn(cc))

        # Eq.(31) of https://doi.org/10.1007/978-3-642-56589-2_9 recalling that \sigma^2/(\gamma*2) = kBT
        generator = self._inv_gamma * self.kt * lap_x + self._inv_gamma * force.dot(
            grad_x
        )
        generator = generator * self.dt

        vals, vecs = np.linalg.eig(generator.toarray())

        # Filter out timescales smaller than dt
        mask = np.abs(vals.real) < 1.0
        vals = vals[mask]
        vecs = vecs[:, mask]

        vals = np.exp(vals)
        # Checking that the eigenvalues are real
        type_ = vals.dtype.type
        f = np.finfo(type_).eps

        tol = f * 1000
        if not np.all(np.abs(vals.imag) < tol):
            raise ValueError(
                "The computed eigenvalues are not real, try to decrease dt"
            )
        else:
            vals = vals.real

        _k = len(vals)
        evd_sorting_perm = topk(vals, _k)
        vals = evd_sorting_perm.values
        vecs = vecs[:, evd_sorting_perm.indices].real

        dx = cc[1] - cc[0]
        boltzmann_pdf = (dx**-1) * scipy.special.softmax(self.force_fn(cc) / self.kt)
        abs2_eigfun = (np.abs(vecs) ** 2).T
        eigfuns_norms = np.sqrt(romb(boltzmann_pdf * abs2_eigfun, dx=dx, axis=-1))
        vecs = vecs * (eigfuns_norms**-1.0)

        # Storing the results
        self._ref_evd = LinalgDecomposition(vals, cc, vecs)
        self._ref_boltzmann_density = boltzmann_pdf

    def _step(self, X: np.ndarray):
        F, bias = self.force_fn(X)
        xi = self.rng.standard_normal(X.shape)
        dX = (
            F * self._inv_gamma * self.dt
            + np.sqrt(2.0 * self.kt * self.dt * self._inv_gamma) * xi
        )
        return X + dX, F, bias
    def meta_force(self,x):
        return  ((x - np.array(self.list_centers)) * self.height * np.exp( - ( x - np.array(self.list_centers))**2 / (2 * self.sigma **2))).sum() / (self.sigma**2)
    def force_fn(self, x: np.ndarray):
        bias = 0
        #f_four =  -20*x*(x**2-1) 
        f_four = -4.0 * (
            - 160 *  1.5 * np.exp(-80 * (x**2)) * x
            + 8 * (x**7)
        )
        f_two =-4.0 * (
            - 160 * 0.5* np.exp(-80 * (x**2)) * x
            + 8 * (x**7)
        ) #  -5
        return f_two, f_two-f_four #-8*(x-1)*(x+1)**2 - 8*(x-1)**2*(x+1)