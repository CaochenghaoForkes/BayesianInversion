"""Lightweight semi-analytical release model for three spherical shells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq


GAS_CONSTANT = 8.31446261815324  # J/(mol K)
CM2_TO_M2 = 1.0e-4


@dataclass(frozen=True)
class SphericalShellSettings:
    """Fixed geometry, boundary condition, and numerical settings."""

    inner_radius: float = 340.0e-6
    thickness_ipyc: float = 40.0e-6
    thickness_sic: float = 35.0e-6
    thickness_opyc: float = 40.0e-6
    mass_transfer_coefficient: float = 1.0e-4
    source_concentration: float = 1.0
    number_of_modes: int = 48
    quadrature_order: int = 64
    samples_per_mode: int = 80


DEFAULT_SETTINGS = SphericalShellSettings()


@lru_cache(maxsize=None)
def _gauss_legendre_rule(order: int) -> tuple[np.ndarray,np.ndarray]:
    """Return immutable Gauss–Legendre nodes and weights for one order."""

    nodes,weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes,weights


@dataclass(frozen=True)
class ModalReleaseSolution:
    """Eigenvalues and coefficients needed by the release-rate series."""

    eigenvalues: np.ndarray
    release_coefficients: np.ndarray


def arrhenius_diffusivity(
    pre_exponential: float,
    activation_energy: float,
    temperature: float,
) -> float:
    """Convert input D0 in cm^2/s to D(T) in m^2/s for meter-based geometry."""

    pre_exponential = float(pre_exponential)
    activation_energy = float(activation_energy)
    temperature = float(temperature)
    if pre_exponential <= 0.0 or not np.isfinite(pre_exponential):
        raise ValueError("扩散前因子 D0 必须为有限正数")
    if activation_energy < 0.0 or not np.isfinite(activation_energy):
        raise ValueError("扩散活化能 A 必须为有限非负数")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("绝对温度必须为有限正数")

    diffusivity = pre_exponential*CM2_TO_M2*np.exp(
        -activation_energy/(GAS_CONSTANT*temperature)
    )
    if diffusivity <= 0.0 or not np.isfinite(diffusivity):
        raise ValueError("Arrhenius 公式得到的扩散系数不是有限正数")
    return float(diffusivity)


class ThreeLayerSphericalShellSolver:
    """Compute radial release from three concentric spherical shells."""

    def __init__(
        self,
        diffusivity_pyc: float,
        diffusivity_sic: float,
        settings: SphericalShellSettings = DEFAULT_SETTINGS,
    ):
        self.settings = settings
        self.inner_radius = float(settings.inner_radius)
        self.thicknesses = np.asarray(
            [
                settings.thickness_ipyc,
                settings.thickness_sic,
                settings.thickness_opyc,
            ],
            dtype=np.float64,
        )
        self.diffusivities = np.asarray(
            [diffusivity_pyc,diffusivity_sic,diffusivity_pyc],
            dtype=np.float64,
        )
        self.beta = float(settings.mass_transfer_coefficient)
        self.source_concentration = float(settings.source_concentration)

        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius 必须为正")
        if np.any(self.thicknesses <= 0.0):
            raise ValueError("三层厚度必须为正")
        if np.any(self.diffusivities <= 0.0) or not np.all(
            np.isfinite(self.diffusivities)
        ):
            raise ValueError("PyC 和 SiC 扩散系数必须为有限正数")
        if self.beta <= 0.0 or self.source_concentration <= 0.0:
            raise ValueError("传质系数和内边界浓度必须为正")

        self.interfaces = np.concatenate(
            ([self.inner_radius],self.inner_radius+np.cumsum(self.thicknesses))
        )

    @property
    def outer_radius(self) -> float:
        """Return the outer radius of the third shell."""

        return float(self.interfaces[-1])

    @property
    def steady_total_release_rate(self) -> float:
        """Return the steady total release rate in mol/s."""

        shell_resistance = np.sum(
            (1.0/self.interfaces[:-1]-1.0/self.interfaces[1:])
            /self.diffusivities
        )
        film_resistance = 1.0/(self.beta*self.outer_radius**2)
        return float(
            4.0*np.pi*self.source_concentration
            /(shell_resistance+film_resistance)
        )

    @property
    def steady_flux(self) -> float:
        """Return the steady outer-surface release rate per unit area."""

        return self.steady_total_release_rate/(4.0*np.pi*self.outer_radius**2)

    def _steady_concentration(self,radii: np.ndarray) -> np.ndarray:
        """Evaluate the piecewise A+B/r steady concentration."""

        result = np.empty_like(radii)
        rate_per_solid_angle = self.steady_total_release_rate/(4.0*np.pi)
        resistance_before = 0.0
        for layer in range(3):
            left,right = self.interfaces[layer:layer+2]
            mask = (radii >= left) & (
                (radii < right) if layer < 2 else (radii <= right)
            )
            local_resistance = (
                1.0/left-1.0/radii[mask]
            )/self.diffusivities[layer]
            result[mask] = self.source_concentration-rate_per_solid_angle*(
                resistance_before+local_resistance
            )
            resistance_before += (
                1.0/left-1.0/right
            )/self.diffusivities[layer]
        return result

    @staticmethod
    def _layer_matrix(
        omega: float,
        diffusivity: float,
        radius_left: float,
        radius_right: float,
    ) -> np.ndarray:
        """Propagate [phi,D*phi'] through one homogeneous shell."""

        distance = radius_right-radius_left
        wave_number = omega/np.sqrt(diffusivity)
        phase = wave_number*distance
        cosine = np.cos(phase)
        sine = np.sin(phase)
        sine_over_wave_number = distance*np.sinc(phase/np.pi)
        m11 = (radius_left*cosine+sine_over_wave_number)/radius_right
        m12 = radius_left*sine_over_wave_number/(diffusivity*radius_right)
        m21 = diffusivity/radius_right**2*(
            distance*cosine-sine_over_wave_number
            -radius_left*radius_right*wave_number*sine
        )
        m22 = radius_left/radius_right*(
            cosine-sine_over_wave_number/radius_right
        )
        return np.array([[m11,m12],[m21,m22]],dtype=np.float64)

    @staticmethod
    def _layer_matrices(
        omegas: np.ndarray,
        diffusivity: float,
        radius_left: float,
        radius_right: float,
    ) -> np.ndarray:
        """Vectorized propagation matrices for an omega grid."""

        distance = radius_right-radius_left
        wave_numbers = omegas/np.sqrt(diffusivity)
        phases = wave_numbers*distance
        cosine = np.cos(phases)
        sine = np.sin(phases)
        sine_over_wave_number = distance*np.sinc(phases/np.pi)
        matrices = np.empty((omegas.size,2,2),dtype=np.float64)
        matrices[:,0,0] = (
            radius_left*cosine+sine_over_wave_number
        )/radius_right
        matrices[:,0,1] = (
            radius_left*sine_over_wave_number/(diffusivity*radius_right)
        )
        matrices[:,1,0] = diffusivity/radius_right**2*(
            distance*cosine-sine_over_wave_number
            -radius_left*radius_right*wave_numbers*sine
        )
        matrices[:,1,1] = radius_left/radius_right*(
            cosine-sine_over_wave_number/radius_right
        )
        return matrices

    def _characteristic(self,omega: float) -> float:
        """Evaluate the outer Robin-boundary characteristic equation."""

        # 原始完整传递矩阵实现（保留供数值对比）：
        # matrix = np.eye(2)
        # for layer in range(3):
        #     matrix = self._layer_matrix(
        #         omega,
        #         self.diffusivities[layer],
        #         self.interfaces[layer],
        #         self.interfaces[layer+1],
        #     )@matrix
        # return float(matrix[1,1]+self.beta*matrix[0,1])

        phi = 0.0
        diffusive_flux = 1.0
        for layer in range(3):
            diffusivity = float(self.diffusivities[layer])
            radius_left = float(self.interfaces[layer])
            radius_right = float(self.interfaces[layer+1])
            distance = radius_right-radius_left
            wave_number = omega/math.sqrt(diffusivity)
            phase = wave_number*distance
            cosine = math.cos(phase)
            sine = math.sin(phase)
            if abs(phase) < 1.0e-4:
                phase_squared = phase*phase
                sine_over_wave_number = distance*(
                    1.0-phase_squared/6.0
                    +phase_squared*phase_squared/120.0
                )
            else:
                sine_over_wave_number = sine/wave_number

            m11 = (
                radius_left*cosine+sine_over_wave_number
            )/radius_right
            m12 = (
                radius_left*sine_over_wave_number
                /(diffusivity*radius_right)
            )
            m21 = diffusivity/radius_right**2*(
                distance*cosine-sine_over_wave_number
                -radius_left*radius_right*wave_number*sine
            )
            m22 = radius_left/radius_right*(
                cosine-sine_over_wave_number/radius_right
            )
            phi,diffusive_flux = (
                m11*phi+m12*diffusive_flux,
                m21*phi+m22*diffusive_flux,
            )

        return diffusive_flux+self.beta*phi

    def _characteristic_values(self,omegas: np.ndarray) -> np.ndarray:
        """Evaluate the characteristic equation on an omega grid."""

        values = np.asarray(omegas,dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("omegas 必须是一维数组")
        matrices = np.broadcast_to(
            np.eye(2,dtype=np.float64),(values.size,2,2)
        ).copy()
        for layer in range(3):
            matrices = self._layer_matrices(
                values,
                self.diffusivities[layer],
                self.interfaces[layer],
                self.interfaces[layer+1],
            )@matrices
        return matrices[:,1,1]+self.beta*matrices[:,0,1]

    def _eigenvalues(self) -> np.ndarray:
        """Find the requested positive eigenvalues with bracketed solves."""

        travel_time = np.sum(self.thicknesses/np.sqrt(self.diffusivities))
        spacing = np.pi/travel_time
        omega_max = (self.settings.number_of_modes+4.0)*spacing

        for _ in range(8):
            grid_size = int(
                self.settings.samples_per_mode*omega_max/spacing
            )+1
            grid = np.linspace(spacing*1.0e-10,omega_max,grid_size)
            values = self._characteristic_values(grid)
            roots: list[float] = []

            for left,right,f_left,f_right in zip(
                grid[:-1],grid[1:],values[:-1],values[1:],strict=True
            ):
                if not (np.isfinite(f_left) and np.isfinite(f_right)):
                    continue
                if np.signbit(f_left) == np.signbit(f_right):
                    continue
                root = brentq(
                    self._characteristic,
                    left,
                    right,
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
                if not roots or not np.isclose(
                    root,roots[-1],rtol=1.0e-10,atol=1.0e-13
                ):
                    roots.append(root)
                if len(roots) == self.settings.number_of_modes:
                    return np.square(np.asarray(roots))
            omega_max *= 1.7

        raise RuntimeError("没有找到足够多的三层球壳特征根")

    def _left_states(self,omega: float) -> list[np.ndarray]:
        """Return [phi,D*phi'] at the left edge of every shell."""

        states = [np.array([0.0,1.0])]
        for layer in range(2):
            states.append(
                self._layer_matrix(
                    omega,
                    self.diffusivities[layer],
                    self.interfaces[layer],
                    self.interfaces[layer+1],
                )@states[-1]
            )
        return states

    def _phi_local(
        self,
        radii,
        layer: int,
        omega: float,
        left_state: np.ndarray,
    ):
        """Evaluate one eigenfunction inside one spherical shell."""

        values = np.asarray(radii,dtype=np.float64)
        radius_left = self.interfaces[layer]
        diffusivity = self.diffusivities[layer]
        distance = values-radius_left
        wave_number = omega/np.sqrt(diffusivity)
        phase = wave_number*distance
        sine_over_wave_number = distance*np.sinc(phase/np.pi)
        w_left = radius_left*left_state[0]
        dw_flux_left = diffusivity*left_state[0]+radius_left*left_state[1]
        w = (
            w_left*np.cos(phase)
            +dw_flux_left/diffusivity*sine_over_wave_number
        )
        result = w/values
        return float(result) if result.ndim == 0 else result

    def _modal_solution(self) -> ModalReleaseSolution:
        """Project the initial condition using the spherical r^2 weight."""

        eigenvalues = self._eigenvalues()
        nodes,weights = _gauss_legendre_rule(self.settings.quadrature_order)
        release_coefficients = np.empty(eigenvalues.size)

        quadrature_data = []
        for layer in range(3):
            left,right = self.interfaces[layer:layer+2]
            radii = 0.5*((right-left)*nodes+right+left)
            local_weights = 0.5*(right-left)*weights
            spherical_weights = local_weights*radii**2
            steady_concentration = self._steady_concentration(radii)
            quadrature_data.append(
                (radii,spherical_weights,steady_concentration)
            )

        for mode,eigenvalue in enumerate(eigenvalues):
            omega = np.sqrt(eigenvalue)
            states = self._left_states(omega)
            numerator = 0.0
            denominator = 0.0

            for layer,(
                radii,
                spherical_weights,
                steady_concentration,
            ) in enumerate(quadrature_data):
                phi = self._phi_local(
                    radii,layer,omega,states[layer]
                )
                numerator += np.sum(
                    spherical_weights*steady_concentration*phi
                )
                denominator += np.sum(spherical_weights*phi*phi)

            coefficient = -numerator/denominator
            surface_mode = self._phi_local(
                self.outer_radius,2,omega,states[2]
            )
            release_coefficients[mode] = self.beta*coefficient*surface_mode

        return ModalReleaseSolution(eigenvalues,release_coefficients)

    def release_rate(self,times) -> np.ndarray:
        """Return the outer-surface release rate per unit area."""

        times = np.asarray(times,dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times 必须是非空一维数组")
        if np.any(times < 0.0) or not np.all(np.isfinite(times)):
            raise ValueError("times 必须包含有限的非负时刻")

        solution = self._modal_solution()
        transient = np.exp(
            -times[:,None]*solution.eigenvalues[None,:]
        )@solution.release_coefficients
        release = np.maximum(self.steady_flux+transient,0.0)
        release[times == 0.0] = 0.0
        return release
