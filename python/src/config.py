import dataclasses
import typing

from .base_config import YamlConfig


@dataclasses.dataclass
class Kohyama21ModelConfig(YamlConfig):
    tmax_year: float  # integration time [year]
    dt_year: float  # time step [year]
    istep_out: int  # interval of output
    ny: int  # grid number for y axis, including boundary points
    Lx_1: float  # zonal width of basin 1 [m]
    Lx_2: float
    H_1: float  # depth of basin 1 [m]
    H_2: float
    R_1: float  # deformation radius of basin 1 [m]
    R_2: float
    delta_1: float  # boundary current width of basin 1 [m]
    delta_2: float
    Lx: float  # zonal width of atmosphere [m]
    Ly: float  # meridional width of atmosphere [m]
    D_a: float  # scale height of atmosphere [m]
    lamd: float  # heat transfer coefficient between atmosphere and ocean [W m^2 / K]
    rho_a: float  # density of atmosphere at z = 0 [kg/m^3]
    rho_w: float  # density of seawater [kg/m^3]
    C_pa: float  # specific heat of atmosphere [J / (kg K)]
    C_pw: float  # specific heat of seawater [J / (kg K)]
    nu_a: float  # eddy relaxation rate of atmosphere at z = 0 [1/s]
    eps: float  # eddy diffusion coefficient of ocean [m^2/s]
    B: float  # coefficient of long wave radiation in atmosphere [W/m^2]
    F: float  # coefficient of short wave radiation in atmosphere [W/m^2]
    S_a: float  # static stability of atmosphere [K/m]
    f_0: float  # Coriolis parameter [1/s]
    beta: float  # beta effect parameter [1/(s m)]
    d_l: float  # vertical scale of  atmospheric eddy relaxation [m]
    amp_noise: float
    amp_noise_theta_1: float
    amp_noise_theta_2: float

    # Class constants
    year_to_sec: typing.ClassVar[float] = 60.0 * 60.0 * 24.0 * 365.0

    def __post_init__(self):
        assert self.nt % self.istep_out == 0

    @property
    def dt(self) -> float:
        return self.dt_year * self.year_to_sec

    @property
    def nt(self) -> int:
        return int(self.tmax_year / self.dt_year)
