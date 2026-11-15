"""All parameters of one simulation run, in one place.

Names follow Aridor, Goncalves & Sikdar (2020) wherever the paper has them.
"""
from dataclasses import asdict, dataclass


@dataclass
class Params:
    # ---- Aridor et al. model -------------------------------------------
    T: int = 20               # items each user consumes
    sigma: float = 1.0        # std of the common value V
    sigma_i: float | None = None  # std of the idiosyncratic value V_i (None = sigma)
    sigma_bar: float = 1.0    # std of users' prior means (paper leaves unspecified)
    beta: float = 1.0         # weight of the common value: x = V_i + beta * V
    gamma: float = 0.0        # CARA risk aversion (0 = risk neutral)
    info: str = "none"        # "none" | "rec" | "oracle"  (paper's 3 regimes)

    # ---- Our extension: fatigue ------------------------------------------
    phi: float = 0.0          # fatigue strength (0 = paper's model)
    fatigue_decay: float = 0.8   # how much fatigue carries over each round
    fatigue_rho: float = 0.8  # similarity of a neighbour for fatigue purposes
    fatigue_aware: bool = True   # do users anticipate fatigue when choosing?
    outside_option: float | None = None  # leave if best option is worse than this

    # ---- News-only options -----------------------------------------------
    common_from_popularity: bool = False  # shift V's mean by real popularity

    def __post_init__(self):
        if self.sigma_i is None:
            self.sigma_i = self.sigma
        assert self.info in ("none", "rec", "oracle"), self.info

    def as_dict(self) -> dict:
        return asdict(self)
