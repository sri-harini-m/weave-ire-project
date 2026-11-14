from dataclasses import asdict, dataclass


@dataclass
class Params:

    T: int = 20
    sigma: float = 1.0
    sigma_i: float | None = None
    sigma_bar: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0
    info: str = "none"


    phi: float = 0.0
    fatigue_decay: float = 0.8
    fatigue_rho: float = 0.8
    fatigue_aware: bool = True
    outside_option: float | None = None


    uncertainty_n0: float | None = None
    true_history_size: int = 10
    min_user_history: int = 20
    reader_types_enabled: bool = False
    reader_type_proportions: dict | None = None
    reader_type_ranges: dict | None = None


    reader_history_visible: bool = False
    boredom_threshold: float = 0.25


    common_from_popularity: bool = False

    def __post_init__(self):
        if self.sigma_i is None:
            self.sigma_i = self.sigma
        assert self.info in ("none", "rec", "oracle"), self.info
        if not isinstance(self.reader_history_visible, bool):
            raise ValueError("reader_history_visible must be boolean")
        if not 0 <= self.boredom_threshold < float("inf"):
            raise ValueError("boredom_threshold must be finite and nonnegative")

    def as_dict(self) -> dict:
        return asdict(self)
