import numpy as np

from .params import Params
from .world import World


READER_TYPES = {
    "cautious": {"gamma": 2.0, "phi": 0.5, "fatigue_decay": 0.8},
    "explorer": {"gamma": 0.0, "phi": 0.2, "fatigue_decay": 0.8},
    "easily_bored": {"gamma": 0.5, "phi": 1.5, "fatigue_decay": 0.6},
    "average": {"gamma": 1.0, "phi": 0.5, "fatigue_decay": 0.8},
}


def _zscore(a: np.ndarray, axis=-1) -> np.ndarray:
    sd = a.std(axis=axis, keepdims=True)
    return (a - a.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1)


class Population:


    def __init__(self, world: World, p: Params, U: int, rng: np.random.Generator,
                 profiles: np.ndarray | None = None, seen: np.ndarray | None = None,
                 true_profiles: np.ndarray | None = None,
                 history_lens: np.ndarray | None = None):
        N = world.N
        self.U, self.N = U, N
        self.history_lens = history_lens


        V = p.sigma * (world.L @ rng.standard_normal(N))
        if p.common_from_popularity and world.popularity is not None:
            V = V + p.sigma * _zscore(np.log1p(world.popularity))
        self.V = V


        if profiles is None:
            self.vbar = p.sigma_bar * rng.standard_normal((U, N))
        else:
            E = world.profile_vecs
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            P = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
            self.vbar = p.sigma_bar * _zscore(P @ E.T, axis=1)


        if true_profiles is not None and world.profile_vecs is not None:
            E = world.profile_vecs
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            Pt = true_profiles / np.linalg.norm(true_profiles, axis=1, keepdims=True)
            self.v_true = p.sigma_bar * _zscore(Pt @ E.T, axis=1)
        else:
            self.v_true = self.vbar.copy()


        Vi = self.v_true + p.sigma_i * (rng.standard_normal((U, N)) @ world.L.T)
        self.X = Vi + p.beta * V[None, :]

        self.seen = np.zeros((U, N), bool) if seen is None else seen.copy()


        self._assign_reader_types(p, U, rng)

    def _assign_reader_types(self, p: Params, U: int, rng: np.random.Generator):
        if p.reader_types_enabled:
            if p.reader_type_ranges:

                g_min, g_max = p.reader_type_ranges.get("gamma", (0.0, 2.0))
                p_min, p_max = p.reader_type_ranges.get("phi", (0.1, 1.5))
                d_min, d_max = p.reader_type_ranges.get("fatigue_decay", (0.5, 0.9))
                self.gammas = rng.uniform(g_min, g_max, size=U)
                self.phis = rng.uniform(p_min, p_max, size=U)
                self.decays = rng.uniform(d_min, d_max, size=U)
                self.reader_types = ["custom"] * U
            else:

                type_names = list(READER_TYPES.keys())
                if p.reader_type_proportions:
                    probs = [p.reader_type_proportions.get(t, 0.0) for t in type_names]
                    probs = np.array(probs, dtype=float)
                    probs /= probs.sum()
                else:
                    probs = np.ones(len(type_names)) / len(type_names)

                chosen_types = rng.choice(type_names, size=U, p=probs)
                self.reader_types = list(chosen_types)
                self.gammas = np.array([READER_TYPES[t]["gamma"] for t in chosen_types], dtype=float)
                self.phis = np.array([READER_TYPES[t]["phi"] for t in chosen_types], dtype=float)
                self.decays = np.array([READER_TYPES[t]["fatigue_decay"] for t in chosen_types], dtype=float)
        else:

            self.reader_types = ["standard"] * U
            self.gammas = np.full(U, p.gamma, dtype=float)
            self.phis = np.full(U, p.phi, dtype=float)
            self.decays = np.full(U, p.fatigue_decay, dtype=float)


class Beliefs:


    def __init__(self, world: World, pop: Population, p: Params):
        self.C = world.C
        self.info = p.info
        self.U, self.N = pop.U, world.N

        if p.info == "none":
            self.mu0 = pop.vbar.copy()
            s2_base = p.sigma_i ** 2 + p.beta ** 2 * p.sigma ** 2
        elif p.info == "rec":
            self.mu0 = pop.vbar + p.beta * pop.V[None, :]
            s2_base = p.sigma_i ** 2
        else:
            self.mu0 = pop.X.copy()
            s2_base = 0.0

        self.s2 = float(s2_base)
        if p.uncertainty_n0 is not None and pop.history_lens is not None and s2_base > 0:
            n0 = float(p.uncertainty_n0)
            self.user_s2 = s2_base * (n0 / (n0 + np.asarray(pop.history_lens, dtype=float)))
            self.var = np.repeat(self.user_s2[:, None], world.N, axis=1)
        else:
            self.user_s2 = None
            if p.info == "oracle":
                self.var = np.zeros((pop.U, world.N), dtype=float)
            else:
                self.var = np.full_like(self.mu0, self.s2)

        self.X = pop.X
        self.S = [[] for _ in range(pop.U)]
        self.mean = self.mu0.copy()

    def observe(self, i: int, n: int):

        self.S[i].append(n)
        if self.info == "oracle":
            return
        S = np.array(self.S[i])
        Css = self.C[np.ix_(S, S)] + 1e-9 * np.eye(len(S))
        A = np.linalg.solve(Css, self.C[S, :])
        resid = self.X[i, S] - self.mu0[i, S]
        self.mean[i] = self.mu0[i] + A.T @ resid
        user_s2 = self.user_s2[i] if self.user_s2 is not None else self.s2
        self.var[i] = np.clip(user_s2 * (1 - (self.C[S, :] * A).sum(0)), 0, None)

    def certainty_equivalent(self, gamma: float | np.ndarray) -> np.ndarray:


        g = np.asarray(gamma)
        if g.ndim == 0:
            return self.mean - 0.5 * float(g) * self.var
        return self.mean - 0.5 * g.reshape(-1, 1) * self.var
