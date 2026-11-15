"""Simulated users (the agents of the agent-based model).

Each user i has, for every item n:
    true value      x_{i,n} = V_{i,n} + beta * V_n
        V    ~ N(0,        sigma^2   * C)   common value, same for everyone
        V_i  ~ N(vbar_i,   sigma_i^2 * C)   idiosyncratic value
    prior mean      vbar_{i,n}
        circle world: drawn i.i.d. N(0, sigma_bar^2)       (paper)
        news world:   sigma_bar * zscore(cos(profile_i, article_n))
                      where profile_i = mean embedding of the user's real
                      MIND click history                    (our grounding)

The user does NOT know x. They hold Gaussian beliefs about it and learn by
consuming items (Bayesian updating, see Beliefs below).
"""
import numpy as np

from .params import Params
from .world import World


def _zscore(a: np.ndarray, axis=-1) -> np.ndarray:
    sd = a.std(axis=axis, keepdims=True)
    return (a - a.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1)


class Population:
    """Draws one population of U users in a world.

    profiles: [U, dim] user profile vectors (news world only)
    seen:     [U, N] bool, items each user already read before the simulation
    """

    def __init__(self, world: World, p: Params, U: int, rng: np.random.Generator,
                 profiles: np.ndarray | None = None, seen: np.ndarray | None = None):
        N = world.N
        self.U, self.N = U, N

        # common value V (one vector for the whole population)
        V = p.sigma * (world.L @ rng.standard_normal(N))
        if p.common_from_popularity and world.popularity is not None:
            V = V + p.sigma * _zscore(np.log1p(world.popularity))
        self.V = V

        # prior means
        if profiles is None:
            self.vbar = p.sigma_bar * rng.standard_normal((U, N))
        else:
            E = world.profile_vecs
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            P = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
            self.vbar = p.sigma_bar * _zscore(P @ E.T, axis=1)

        # idiosyncratic values and true values
        Vi = self.vbar + p.sigma_i * (rng.standard_normal((U, N)) @ world.L.T)
        self.X = Vi + p.beta * V[None, :]

        self.seen = np.zeros((U, N), bool) if seen is None else seen.copy()


class Beliefs:
    """Gaussian beliefs of every user about their own values x.

    Prior for user i:  x_i ~ N(mu0_i, s2 * C)
        info="none":   mu0 = vbar,            s2 = sigma_i^2 + beta^2 sigma^2
        info="rec":    mu0 = vbar + beta*V,   s2 = sigma_i^2
                       (the recommender knows V and tells the user, as in the paper)
        info="oracle": the user knows x exactly (mean = x, variance = 0)

    After consuming the set S and observing x_S, standard Gaussian conditioning:
        mean = mu0 + C[:,S] C[S,S]^-1 (x_S - mu0_S)
        var  = s2 * (1 - diag(C[:,S] C[S,S]^-1 C[S,:]))
    Because the prior covariance is s2 * C, the s2 cancels in the mean.
    This is how "learning spillovers" happen: observing x_S moves beliefs about
    every item correlated with S.
    """

    def __init__(self, world: World, pop: Population, p: Params):
        self.C = world.C
        self.info = p.info
        if p.info == "none":
            self.mu0 = pop.vbar.copy()
            self.s2 = p.sigma_i ** 2 + p.beta ** 2 * p.sigma ** 2
        elif p.info == "rec":
            self.mu0 = pop.vbar + p.beta * pop.V[None, :]
            self.s2 = p.sigma_i ** 2
        else:
            self.mu0 = pop.X.copy()
            self.s2 = 0.0
        self.X = pop.X
        self.S = [[] for _ in range(pop.U)]
        self.mean = self.mu0.copy()
        self.var = np.full_like(self.mu0, self.s2)

    def observe(self, i: int, n: int):
        """User i consumed item n and saw its true value; recompute beliefs."""
        self.S[i].append(n)
        if self.info == "oracle":
            return
        S = np.array(self.S[i])
        Css = self.C[np.ix_(S, S)] + 1e-9 * np.eye(len(S))
        A = np.linalg.solve(Css, self.C[S, :])            # [|S|, N]
        resid = self.X[i, S] - self.mu0[i, S]
        self.mean[i] = self.mu0[i] + A.T @ resid
        self.var[i] = np.clip(self.s2 * (1 - (self.C[S, :] * A).sum(0)), 0, None)

    def certainty_equivalent(self, gamma: float) -> np.ndarray:
        """CARA utility with Gaussian beliefs -> mean - gamma/2 * variance."""
        return self.mean - 0.5 * gamma * self.var
