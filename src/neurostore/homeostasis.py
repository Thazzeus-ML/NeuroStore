"""Homeostasis: per-unit adaptive thresholds move toward a target firing rate; the global budget
(exactly k winners per event) is enforced by the encoder's top-k, so k never changes."""
from __future__ import annotations
import numpy as np


class Homeostasis:
    def __init__(self, enc, alpha=0.02, eta=0.002, clip=2.0):
        self.enc, self.alpha, self.eta, self.clip = enc, alpha, eta, clip
        self.target = enc.k / enc.N
        self.rate = np.full(enc.N, self.target, np.float32)     # EMA of P(unit fires per event)

    def update(self, units):
        self.rate *= (1.0 - self.alpha)
        self.rate[units] += self.alpha
        th = self.enc.theta
        th += self.eta * (self.rate - self.target) / self.target
        np.clip(th, -self.clip, self.clip, out=th)
