"""nsharness: a domain-neutral test harness for NeuroStore-style associative memories.

A *domain adapter* turns raw multichannel data into windows (n, T, C) with targets and groups; a *featurizer* maps
windows to fixed-length vectors; interchangeable *backends* (exact, ANN, FlyHash, Hopfield, VSA) store and retrieve
them; standardized *corruptions* and OOD/chimera constructions probe robustness and novelty detection; *bench* and
*accounting* report latency, throughput and memory. Pure NumPy; every random draw takes an explicit seed.
"""
__version__ = "0.1.0"
