"""Every constant of the PdM study in one place. PROTOCOL.md quotes these values; changing any of them after the protocol
commit is a protocol deviation and must be listed in the report."""

# ---- data and preprocessing -------------------------------------------------------------------------------------
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
PRIMARY = "FD001"
WINDOW = 30            # cycles per window (left-padded with the first cycle)
CAP = 125              # piecewise-linear RUL cap used for stored targets
MIN_UNIQUE = 10        # a channel is kept iff it takes >= MIN_UNIQUE distinct training values in every operating regime
FEAT = {"kind": "stats", "standardize": True}      # last / mean / slope / std per channel, then per-feature z-score; no PCA

# ---- randomness -------------------------------------------------------------------------------------------------
SEEDS = (0, 1, 2)                  # model seeds for the stochastic backends (ann, ann_sq8, flyhash, vsa)
CV_SEED = 0
CORRUPTION_SEEDS = (0, 1)          # evaluation realisations of every corruption
CALIB_SEED = 100                   # corruption seed used only to calibrate severities

# ---- engine-grouped cross-validation (training engines only) -------------------------------------------------------
CV_FOLDS = 5
CV_PER_GROUP = 10                  # random truncation points per held-out engine
CV_MIN_T = 30                      # a query window must end at cycle >= CV_MIN_T

# ---- hyper-parameter grids ----------------------------------------------------------------------------------------
BETAS = (None, 3.0, 10.0, 30.0, 100.0, 300.0)
KS = (1, 3, 5, 10, 20, 50)
READOUT_GRID = [(k, b) for k in KS for b in BETAS]                     # exact, ann, flyhash, vsa
HOPFIELD_BETAS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)       # attention over every stored pattern (k is not used)
STRUCT_GRID = {
    "exact": [{}],
    "exact_l2": [{}],
    "hopfield": [{}],
    "ann": [{"nprobe": p, "sq8": False} for p in (1, 2, 4, 8, 16)],
    "ann_sq8": [{"nprobe": p, "sq8": True} for p in (1, 2, 4, 8, 16)],
    "flyhash": [{"m": m, "frac": f} for m in (1024, 2048, 4096) for f in (0.02, 0.05, 0.10)],
    "vsa": [{"D": D} for D in (2048, 4096)],
}
BACKEND_OF = {"exact": "exact", "exact_l2": "exact_l2", "hopfield": "hopfield", "ann": "ann", "ann_sq8": "ann", "flyhash": "flyhash", "vsa": "vsa"}
REFERENCE_CANDIDATES = ("exact", "exact_l2")                            # the better of the two (lower CV RMSE) is THE exact reference for every comparison
# hopfield / ann / ann_sq8 use the similarity of the reference (cosine or Euclidean), so that they approximate or re-read THE SAME exact store
STOCHASTIC = ("ann", "ann_sq8", "flyhash", "vsa")                      # evaluated with every seed in SEEDS; exact and hopfield are deterministic
ANN_RULE_TOL = 1.02                # ann/ann_sq8: smallest nprobe whose CV RMSE <= ANN_RULE_TOL * exact's best CV RMSE
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

# ---- corruption ---------------------------------------------------------------------------------------------------
FAMILIES = ("noise", "dropout", "stuck", "drift", "spikes", "quantize", "combined")
LEVELS = {"L25": 1.25, "L50": 1.5, "L100": 2.0}                        # exact's RMSE multiple that defines each severity level
MASK_FAMILIES = ("dropout", "combined")                                # families where an observed-sensor mask exists

# ---- completion ---------------------------------------------------------------------------------------------------
COMPLETION_SEVERITIES = (0.1, 0.3, 0.5)

# ---- OOD / chimeras -----------------------------------------------------------------------------------------------
OOD_STRIDE = 5                     # ID windows of held-out engines and foreign windows are taken every 5th cycle
OOD_MAX_FOREIGN = 800              # foreign windows scored per fold and per set (random subsample, seeded)
OOD_LATE_RUL = 50                  # "late-life" = raw RUL <= 50
CHIMERA_BANDS = {"same": (0.0, 5.0), "mid": (30.0, 60.0), "far": (80.0, 125.0)}     # |capped-RUL gap| in cycles
CHIMERA_N = 500                    # chimeras per band and fold
CHIMERA_KINDS = ("channel", "time")
FOREIGN = ("FD002", "FD003", "FD004")

# ---- bootstrap sizes ------------------------------------------------------------------------------------------------
N_BOOT = {"table": 500, "paired": 1000, "auroc": 300}

# ---- timing -------------------------------------------------------------------------------------------------------
LATENCY_SAMPLES = 1000
THROUGHPUT_BATCH = 256
THROUGHPUT_SECONDS = 0.5           # each throughput repeat runs for at least this long
THROUGHPUT_REPEATS = 3
SCALE_SIZES = {"exact": (20_000, 100_000, 500_000), "ann": (20_000, 100_000, 500_000), "ann_sq8": (20_000, 100_000, 500_000),
               "flyhash": (20_000, 100_000), "vsa": (20_000, 100_000)}
SCALE_D = 56                        # matches 4 statistics x 14 channels
SCALE_NPROBES = (1, 2, 4, 8, 16, 32, 64)
SCALE_RECALL_TARGET = 0.90
SCALE_LATENCY_SAMPLES = 500

# ---- harness validation ---------------------------------------------------------------------------------------------
VALIDATE_CONFIGS = [(256, 512, 0.35), (256, 512, 0.40), (512, 512, 0.40), (1024, 1024, 0.44), (128, 256, 0.30)]   # (D, N, p_flip)
VALIDATE_QUERIES = 4000            # per configuration and seed
VALIDATE_Z = 3.0                   # Wilson interval half-width in sigma (99.7 %)
RECALL_PANEL = {"D": 512, "N": 2048, "p": (0.10, 0.20, 0.30, 0.35, 0.40), "queries": 2000}


def fast():
    """Reduced settings for smoke tests on FAKE data. Never used for reported results (the runner refuses --fast with the real output dir)."""
    import copy
    g = globals()
    over = {"SEEDS": (0,), "CORRUPTION_SEEDS": (0,), "CV_FOLDS": 3, "CV_PER_GROUP": 3, "CV_MIN_T": 8, "WINDOW": 8, "CAP": 30, "MIN_UNIQUE": 10,
            "KS": (3, 10), "BETAS": (None, 30.0), "HOPFIELD_BETAS": (100.0, 1000.0),
            "COMPLETION_SEVERITIES": (0.3,), "OOD_MAX_FOREIGN": 100, "CHIMERA_N": 60, "CHIMERA_BANDS": {"same": (0.0, 1.0), "mid": (6.0, 12.0), "far": (18.0, 30.0)},
            "N_BOOT": {"table": 30, "paired": 30, "auroc": 20}, "LATENCY_SAMPLES": 60, "THROUGHPUT_SECONDS": 0.02, "THROUGHPUT_REPEATS": 2, "SCALE_SIZES": {"exact": (2000,), "ann": (2000,), "ann_sq8": (2000,), "flyhash": (2000,), "vsa": (2000,)},
            "SCALE_LATENCY_SAMPLES": 30, "VALIDATE_QUERIES": 300, "RECALL_PANEL": {"D": 128, "N": 128, "p": (0.2,), "queries": 100}, "LEVELS": {"L50": 1.5}, "FAMILIES": ("noise", "dropout", "combined")}
    over["READOUT_GRID"] = [(k, b) for k in over["KS"] for b in over["BETAS"]]
    over["STRUCT_GRID"] = {"exact": [{}], "exact_l2": [{}], "hopfield": [{}], "ann": [{"nprobe": 2, "sq8": False}, {"nprobe": 8, "sq8": False}],
                           "ann_sq8": [{"nprobe": 8, "sq8": True}], "flyhash": [{"m": 512, "frac": 0.05}], "vsa": [{"D": 1024}]}
    for k, v in over.items():
        g[k] = copy.deepcopy(v)
