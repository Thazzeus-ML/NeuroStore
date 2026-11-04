# %% id=config
#@title Benchmark configuration (run first)
PROFILE = "quick" #@param ["smoke", "quick", "extended"]
SEED = 2026 #@param {type:"integer"}
D = 512 #@param {type:"integer"}
BETA = 16.0 #@param {type:"number"}
SIMULATE_CPU_DEVICES = 0 #@param {type:"integer"}
OUTPUT_ROOT = "" #@param {type:"string"}

RUN_HOPFIELD = True #@param {type:"boolean"}
RUN_KERNELS = True #@param {type:"boolean"}
RUN_SCALING = True #@param {type:"boolean"}
RUN_FLY = True #@param {type:"boolean"}
RUN_VSA = True #@param {type:"boolean"}
RUN_SDM = True #@param {type:"boolean"}
RUN_OOD = True #@param {type:"boolean"}
RUN_FAULTS = True #@param {type:"boolean"}
RUN_MULTIDEVICE = True #@param {type:"boolean"}
RUN_SPARSE = True #@param {type:"boolean"}

# `smoke` only checks that the notebook runs; `quick` is meant for a Colab session of a few minutes;
# `extended` uses more seeds, larger N, and long latency samples. nmax_* cap the largest stored-pattern count N
# (CPU runtimes are far smaller). budget_s stops a single sweep early (the cut-off is reported, not silent).
PROFILES = {
    "smoke": dict(seeds=2, Q=64, n_values=[64, 256], noise_levels=[0.0, 0.25, 0.40], repeats=4, warmup=1, lat_samples=40,
                  nmax_cpu=2 ** 11, nmax_acc=2 ** 11, q_sweep=[1, 8, 64], hop_betas=[1, 16, 256], hop_steps=[1, 2],
                  hop_cap_n=[64, 256, 1024], fly_noise=[0.1, 0.3, 0.4], vsa_dims=[1024], vsa_records=64, vsa_ks=[1, 5, 15],
                  vsa_trials=100, sdm_dim=128, sdm_M=[256], sdm_K=[8], sdm_loads=[10, 20, 40], sdm_noise=[0.0, 0.1],
                  ood_n=128, ood_q=128, fault_n=128, fault_q=128, sparse_N=[256, 1024], sparse_B=16, budget_s=10),
    "quick": dict(seeds=3, Q=256, n_values=[128, 512, 2048], noise_levels=[0.0, 0.10, 0.25, 0.30, 0.35, 0.40, 0.45],
                  repeats=20, warmup=3, lat_samples=300, nmax_cpu=2 ** 15, nmax_acc=2 ** 18,
                  q_sweep=[1, 4, 16, 64, 256, 1024, 4096], hop_betas=[1, 2, 4, 8, 16, 32, 64, 128, 256], hop_steps=[1, 2, 3],
                  hop_cap_n=[128, 512, 2048, 8192], fly_noise=[0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40],
                  vsa_dims=[1024, 2048, 4096], vsa_records=160, vsa_ks=[1, 3, 5, 9, 15, 25, 41], vsa_trials=400,
                  sdm_dim=256, sdm_M=[1024, 4096], sdm_K=[8, 16, 32], sdm_loads=[25, 50, 100, 200, 400, 800],
                  sdm_noise=[0.0, 0.10, 0.20], ood_n=512, ood_q=512, fault_n=512, fault_q=512,
                  sparse_N=[1024, 4096, 16384, 65536], sparse_B=256, budget_s=90),
    "extended": dict(seeds=5, Q=512, n_values=[128, 512, 2048, 8192], noise_levels=[0.0, 0.10, 0.25, 0.30, 0.35, 0.40, 0.45],
                     repeats=50, warmup=5, lat_samples=2000, nmax_cpu=2 ** 17, nmax_acc=2 ** 20,
                     q_sweep=[1, 4, 16, 64, 256, 1024, 4096, 16384], hop_betas=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                     hop_steps=[1, 2, 3, 5], hop_cap_n=[128, 512, 2048, 8192, 32768],
                     fly_noise=[0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
                     vsa_dims=[1024, 2048, 4096, 8192], vsa_records=640, vsa_ks=[1, 3, 5, 9, 15, 25, 41, 61], vsa_trials=1000,
                     sdm_dim=256, sdm_M=[1024, 4096, 16384], sdm_K=[8, 16, 32, 64], sdm_loads=[25, 50, 100, 200, 400, 800, 1600],
                     sdm_noise=[0.0, 0.10, 0.20], ood_n=1024, ood_q=1024, fault_n=1024, fault_q=1024,
                     sparse_N=[1024, 4096, 16384, 65536, 262144], sparse_B=512, budget_s=300),
}
assert PROFILE in PROFILES, f"unknown PROFILE {PROFILE!r}"
assert D % 32 == 0, "D must be a multiple of 32 (bit-packed kernels)"
P = PROFILES[PROFILE]
print(f"profile={PROFILE} seed={SEED} D={D} beta={BETA} seeds/profile={P['seeds']} Q={P['Q']} N={P['n_values']}")
