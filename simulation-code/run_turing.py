#%%
"""
Turing pattern pipeline: choose model -> design parameters -> simulate (vectorised) -> show & analyse.

Run top to bottom (`python run_turing.py`) or cell by cell in VS Code.
Everything you'd normally tweak lives in the CONFIG cell.

Works for any two-species reaction-diffusion model
    du/dt = f(u, v) + Du x* L u
    dv/dt = g(u, v) + Dv * L v
where L = A - D is the graph Laplacian of the node network (eigenvalues Λ <= 0).
The steady state and Jacobian are found numerically, so nothing below assumes linear kinetics.

Rather than picking Du, Dv by hand you can pick *what the pattern should do*:
    target_mode   Λ_m : Laplacian eigenvalue that grows fastest (sets the wavelength, ~2π/√-Λ_m)
    target_growth λ_m : how fast that mode grows
and Du, Dv are solved for with parameters.py. Set Du/Dv in the config to skip this.

ADDING A NEW MODEL
    1. Write a factory in models.py that takes parameters and returns (f, g), e.g.
           def brusselator(a, b):
               def f(u, v): return a - (b + 1)*u + u**2*v
               def g(u, v): return b*u - u**2*v
               return f, g
    2. Add an entry to MODELS below with default parameters and a guess for the steady state.
    3. Set cfg.model = 'brusselator'.
"""

import importlib
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage

import simulation
import models
import parameters
import analysis
import shapes

for module in (simulation, models, parameters, analysis, shapes):
    importlib.reload(module)

plt.rcParams.update({'font.size': 11})  # plots.py (imported by analysis) sets this to 20

#%% MODELS
@dataclass
class Config:
    model: str = 'gray_scott'
    model_params: dict = field(default_factory=dict)  # overrides MODELS[model].params
    # pattern design (used when Du, Dv are None)
    target_mode: float = -0.1    # Λ_m: more negative -> shorter wavelength
    target_growth: float = 0.5   # λ_m: must be below the model's maximum (an error says if not)
    Du: float = None
    Dv: float = None
    # network / domain
    grid_size: int = 10
    threshold: float = 1.1       # connection radius: 1.1 -> 4 neighbours, 1.5/1.6 -> 8, 2.1 -> 12
    shape: callable = shapes.grid_check
    # integration
    T: float = 4000              # None -> 20 e-folds of the predicted growth rate
    dt: float = 1
    initial_noise: float = 0.01   # std of random perturbation around the steady state
    start_offset: float = 0.0     # uniform shift of the initial u (biases a bistable start towards one state)
    start_seed: tuple = (1, None, shapes.heart_outline_check)  # needs the ': tuple' to be a Config field
                                  # (radius, amplitude): instead of noise, a uniform state with a disc of raised u
                                  # in the centre. amplitude None -> 0.5·max(|u0|, 1), or for bistable models
                                  # the low stable state with a disc of the high one. See cell 14.
                                  # Optional 3rd entry: a shape from shapes.py instead of a disc, e.g.
                                  # (12, None, shapes.heart_outline_check) - radius is then half its width.
    limits: tuple = None         # (u_min, u_max, v_min, v_max) clipping, None -> ±65504
    seed: int = 0
    n_repeats: int = 10          # seeds for the robustness check (cell 11)
    sweep_modes: tuple = (-0.5, -1, -1.5, -2, -3, -4, -5)  # target Λ_m values for the sweep (cell 12)
    sweep_repeats: int = 5       # seeds per target in the sweep
    n_snapshots: int = 6
    n_frames: int = 150          # frames in the animation (cell 15), spread evenly over the run
    animation_window: tuple = None  # (t_start, t_end) to animate only part of the run, None -> whole run.
                                    # Aim for ~20 frames per oscillation: (t_end - t_start) / n_frames ≈ period / 20
    animation_fps: int = 20      # playback speed: lower -> slower, easier to follow
    animation_path: str = "animation.mp4"   # e.g. 'pattern.mp4' or 'pattern.gif' to save it; None -> only show inline

# oscillating Gray-Scott on a 10×10 grid: diffusion scaled ×0.15 so the breathing pattern fits, and a small
# seed (radius 3) so the domain doesn't tip into the uniform red state. Narrow window - see notes below.
cfg = Config(model='gray_scott', model_params={'a': 0.02, 'b': 0.05},
             Du=0.035*0.4, Dv=0.07*0.4, grid_size=10, start_seed=(3, None), T=10000, dt=1.0,
             animation_window=(4000, 6000), n_frames=200, animation_fps=15) 

#%% 1. MODEL: steady state & Jacobian

def numerical_jacobian(f, g, u0, v0, h=1e-6):
    """Central-difference Jacobian [[fu, fv], [gu, gv]] at (u0, v0)."""
    hu, hv = h * max(1, abs(u0)), h * max(1, abs(v0))
    return np.array([
        [(f(u0 + hu, v0) - f(u0 - hu, v0)) / (2 * hu), (f(u0, v0 + hv) - f(u0, v0 - hv)) / (2 * hv)],
        [(g(u0 + hu, v0) - g(u0 - hu, v0)) / (2 * hu), (g(u0, v0 + hv) - g(u0, v0 - hv)) / (2 * hv)],
    ])


def classify(J):
    tr, det = np.trace(J), np.linalg.det(J)
    if det < 0:
        return 'saddle (unstable)'
    kind = 'spiral' if tr**2 - 4*det < 0 else 'node'
    return f"stable {kind}" if tr < 0 else f"unstable {kind}"


def find_fixed_points(f, g, u_range, v_range, n_guesses=8):
    """Run the root finder from a grid of guesses and keep the distinct solutions."""
    found = []
    for u in np.linspace(*u_range, n_guesses):
        for v in np.linspace(*v_range, n_guesses):
            try:
                x = parameters.find_steady_state(f, g, (u, v))
            except RuntimeError:
                continue
            if np.isfinite(x).all() and not any(np.allclose(x, y, atol=1e-6) for y in found):
                found.append(x)
    return found


def build_model(cfg):
    """Pick the fixed point the run starts from (see ModelSpec.start) and linearise there.

    'stable': the fixed point found from spec.guess, or - if that one is unstable - the stable fixed
              point nearest to it, so models with several fixed points still work.
    'saddle': the saddle between two stable states. Its uniform mode is unstable, so the network
              splits into domains of the two states instead of forming a Turing pattern.
    """
    spec = MODELS[cfg.model]
    kinetic_params = {**spec.params, **cfg.model_params}
    f, g = spec.factory(**kinetic_params)

    u0, v0 = parameters.find_steady_state(f, g, initial_guess=spec.guess)
    J = numerical_jacobian(f, g, u0, v0)
    stable_states = []

    if spec.start == 'saddle' or not classify(J).startswith('stable'):
        points = [(x, numerical_jacobian(f, g, *x)) for x in find_fixed_points(f, g, *spec.phase_range)]
        stable_states = [x for x, Jx in points if classify(Jx).startswith('stable')]
        candidates = [(x, Jx) for x, Jx in points
                      if classify(Jx).startswith('saddle' if spec.start == 'saddle' else 'stable')]
        if spec.start == 'saddle' and len(stable_states) < 2:
            raise ValueError(f"{cfg.model} is not bistable for {kinetic_params}: "
                             f"{len(stable_states)} stable fixed point(s) in {spec.phase_range}.")
        if not candidates:
            raise ValueError(f"{cfg.model} has no stable fixed point in {spec.phase_range} "
                             f"(the one at ({u0:.3g}, {v0:.3g}) is a {classify(J)}), so it cannot form a Turing pattern.")
        (u0, v0), J = min(candidates, key=lambda c: np.linalg.norm(c[0] - np.array(spec.guess)))

    return {'name': cfg.model, 'params': kinetic_params, 'f': f, 'g': g,
            'steady_state': np.array([u0, v0]), 'J': J, 'kind': classify(J),
            'start': spec.start, 'stable_states': stable_states}


model = build_model(cfg)

#%% 2. DIFFUSION PARAMETERS

def design_diffusion(cfg, model):
    """Solve for sigma = Dv/Du and Du so that mode Λ_m grows at rate λ_m (unless Du, Dv are given)."""
    J = model['J']
    designed = cfg.Du is None or cfg.Dv is None
    spec = MODELS[cfg.model]

    if model['start'] == 'saddle':  # not a Turing instability: no target to design for
        if designed:
            cfg.Du, cfg.Dv = spec.diffusion
        return {'Du': cfg.Du, 'Dv': cfg.Dv, 'sigma': cfg.Dv / cfg.Du, 'critical_sigma': np.nan,
                'designed': False}

    critical_sigma = float(np.real(parameters.critical_ratio(J)))

    if designed and spec.diffusion is not None:
        cfg.Du, cfg.Dv = spec.diffusion
        designed = False
    elif designed:
        try:
            sigma = float(np.real(parameters.find_sigma_for_growth(J, cfg.target_growth)))
        except ValueError:
            max_growth = np.real(parameters.peak_lambda(J, 300))
            raise ValueError(f"target_growth = {cfg.target_growth} is not reachable for {cfg.model}; "
                             f"choose something below ~{max_growth:.3g}.") from None
        cfg.Du = float(np.real(parameters.epsilon_for_eigenvalue(J, cfg.target_mode, sigma)))
        cfg.Dv = sigma * cfg.Du

    sigma = cfg.Dv / cfg.Du
    if J[0, 0] <= 0:
        print(f"Note: fu = {J[0, 0]:.3g} <= 0, so {cfg.model} has no Turing instability at this state - "
              "any pattern comes from the nonlinear dynamics (e.g. a seeded start).")
    elif not parameters.turing_check(J, sigma)[1]:
        print(f"WARNING: σ = Dv/Du = {sigma:.2f} is below the critical ratio {critical_sigma:.2f}; "
              "no mode is unstable, so expect no pattern.")

    return {'Du': cfg.Du, 'Dv': cfg.Dv, 'sigma': sigma, 'critical_sigma': critical_sigma, 'designed': designed}


diffusion = design_diffusion(cfg, model)

#%% 3. NETWORK & LINEAR STABILITY

def seed_background(model):
    """(background, patch) states for a seeded start: bistable -> low and high stable states,
    otherwise the steady state and MODELS[...].seed_patch (None -> raise u by the seed amplitude)."""
    if model['start'] == 'saddle':
        low, high = sorted(model['stable_states'], key=lambda x: x[0])
        return np.array(low), np.array(high)
    patch = MODELS[model['name']].seed_patch
    return model['steady_state'], None if patch is None else np.array(patch, dtype=float)


def initial_state(cfg, model):
    """Steady state plus noise - same noise as models.initial_fill(pert=True) in the other scripts, but seeded.
    With cfg.start_seed: no noise, just a disc of raised u in the middle of the domain."""
    if cfg.start_seed is not None:
        radius, amplitude, *seed_shape = cfg.start_seed
        background, patch = seed_background(model)
        if amplitude is not None or patch is None:
            amplitude = 0.5 * max(abs(background[0]), 1) if amplitude is None else amplitude
            patch = background + np.array([amplitude, 0.0])
        index = np.arange(cfg.grid_size ** 2)  # same ordering as simulation.shape_nodes: x fastest
        centre = (cfg.grid_size + 1) / 2
        x, y = index % cfg.grid_size + 1 - centre, index // cfg.grid_size + 1 - centre
        if seed_shape:  # a shapes.py check(r, coords), which draws its shape in [1, 2r]: shift it to the centre
            inside = np.asarray(seed_shape[0](radius, np.column_stack([x + radius, y + radius])), dtype=bool)
        else:
            inside = np.hypot(x, y) < radius
        state = np.tile(background + np.array([cfg.start_offset, 0.0]), (cfg.grid_size ** 2, 1))
        state[inside] = patch
        return state

    np.random.seed(cfg.seed)
    start = model['steady_state'] + np.array([cfg.start_offset, 0.0])
    return start + cfg.initial_noise * models.initial_fill(cfg.grid_size, fill=0, pert=True)


def build_network(cfg, model):
    """Node positions, initial state, Laplacian and its eigen-decomposition."""
    initial = initial_state(cfg, model)
    # shape_nodes reads one initial row per node *inside* the shape, so drop the rows outside it
    index = np.arange(cfg.grid_size ** 2)
    inside = cfg.shape(cfg.grid_size / 2, np.column_stack([index % cfg.grid_size + 1, index // cfg.grid_size + 1]))
    initial = initial[np.broadcast_to(np.asarray(inside, dtype=bool), index.shape)]

    A, D, nodes = simulation.generate_nodes(threshold=cfg.threshold, shape=cfg.shape,
                                            grid_size=cfg.grid_size, initial_conditions=initial)
    eigvals, eigvecs = np.linalg.eigh(A - D)  # ascending: most negative (shortest wavelength) first

    return {'initial': initial,
            'xy': np.array([(node.x, node.y) for node in nodes], dtype=int),
            'eigvals': eigvals, 'eigvecs': eigvecs,
            'mean_degree': np.diag(D).mean()}


def growth_rates(model, diffusion, Lambda):
    """Leading growth rate Re(λ) of each Laplacian mode Λ, from analysis.growth_rate."""
    (fu, fv), (gu, gv) = model['J']
    params = (fu, fv, 0, gu, gv, 0, diffusion['Du'], diffusion['Dv'])  # (a, b, c, d, e, w, Du, Dv) layout
    return analysis.growth_rate(params, np.atleast_1d(Lambda)).real


def mode_eigenvalues(J, Du, Dv, Lambda):
    """Both eigenvalues (leading first) of J + Λ diag(Du, Dv). J may be stacked (..., 2, 2) and broadcasts with Λ."""
    J = np.asarray(J)
    tr = J[..., 0, 0] + J[..., 1, 1] + (Du + Dv) * Lambda
    det = (J[..., 0, 0] + Du * Lambda) * (J[..., 1, 1] + Dv * Lambda) - J[..., 0, 1] * J[..., 1, 0]
    root = np.sqrt(tr ** 2 / 4 - det + 0j)
    return tr / 2 + root, tr / 2 - root


def predict_behaviour(eigvals, leading, tol=1e-9):
    """What linear theory says the network will do, from the leading eigenvalue of each mode."""
    unstable = leading.real > tol
    if not unstable.any():
        return 'no pattern (returns to uniform)'
    uniform = unstable & (np.abs(eigvals) < tol)
    if uniform.any():
        if (np.abs(leading.imag[uniform]) > tol).any():
            return 'Hopf: whole domain oscillates'
        return 'saddle: nodes fall to either stable state -> domains and fronts (not Turing)'
    if (np.abs(leading.imag[unstable]) > tol).any():
        return 'wave instability (oscillating pattern)'
    return 'stationary Turing pattern'


def check_timestep(cfg, model, diffusion, net):
    """Forward Euler needs |1 + dt·μ| < 1 for every decaying mode μ of J + Λ diag(Du, Dv)."""
    mu = np.concatenate(mode_eigenvalues(model['J'], diffusion['Du'], diffusion['Dv'], net['eigvals']))
    decaying = mu[mu.real < 0]
    if np.abs(1 + cfg.dt * decaying).max() >= 1:
        dt_max = np.min(-2 * decaying.real / np.abs(decaying) ** 2)
        print(f"WARNING: dt = {cfg.dt} is unstable for forward Euler; use dt < {dt_max:.2e}.")


net = build_network(cfg, model)
if cfg.start_seed is not None:
    seed_fraction = np.any(net['initial'] != seed_background(model)[0] + np.array([cfg.start_offset, 0.0]), axis=1).mean()
    if seed_fraction > 0.5:
        print(f"WARNING: the seeded patch covers {seed_fraction:.0%} of the domain - it has no room to spread or "
              "divide (Gray-Scott starves and dies). Shrink start_seed's radius or increase grid_size.")
growth = growth_rates(model, diffusion, net['eigvals'])
leading = mode_eigenvalues(model['J'], diffusion['Du'], diffusion['Dv'], net['eigvals'])[0]
prediction = predict_behaviour(net['eigvals'], leading)
check_timestep(cfg, model, diffusion, net)

if cfg.T is None:
    cfg.T = 20 / max(growth.max(), 1e-3)
n_steps = int(round(cfg.T / cfg.dt))

#%% 4. SIMULATE (vectorised)

def run(cfg, model, diffusion, net, n_steps):
    out = simulation.simulation_vectorised(
        model['f'], model['g'], diffusion['Du'], diffusion['Dv'],
        grid_size=cfg.grid_size, n=n_steps, dt=cfg.dt,
        initial_conditions=net['initial'].copy(), shape=cfg.shape,
        threshold=cfg.threshold, limits=cfg.limits, show=False)

    if len(out) == 2:  # simulation_vectorised returns only (u, v) on blow-up
        raise RuntimeError("Simulation blew up - reduce dt (see any timestep warning above).")

    u, v, (u_log, v_log) = out
    return {'u': u.copy(), 'v': v.copy(), 'u_log': u_log, 'v_log': v_log,
            't': np.arange(1, n_steps + 1) * cfg.dt}


sim = run(cfg, model, diffusion, net, n_steps)

#%% 5. ANALYSE

def to_grid(values, xy, grid_size):
    """Place node values on an image (NaN where there is no node, e.g. outside a circle)."""
    grid = np.full((grid_size, grid_size), np.nan)
    grid[xy[:, 1] - 1, xy[:, 0] - 1] = values
    return grid


def graph_fourier_power(u, eigvecs):
    """Fraction of the pattern's power in each Laplacian eigenmode (uniform mode removed)."""
    power = (eigvecs.T @ (u - u.mean())) ** 2
    return power / power.sum()


def linear_phase(amplitude):
    """Indices where the amplitude grows exponentially: after the initial decay, before 10% of its max."""
    end = np.argmax(amplitude > 0.1 * amplitude.max())
    start = np.argmin(amplitude[:end]) if end > 0 else 0
    return slice(start, end) if end - start > 2 else None


def rate_of_change(sim, n_points=100):
    """RMS |du/dt| across nodes, from differences over ~1% of the run (smooths float32 log noise)."""
    stride = max(len(sim['t']) // n_points, 1)
    idx = np.arange(0, len(sim['t']), stride)
    rate = np.linalg.norm(np.diff(sim['u_log'][:, idx], axis=1), axis=0) / np.sqrt(len(sim['u']))
    return sim['t'][idx[1:]], rate / (sim['t'][stride] - sim['t'][0])


def pattern_shape(u_grid, u0, spot_roundness=0.6, stripe_roundness=0.4):
    """Spots vs stripes, from the blobs of the minority phase after thresholding at u0.

    roundness = 16·area / perimeter² per blob (area-weighted mean over blobs):
    1 for a square, ~0.8 for a pixelated disc, ~4·width/length for a stripe, lower for labyrinths.
    Calibrated on a 30×30 grid: labyrinths (BVAM C = 0, linear model) ~0.3, mixed (BVAM C = 0.5-1) ~0.45-0.55,
    spots (BVAM C >= 1.5) ~0.65, Schnakenberg spots ~0.8. Small, pixelated spots score lower.
    """
    valid = np.isfinite(u_grid)
    above = (u_grid > u0) & valid
    fraction_above = above.sum() / valid.sum()
    if not 0.01 < fraction_above < 0.99:  # pattern sits on one side of u0 (e.g. Gray-Scott, u >= 0): use mid-range
        above = (u_grid > (np.nanmin(u_grid) + np.nanmax(u_grid)) / 2) & valid
        fraction_above = above.sum() / valid.sum()
    minority_above = fraction_above <= 0.5
    phase = above if minority_above else (~above & valid)

    labels, n_blobs = ndimage.label(phase)
    roundness, areas = [], []
    for k in range(1, n_blobs + 1):
        blob = np.pad(labels == k, 1).astype(int)
        area = blob.sum()
        if area < 3:
            continue
        perimeter = np.abs(np.diff(blob, axis=0)).sum() + np.abs(np.diff(blob, axis=1)).sum()
        roundness.append(16 * area / perimeter ** 2)
        areas.append(area)
    mean_roundness = np.average(roundness, weights=areas) if roundness else np.nan

    if not np.isfinite(mean_roundness):
        kind = 'none'
    elif mean_roundness >= spot_roundness:
        kind = 'spots' if minority_above else 'holes (inverse spots)'
    elif mean_roundness <= stripe_roundness:
        kind = 'stripes / labyrinth'
    else:
        kind = 'mixed spots & stripes'

    return {'type': kind, 'roundness': mean_roundness, 'n_blobs': len(areas),
            'minority_fraction': min(fraction_above, 1 - fraction_above)}


def observe_behaviour(model, amplitude, relative_rate, growth_scale):
    """Label the final state: uniform, settled pattern, or still changing."""
    if amplitude[-1] < 1e-3 * max(abs(model['steady_state'][0]), 1):
        return 'uniform (no pattern)'
    if relative_rate < 0.01 * growth_scale:
        return 'stationary pattern'
    return 'still changing (oscillating, drifting or not yet settled)'


def analyse(cfg, model, net, sim, growth):
    eigvals = net['eigvals']
    power = graph_fourier_power(sim['u'], net['eigvecs'])

    amplitude = sim['u_log'].std(axis=0)
    window = linear_phase(amplitude)
    measured_growth = (np.polyfit(sim['t'][window], np.log(amplitude[window]), 1)[0]
                       if window is not None else np.nan)

    # late-time |du/dt| relative to the pattern amplitude: ~0 once the pattern has settled
    rate_t, rate = rate_of_change(sim)
    relative_rate = rate[-1] / max(amplitude[-1], 1e-12)

    res = {
        'rate_t': rate_t,
        'rate': rate,
        'relative_rate': relative_rate,
        'observed': observe_behaviour(model, amplitude, relative_rate, max(abs(growth.max()), 1e-3)),
        'power': power,
        'amplitude': amplitude,
        'window': window,
        'predicted_mode': eigvals[np.argmax(growth)],
        'predicted_growth': growth.max(),
        'measured_mode': eigvals[np.argmax(power)],
        'measured_growth': measured_growth,
        'unstable_modes': int((growth > 0).sum()),
        'spectral_centroid': np.sum(eigvals * power),
        'contrast': np.ptp(sim['u']) / max(abs(model['steady_state'][0]), 1),
        'u_grid': to_grid(sim['u'], net['xy'], cfg.grid_size),
    }
    res['shape'] = (pattern_shape(res['u_grid'], model['steady_state'][0])
                    if res['observed'] != 'uniform (no pattern)' else
                    {'type': 'none', 'roundness': np.nan, 'n_blobs': 0, 'minority_fraction': 0})
    if np.isfinite(res['u_grid']).all():  # DCT wavelength only makes sense on a full square
        res['dct_wavelength'] = analysis.strongest_wavelength(res['u_grid'])
    return res


def wavelength(Lambda):
    return 2 * np.pi / np.sqrt(-Lambda) if Lambda < -1e-9 else np.inf  # uniform mode is ~-1e-16, not exactly 0


res = analyse(cfg, model, net, sim, growth)

#%% 6. SUMMARY

def summary(cfg, model, diffusion, net, res, prediction, leading):
    kinetic = ', '.join(f'{k}={v:g}' for k, v in model['params'].items())
    u0, v0 = model['steady_state']
    designed = diffusion['designed'] and cfg.target_mode is not None

    def row(name, target, predicted, measured, fmt='.3f'):
        t = f'{target:{fmt}}' if target is not None else '-'
        return f"  {name:<20}{t:>10}{predicted:>14{fmt}}{measured:>12{fmt}}"

    print('\n' + '=' * 60)
    print(f"  Model        {model['name']} ({kinetic})")
    print(f"  Start        u0 = {u0:.4g}, v0 = {v0:.4g}  ({model['kind']})")
    for u_s, v_s in model['stable_states'] if model['start'] == 'saddle' else []:
        print(f"  Stable state u = {u_s:.4g}, v = {v_s:.4g}")
    print(f"  Jacobian     {np.array2string(model['J'], precision=3, separator=', ')}")
    critical = (f", critical {diffusion['critical_sigma']:.2f}" if np.isfinite(diffusion['critical_sigma']) else '')
    print(f"  Diffusion    Du = {diffusion['Du']:.4g}, Dv = {diffusion['Dv']:.4g}  "
          f"(σ = {diffusion['sigma']:.2f}{critical}, {'designed' if diffusion['designed'] else 'fixed'})")
    print(f"  Network      {len(net['xy'])} nodes, {net['mean_degree']:.1f} neighbours on average")
    print(f"  Run          T = {cfg.T:.3g}, dt = {cfg.dt}")
    print('-' * 60)
    print(f"  {'':<20}{'target':>10}{'prediction':>14}{'measured':>12}")
    print(row('Dominant mode Λ', cfg.target_mode if designed else None, res['predicted_mode'], res['measured_mode']))
    print(row('Growth rate λ', cfg.target_growth if designed else None, res['predicted_growth'], res['measured_growth']))
    print(row('Wavelength 2π/√-Λ', wavelength(cfg.target_mode) if designed else None,
              wavelength(res['predicted_mode']), wavelength(res['measured_mode']), '.2f'))
    print('-' * 60)
    unstable = leading.real > 1e-9
    max_imag = np.abs(leading.imag[unstable]).max() if unstable.any() else 0
    print(f"  Unstable modes          {res['unstable_modes']} of {len(net['eigvals'])}"
          f"  (max |Im λ| among them: {max_imag:.3g})")
    print(f"  GFT spectral centroid   {res['spectral_centroid']:.3f}")
    if 'dct_wavelength' in res:
        print(f"  DCT strongest wavelength {res['dct_wavelength']} grid units")
    print('-' * 60)
    print(f"  Predicted (linear)      {prediction}")
    print(f"  Observed (simulation)   {res['observed']}")
    print(f"  Late |du/dt| / std(u)   {res['relative_rate']:.2e} per unit time")
    shape = res['shape']
    print(f"  Pattern type            {shape['type']}  (roundness {shape['roundness']:.2f}, "
          f"{shape['n_blobs']} blobs, minority phase {shape['minority_fraction']:.0%} of nodes)")
    print('=' * 60 + '\n')


summary(cfg, model, diffusion, net, res, prediction, leading)

#%% 7. DASHBOARD

def dashboard(cfg, model, diffusion, net, sim, growth, res, leading):
    eigvals = net['eigvals']
    u0 = model['steady_state'][0]
    show_target = diffusion['designed'] and cfg.target_mode is not None

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = GridSpec(3, cfg.n_snapshots, figure=fig, height_ratios=[1, 1.6, 1.6])
    half = cfg.n_snapshots // 2

    def show_pattern(ax, values, title):
        # diverging colours centred on the steady state: red = above u0, blue = below
        lim = np.nanmax(np.abs(values - u0)) or 1
        im = ax.imshow(values, cmap='bwr', vmin=u0 - lim, vmax=u0 + lim,
                       origin='lower', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        return im

    # row 1: pattern forming over time
    for k, i in enumerate(np.linspace(0, len(sim['t']) - 1, cfg.n_snapshots).astype(int)):
        show_pattern(fig.add_subplot(gs[0, k]), to_grid(sim['u_log'][:, i], net['xy'], cfg.grid_size),
                     f"t = {sim['t'][i]:.3g}")

    # final pattern
    ax = fig.add_subplot(gs[1, :half])
    im = show_pattern(ax, res['u_grid'], f"Final pattern u (centred on the starting {model['kind'].split(' (')[0]})")
    fig.colorbar(im, ax=ax, shrink=0.85)

    # dispersion relation: which modes should grow
    ax = fig.add_subplot(gs[1, half:])
    Lambda = np.linspace(eigvals.min() * 1.05, 0, 400)
    ax.plot(Lambda, growth_rates(model, diffusion, Lambda), color='0.6', lw=1.5, label='continuous')
    ax.scatter(eigvals, growth, c=np.where(growth > 0, 'tab:red', 'tab:blue'), s=14, zorder=3,
               label='network modes')
    oscillatory = (growth > 0) & (np.abs(leading.imag) > 1e-9)
    if oscillatory.any():
        ax.scatter(eigvals[oscillatory], growth[oscillatory], s=60, facecolors='none', edgecolors='k',
                   zorder=4, label='oscillatory (Im λ ≠ 0)')
    ax.axhline(0, color='k', lw=0.8)
    if show_target:
        ax.axvline(cfg.target_mode, color='tab:green', ls='--', label=f'target Λ_m = {cfg.target_mode}')
    ax.axvline(res['measured_mode'], color='tab:purple', ls=':', lw=2,
               label=f"measured Λ = {res['measured_mode']:.2f}")
    if growth.max() > 0:
        top = growth.max()
        ax.set_ylim(max(growth.min(), -3 * top), 1.5 * top)
    else:  # every mode decays: show them all, with some room above zero
        ax.set_ylim(1.1 * growth.min(), -0.2 * growth.min())
    ax.set_xlabel('Laplacian eigenvalue Λ')
    ax.set_ylabel('Growth rate Re(λ)')
    ax.set_title('Dispersion relation (red = unstable)')
    ax.legend(fontsize=9, loc='lower left')

    # graph Fourier spectrum of the final pattern
    ax = fig.add_subplot(gs[2, :half])
    unstable = eigvals[growth > 0]
    if unstable.size:
        ax.axvspan(unstable.min(), unstable.max(), color='tab:red', alpha=0.1, label='unstable band')
    ax.vlines(eigvals, 0, res['power'], color='tab:purple', lw=1.5)
    if show_target:
        ax.axvline(cfg.target_mode, color='tab:green', ls='--', label='target Λ_m')
    ax.set_xlabel('Laplacian eigenvalue Λ')
    ax.set_ylabel('Fraction of power')
    ax.set_title('Graph Fourier spectrum of final u')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)

    # pattern amplitude over time vs linear theory
    ax = fig.add_subplot(gs[2, half:])
    t, amp, w = sim['t'], res['amplitude'], res['window']
    ax.semilogy(t, amp, color='k', lw=1.5, label='std(u) across nodes')
    if w is not None:
        t0, a0 = t[w][0], amp[w][0]
        ax.semilogy(t[w], a0 * np.exp(res['predicted_growth'] * (t[w] - t0)), color='tab:red', ls='--',
                    label=f"theory λ = {res['predicted_growth']:.3g}")
        ax.semilogy(t[w], a0 * np.exp(res['measured_growth'] * (t[w] - t0)), color='tab:purple', ls=':',
                    lw=2, label=f"fit λ = {res['measured_growth']:.3g}")
    ax.set_xlabel('time')
    ax.set_ylabel('pattern amplitude')
    ax.set_title(f"Growth of the pattern  →  {res['observed']}, {res['shape']['type']}", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle(f"{model['name']}  |  Du = {diffusion['Du']:.3g}, Dv = {diffusion['Dv']:.3g}  |  "
                 f"{len(net['xy'])} nodes, threshold {cfg.threshold}", fontsize=13)
    plt.show()
    return fig


fig = dashboard(cfg, model, diffusion, net, sim, growth, res, leading)

#%% 8. NO DIFFUSION: fixed points & stability
# With Du = Dv = 0 the Laplacian term vanishes and every node is an independent copy of
# du/dt = f(u, v), dv/dt = g(u, v), so one vectorised run traces a whole grid of trajectories.
# A Turing pattern needs the fixed point to be stable here (and unstable once diffusion is on).
# Bistable models have a saddle between two stable states; its stable manifold (dashed, the
# separatrix) splits the plane into the two basins, and each node falls into one of them.
# (classify and find_fixed_points are defined in cell 1.)

def separatrix(f, g, saddle, J, u_range, v_range, dt=0.01, n=5000):
    """Stable manifold of a saddle: integrate backwards in time from just either side of it."""
    w, V = np.linalg.eig(J)
    direction = V[:, np.argmin(w.real)].real
    (u_lo, u_hi), (v_lo, v_hi) = u_range, v_range
    curves = []
    for sign in (1, -1):
        x = saddle + 1e-4 * sign * direction
        path = [x]
        for _ in range(n):
            x = x - dt * np.array([f(*x), g(*x)])
            if not (u_lo <= x[0] <= u_hi and v_lo <= x[1] <= v_hi):
                break
            path.append(x)
        curves.append(np.array(path))
    return curves


def phase_plane(model, u_range, v_range, side=6, dt=0.001):
    f, g = model['f'], model['g']

    fixed_points = find_fixed_points(f, g, u_range, v_range)
    print('\nFixed points without diffusion:')
    for u0, v0 in fixed_points:
        J = numerical_jacobian(f, g, u0, v0)
        print(f"  ({u0:.4g}, {v0:.4g}):  {classify(J):<18} eigenvalues {np.round(np.linalg.eigvals(J), 3)}")

    # run long enough for the slowest decay at the main steady state to settle (~5 e-folds)
    slowest = np.abs(np.linalg.eigvals(model['J']).real).min()
    n = int(min(5 / max(slowest, 1e-3), 200) / dt)

    U0, V0 = np.meshgrid(np.linspace(*u_range, side), np.linspace(*v_range, side))
    starts = np.column_stack([U0.ravel(), V0.ravel()])
    _, _, (u_traj, v_traj) = simulation.simulation_vectorised(
        f, g, 0, 0, grid_size=side, n=n, dt=dt, initial_conditions=starts.copy(), show=False)

    uu, vv = np.meshgrid(np.linspace(*u_range, 200), np.linspace(*v_range, 200))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.streamplot(uu, vv, f(uu, vv), g(uu, vv), color='0.8', density=1.2)
    ax.contour(uu, vv, f(uu, vv), levels=[0], colors='tab:red')     # u-nullcline: du/dt = 0
    ax.contour(uu, vv, g(uu, vv), levels=[0], colors='tab:blue')    # v-nullcline: dv/dt = 0
    ax.plot(u_traj.T, v_traj.T, color='k', lw=0.8, alpha=0.6)
    for u0, v0 in fixed_points:
        J = numerical_jacobian(f, g, u0, v0)
        stable = classify(J).startswith('stable')
        ax.plot(u0, v0, 'o', ms=10, mfc='k' if stable else 'w', mec='k', zorder=5)
        if classify(J).startswith('saddle'):
            for k, curve in enumerate(separatrix(f, g, np.array([u0, v0]), J, u_range, v_range)):
                ax.plot(curve[:, 0], curve[:, 1], color='k', ls='--', lw=1.5, zorder=4,
                        label='separatrix (basin boundary)' if k == 0 else None)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlim(u_range)
    ax.set_ylim(v_range)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f"{model['name']} phase plane (no diffusion)\nred = u-nullcline, blue = v-nullcline, "
                 "● stable / ○ unstable", fontsize=10)
    plt.show()
    return fixed_points, fig

fixed_points, phase_fig = phase_plane(model, *MODELS[cfg.model].phase_range)

#%% 9. PARAMETER PLANE: where can this model make Turing patterns?
# For each pair of kinetic parameters: find the fixed point, linearise, and classify it.
#   unstable           tr J > 0 or det J < 0  -> oscillates (Hopf) or runs away; no Turing pattern
#   stable, no Turing  stable but fu <= 0     -> activator doesn't self-activate, no Dv/Du helps
#   Turing possible    stable and fu > 0      -> patterns once Dv/Du exceeds the critical ratio σc
# Dashed line: spiral (damped oscillation) vs node.  Thin grey lines: critical σc = Dv/Du needed.

REGION_COLOURS = ['#8a8a85', '#eb6834', '#2a78d6']  # stable no Turing, unstable, Turing possible


def classify_parameter_plane(cfg, resolution=150):
    spec = MODELS[cfg.model]
    x_name, x_range, y_name, y_range = spec.param_plane
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)

    region = np.full((resolution, resolution), np.nan)
    fields = {name: np.full((resolution, resolution), np.nan)
              for name in ('u0', 'trace', 'det', 'fu', 'sigma_c')}
    fields['J'] = np.full((resolution, resolution, 2, 2), np.nan)

    def solve(f, g, guesses):
        for guess in guesses:
            try:
                x = parameters.find_steady_state(f, g, initial_guess=guess)
            except (RuntimeError, ZeroDivisionError, FloatingPointError):
                continue
            if np.isfinite(x).all():
                return x
        return None

    row_start = spec.guess
    for i, y in enumerate(ys):
        previous = row_start
        for j, x in enumerate(xs):
            f, g = spec.factory(**{**spec.params, **cfg.model_params, x_name: x, y_name: y})
            # the fixed point moves smoothly, so start from the neighbouring solution
            solution = solve(f, g, [previous, spec.guess])
            if solution is None:
                continue
            u0, v0 = previous = solution
            if j == 0:
                row_start = solution
            J = numerical_jacobian(f, g, u0, v0)
            if not np.isfinite(J).all():
                continue

            tr, det, fu = np.trace(J), np.linalg.det(J), J[0, 0]
            stable = tr < 0 and det > 0
            region[i, j] = 0 if (stable and fu <= 0) else 1 if not stable else 2
            fields['u0'][i, j], fields['trace'][i, j], fields['det'][i, j], fields['fu'][i, j] = u0, tr, det, fu
            fields['J'][i, j] = J
            if region[i, j] == 2:
                fields['sigma_c'][i, j] = np.real(parameters.critical_ratio(J))

    return xs, ys, region, fields, (x_name, y_name)


def plot_parameter_plane(cfg, xs, ys, region, fields, names):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    x_name, y_name = names
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.pcolormesh(X, Y, region, cmap=ListedColormap(REGION_COLOURS), vmin=0, vmax=2,
                  alpha=0.35, shading='auto', linewidth=0)

    # region boundaries
    ax.contour(X, Y, fields['trace'], levels=[0], colors=REGION_COLOURS[1], linewidths=2)
    ax.contour(X, Y, np.where(fields['trace'] < 0, fields['fu'], np.nan), levels=[0],
               colors=REGION_COLOURS[2], linewidths=2)
    ax.contour(X, Y, fields['trace'] ** 2 - 4 * fields['det'], levels=[0],
               colors='0.25', linestyles='--', linewidths=1)

    # how much faster the inhibitor must diffuse
    levels = [l for l in (5, 10, 20, 50, 100) if np.nanmin(fields['sigma_c']) < l < np.nanmax(fields['sigma_c'])]
    if levels:
        cs = ax.contour(X, Y, fields['sigma_c'], levels=levels, colors='0.45', linewidths=0.7)
        ax.clabel(cs, fmt=lambda s: f'σc={s:g}', fontsize=8)

    # current parameters
    current = {**MODELS[cfg.model].params, **cfg.model_params}
    ax.plot(current[x_name], current[y_name], marker='*', ms=16, color='k', mec='w', mew=1.5, zorder=5)
    ax.annotate(f"current ({x_name}={current[x_name]:g}, {y_name}={current[y_name]:g})",
                (current[x_name], current[y_name]), xytext=(8, 8), textcoords='offset points', fontsize=9)

    # direct labels at the centre of each region
    for code, text in enumerate(['stable,\nno Turing', 'unstable\n(Hopf or saddle)', 'Turing\npossible']):
        mask = region == code
        if mask.sum() > 0.02 * mask.size:
            ax.text(X[mask].mean(), Y[mask].mean(), text, ha='center', va='center', fontsize=10,
                    color='0.15', weight='bold')

    ax.legend(handles=[Patch(color=c, alpha=0.5, label=l) for c, l in
                       zip(REGION_COLOURS, ['stable, no Turing', 'unstable (Hopf / saddle)', 'Turing possible'])]
                      + [plt.Line2D([], [], color='0.25', ls='--', label='spiral | node'),
                         plt.Line2D([], [], color='0.45', lw=0.7, label='critical Dv/Du')],
              loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{cfg.model}: behaviour of the fixed point across ({x_name}, {y_name})")
    plt.show()
    return fig


if MODELS[cfg.model].param_plane is None:
    print(f"No param_plane set for '{cfg.model}' in MODELS - skipping the parameter-plane map.")
else:
    plane = classify_parameter_plane(cfg)
    plane_fig = plot_parameter_plane(cfg, *plane)

#%% 10. PARAMETER PLANE WITH DIFFUSION: where does *this* network, with *these* Du, Dv, pattern?
# Same (x, y) grid as cell 9, but now every Laplacian mode of the actual network is checked
# with the current Du, Dv. Only the discrete eigenvalues count, so this can differ from cell 9's
# "Turing possible" region: a narrow unstable band can fall between the network's modes.
# Contours: predicted wavelength 2π/√-Λ of the fastest-growing mode.

OUTCOME_COLOURS = ['#8a8a85', '#eb6834', '#2a78d6', '#1baf7a']
OUTCOME_LABELS = ['no pattern', 'unstable without diffusion', 'stationary Turing pattern',
                  'oscillating pattern (Im λ ≠ 0)']


def classify_with_diffusion(diffusion, net, region, fields, tol=1e-9):
    Lambda = net['eigvals']
    lam = mode_eigenvalues(fields['J'][..., None, :, :], diffusion['Du'], diffusion['Dv'], Lambda)[0]
    growth_re = np.where(np.isfinite(lam.real), lam.real, -np.inf)

    fastest = np.argmax(growth_re, axis=-1)
    max_growth = np.take_along_axis(growth_re, fastest[..., None], axis=-1)[..., 0]
    oscillating = ((growth_re > tol) & (np.abs(lam.imag) > tol)).any(axis=-1)

    outcome = np.full(region.shape, np.nan)
    outcome[region == 1] = 1
    stable = np.isin(region, (0, 2))
    outcome[stable & (max_growth <= tol)] = 0
    outcome[stable & (max_growth > tol) & ~oscillating] = 2
    outcome[stable & (max_growth > tol) & oscillating] = 3

    fastest_mode = Lambda[fastest]
    with np.errstate(invalid='ignore', divide='ignore'):
        wavelength_map = np.where(outcome == 2, 2 * np.pi / np.sqrt(-fastest_mode), np.nan)
    return outcome, wavelength_map, np.where(np.isfinite(max_growth), max_growth, np.nan)


def plot_diffusion_plane(cfg, diffusion, n_nodes, xs, ys, outcome, wavelength_map, names):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    x_name, y_name = names
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.pcolormesh(X, Y, outcome, cmap=ListedColormap(OUTCOME_COLOURS), vmin=0, vmax=3,
                  alpha=0.35, shading='auto', linewidth=0)

    if np.isfinite(wavelength_map).any():
        lo, hi = np.nanpercentile(wavelength_map, [5, 95])
        levels = np.unique(np.round(np.linspace(lo, hi, 5), 1))
        if levels.size > 1:
            cs = ax.contour(X, Y, wavelength_map, levels=levels, colors='0.35', linewidths=0.8)
            ax.clabel(cs, fmt=lambda w: f'ℓ={w:g}', fontsize=8)

    current = {**MODELS[cfg.model].params, **cfg.model_params}
    ax.plot(current[x_name], current[y_name], marker='*', ms=16, color='k', mec='w', mew=1.5, zorder=5)
    ax.annotate('current', (current[x_name], current[y_name]), xytext=(8, 8),
                textcoords='offset points', fontsize=9)

    for code, text in enumerate(OUTCOME_LABELS):
        mask = outcome == code
        if mask.sum() > 0.02 * mask.size:
            ax.text(X[mask].mean(), Y[mask].mean(), text.replace(' (', '\n('), ha='center', va='center',
                    fontsize=10, color='0.15', weight='bold')

    present = [code for code in range(4) if (outcome == code).any()]
    ax.legend(handles=[Patch(color=OUTCOME_COLOURS[c], alpha=0.5, label=OUTCOME_LABELS[c]) for c in present]
                      + [plt.Line2D([], [], color='0.35', lw=0.8, label='predicted wavelength ℓ')],
              loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{cfg.model} with diffusion: Du = {diffusion['Du']:.3g}, Dv = {diffusion['Dv']:.3g}, "
                 f"{n_nodes}-node network")
    plt.show()
    return fig


if MODELS[cfg.model].param_plane is not None:
    xs, ys, region, fields, names = plane
    outcome, wavelength_map, max_growth_map = classify_with_diffusion(diffusion, net, region, fields)
    diffusion_plane_fig = plot_diffusion_plane(cfg, diffusion, len(net['xy']), xs, ys, outcome,
                                               wavelength_map, names)

#%% 11. ROBUSTNESS: repeat with different random initial conditions
# Same network, parameters and run length; only the seed of the initial noise changes.
# A robust design gives the same behaviour and dominant mode every time.

from dataclasses import replace


def repeat_seeds(cfg, model, diffusion, net, growth, n_steps):
    runs = []
    for seed in range(cfg.n_repeats):
        cfg_seed = replace(cfg, seed=seed)
        net_seed = {**net, 'initial': initial_state(cfg_seed, model)}  # same network, new noise
        res_seed = analyse(cfg_seed, model, net_seed, run(cfg_seed, model, diffusion, net_seed, n_steps), growth)
        runs.append({'seed': seed, 'u_grid': res_seed['u_grid'], 'observed': res_seed['observed'],
                     'mode': res_seed['measured_mode'], 'growth': res_seed['measured_growth'],
                     'dct_wavelength': res_seed.get('dct_wavelength', np.nan)})
        print(f"  seed {seed}: Λ = {res_seed['measured_mode']:.3f}, λ = {res_seed['measured_growth']:.3f}, "
              f"{res_seed['observed']}")
    return runs


def summarise_repeats(cfg, runs, res):
    modes = np.array([r['mode'] for r in runs])
    rates = np.array([r['growth'] for r in runs])
    lengths = np.array([wavelength(m) for m in modes])
    outcomes, counts = np.unique([r['observed'] for r in runs], return_counts=True)

    print('\n' + '=' * 60)
    print(f"  {cfg.n_repeats} seeds")
    print(f"  Dominant mode Λ      {modes.mean():.3f} ± {modes.std():.3f}   (predicted {res['predicted_mode']:.3f})")
    print(f"  Wavelength 2π/√-Λ    {lengths.mean():.2f} ± {lengths.std():.2f}")
    print(f"  Growth rate λ        {np.nanmean(rates):.3f} ± {np.nanstd(rates):.3f}   "
          f"(predicted {res['predicted_growth']:.3f})")
    for outcome, count in zip(outcomes, counts):
        print(f"  {count:>3} × {outcome}")
    print('=' * 60 + '\n')


def plot_repeats(cfg, model, diffusion, runs, res):
    n = len(runs)
    fig = plt.figure(figsize=(max(2 * n, 8), 6), constrained_layout=True)
    gs = GridSpec(2, n, figure=fig, height_ratios=[1, 1.4])
    u0 = model['steady_state'][0]

    for k, r in enumerate(runs):
        ax = fig.add_subplot(gs[0, k])
        lim = np.nanmax(np.abs(r['u_grid'] - u0)) or 1
        ax.imshow(r['u_grid'], cmap='bwr', vmin=u0 - lim, vmax=u0 + lim, origin='lower', interpolation='nearest')
        ax.set_title(f"seed {r['seed']}\nΛ = {r['mode']:.2f}", fontsize=9)
        ax.set_axis_off()

    ax = fig.add_subplot(gs[1, :])
    modes, counts = np.unique(np.round([r['mode'] for r in runs], 4), return_counts=True)
    ax.vlines(modes, 0, counts, color='tab:purple', lw=4, zorder=3)
    ax.axvline(res['predicted_mode'], color='tab:red', ls='--', label=f"predicted Λ = {res['predicted_mode']:.2f}")
    if diffusion['designed'] and cfg.target_mode is not None:
        ax.axvline(cfg.target_mode, color='tab:green', ls='--', label=f'target Λ_m = {cfg.target_mode}')
    ax.set_xlabel('Dominant Laplacian eigenvalue Λ of the final pattern')
    ax.set_ylabel('Number of seeds')
    ax.set_yticks(range(0, counts.max() + 1))
    ax.set_title('Which mode wins, across random initial conditions')
    ax.legend(fontsize=9)
    plt.show()
    return fig


repeats = repeat_seeds(cfg, model, diffusion, net, growth, n_steps)
summarise_repeats(cfg, repeats, res)
repeats_fig = plot_repeats(cfg, model, diffusion, repeats, res)

#%% 12. SWEEP: target mode vs the mode that actually wins
# For each target Λ_m in cfg.sweep_modes: redesign Du, Dv (same target_growth), note which discrete
# network mode linear theory picks, then run cfg.sweep_repeats seeds and measure the dominant GFT mode.
# Points on the dashed y = x line mean the design hits its target.
# Cost ~ len(sweep_modes) × sweep_repeats simulations - reduce grid_size or sweep_repeats to speed up.

def sweep_target_modes(cfg, model, net):
    rows = []
    for target in cfg.sweep_modes:
        cfg_t = replace(cfg, target_mode=target, Du=None, Dv=None)
        diffusion_t = design_diffusion(cfg_t, model)
        growth_t = growth_rates(model, diffusion_t, net['eigvals'])
        check_timestep(cfg_t, model, diffusion_t, net)
        cfg_t.T = 20 / max(growth_t.max(), 1e-3)
        n_steps_t = int(round(cfg_t.T / cfg_t.dt))

        predicted = net['eigvals'][np.argmax(growth_t)]
        measured, observed, shapes_seen = [], [], []
        for seed in range(cfg.sweep_repeats):
            cfg_s = replace(cfg_t, seed=seed)
            net_s = {**net, 'initial': initial_state(cfg_s, model)}
            res_s = analyse(cfg_s, model, net_s, run(cfg_s, model, diffusion_t, net_s, n_steps_t), growth_t)
            if seed == 0:  # keep one final pattern per target for plot_sweep_examples
                example = {'u_grid': res_s['u_grid'], 'mode': res_s['measured_mode'], 'type': res_s['shape']['type']}
            measured.append(res_s['measured_mode'])
            observed.append(res_s['observed'])
            shapes_seen.append(res_s['shape']['type'])

        measured = np.array(measured)
        types, counts = np.unique(shapes_seen, return_counts=True)
        rows.append({'target': target, 'predicted': predicted, 'measured': measured,
                     'Du': diffusion_t['Du'], 'Dv': diffusion_t['Dv'],
                     'pattern': types[np.argmax(counts)], 'example': example,
                     'hit_rate': np.isclose(measured, predicted).mean(),
                     'settled': np.mean([o == 'stationary pattern' for o in observed])})
        print(f"  target {target:>6.2f}: predicted {predicted:.3f}, measured {measured.mean():.3f} ± {measured.std():.3f}")
    return rows


def summarise_sweep(rows):
    print('\n' + '=' * 108)
    print(f"  {'target Λ_m':>10}{'Du':>9}{'Dv':>9}{'predicted Λ':>13}{'measured Λ':>18}"
          f"{'hits':>8}{'settled':>9}   pattern (most common)")
    for r in rows:
        m = r['measured']
        print(f"  {r['target']:>10.2f}{r['Du']:>9.3g}{r['Dv']:>9.3g}{r['predicted']:>13.3f}"
              f"{f'{m.mean():.3f} ± {m.std():.3f}':>18}{r['hit_rate']:>8.0%}{r['settled']:>9.0%}   {r['pattern']}")
    print("  hits = seeds whose dominant mode is exactly the predicted one; settled = stationary at the end")
    print('=' * 108 + '\n')


def plot_sweep(cfg, rows):
    targets = np.array([r['target'] for r in rows])
    predicted = np.array([r['predicted'] for r in rows])
    measured = [r['measured'] for r in rows]
    to_length = np.vectorize(wavelength)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    panels = [
        (axes[0], targets, predicted, measured, 'Laplacian eigenvalue Λ'),
        (axes[1], to_length(targets), to_length(predicted), [to_length(m) for m in measured],
         'wavelength ℓ = 2π/√-Λ (grid units)'),
    ]
    for ax, x, pred, meas, quantity in panels:
        lo, hi = min(x.min(), np.min(np.concatenate(meas))), max(x.max(), np.max(np.concatenate(meas)))
        ax.plot([lo, hi], [lo, hi], color='0.6', ls='--', lw=1, label='target (y = x)')
        for xi, mi in zip(x, meas):
            ax.scatter(np.full(len(mi), xi), mi, s=14, color='tab:purple', alpha=0.35, linewidths=0)
        ax.errorbar(x, [m.mean() for m in meas], yerr=[m.std() for m in meas], fmt='o', color='tab:purple',
                    ms=7, capsize=4, label=f'measured (mean ± std, {cfg.sweep_repeats} seeds)')
        ax.plot(x, pred, 'D', color='tab:red', ms=6, mfc='none', mew=1.5, label='predicted network mode')
        ax.set_xlabel(f'target {quantity}')
        ax.set_ylabel(f'dominant {quantity}')
        ax.legend(fontsize=9)

    axes[0].set_title('Dominant mode vs target')
    axes[1].set_title('Pattern wavelength vs target')
    fig.suptitle(f"{cfg.model}, target growth λ_m = {cfg.target_growth}, "
                 f"{cfg.grid_size}×{cfg.grid_size} grid, threshold {cfg.threshold}", fontsize=12)
    plt.show()
    return fig


def plot_sweep_examples(cfg, model, rows):
    """Final pattern (seed 0) for each target, with a bar the length of the target wavelength."""
    u0 = model['steady_state'][0]
    fig, axes = plt.subplots(1, len(rows), figsize=(max(2.3 * len(rows), 8), 3.9),
                             constrained_layout=True, squeeze=False)
    for ax, r in zip(axes[0], rows):
        ex = r['example']
        lim = np.nanmax(np.abs(ex['u_grid'] - u0)) or 1
        ax.imshow(ex['u_grid'], cmap='bwr', vmin=u0 - lim, vmax=u0 + lim, origin='lower', interpolation='nearest')
        target_length = wavelength(r['target'])
        if target_length < 0.9 * cfg.grid_size:
            y = -0.06 * cfg.grid_size
            ax.plot([0, target_length], [y, y], color='k', lw=3, clip_on=False)
        ax.set_title(f"target Λ_m = {r['target']:g}  (ℓ = {target_length:.1f})\n"
                     f"measured Λ = {ex['mode']:.2f}  (ℓ = {wavelength(ex['mode']):.1f})\n{ex['type']}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{cfg.model}: one example per target (seed 0)  -  black bar = target wavelength", fontsize=11)
    plt.show()
    return fig


if not diffusion['designed']:
    print(f"Du, Dv are fixed for '{cfg.model}' (not designed from a target mode) - skipping the target-mode sweep.")
else:
    sweep = sweep_target_modes(cfg, model, net)
    summarise_sweep(sweep)
    sweep_fig = plot_sweep(cfg, sweep)
    sweep_examples_fig = plot_sweep_examples(cfg, model, sweep)

#%% 13. PATTERN TYPE: sweep a kinetic parameter and watch spots turn into stripes
# Uses MODELS[cfg.model].pattern_sweep = (param, values), e.g. BVAM's C: C = 0 is symmetric
# (stripes), larger C adds a quadratic asymmetry (spots). For each value Du, Dv are redesigned
# for the same target mode and growth, so only the *shape* of the pattern should change.
# Run with cfg = Config(model='bvam', ...) to use it, or 'fitzhugh_nagumo', which sweeps a Config field
# (start_offset) instead: spots of one state -> labyrinth -> spots of the other, from the same kinetics.

def sweep_pattern_type(cfg, net, n_steps):
    name, values = MODELS[cfg.model].pattern_sweep
    rows = []
    for value in values:
        if hasattr(cfg, name):  # a Config field, e.g. start_offset
            cfg_p = replace(cfg, **{name: value}, Du=None, Dv=None)
        else:                   # a kinetic parameter
            cfg_p = replace(cfg, model_params={**cfg.model_params, name: value}, Du=None, Dv=None)
        try:
            model_p = build_model(cfg_p)
            diffusion_p = design_diffusion(cfg_p, model_p)
        except ValueError as err:
            print(f"  {name} = {value:g}: skipped - {err}")
            continue
        growth_p = growth_rates(model_p, diffusion_p, net['eigvals'])

        seeds = []
        for seed in range(cfg.sweep_repeats):
            cfg_s = replace(cfg_p, seed=seed)
            net_s = {**net, 'initial': initial_state(cfg_s, model_p)}
            res_s = analyse(cfg_s, model_p, net_s, run(cfg_s, model_p, diffusion_p, net_s, n_steps), growth_p)
            seeds.append(res_s)

        types, counts = np.unique([s['shape']['type'] for s in seeds], return_counts=True)
        rows.append({'value': value, 'u0': model_p['steady_state'][0],
                     'roundness': np.array([s['shape']['roundness'] for s in seeds]),
                     'type': types[np.argmax(counts)], 'u_grid': seeds[0]['u_grid'],
                     'settled': np.mean([s['observed'] == 'stationary pattern' for s in seeds])})
        print(f"  {name} = {value:g}: {rows[-1]['type']} (roundness {np.nanmean(rows[-1]['roundness']):.2f}, "
              f"{rows[-1]['settled']:.0%} settled)")
    return name, rows


def plot_pattern_sweep(cfg, name, rows):
    n = len(rows)
    fig = plt.figure(figsize=(max(2.2 * n, 8), 6.5), constrained_layout=True)
    gs = GridSpec(2, n, figure=fig, height_ratios=[1, 1.2])

    for k, r in enumerate(rows):
        ax = fig.add_subplot(gs[0, k])
        lim = max(np.nanmax(np.abs(r['u_grid'] - r['u0'])), 1e-3 * max(abs(r['u0']), 1))  # floor: uniform stays white
        ax.imshow(r['u_grid'], cmap='bwr', vmin=r['u0'] - lim, vmax=r['u0'] + lim,
                  origin='lower', interpolation='nearest')
        ax.set_title(f"{name} = {r['value']:g}\n{r['type']}", fontsize=9)
        ax.set_axis_off()

    ax = fig.add_subplot(gs[1, :])
    values = np.array([r['value'] for r in rows])
    ax.axhspan(0, 0.4, color='#eb6834', alpha=0.12)
    ax.axhspan(0.6, 1.0, color='#2a78d6', alpha=0.12)
    ax.text(0.99, 0.2, 'stripes / labyrinth', transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=9, color='0.3')
    ax.text(0.99, 0.82, 'spots', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=9, color='0.3')
    for r in rows:
        ax.scatter(np.full(len(r['roundness']), r['value']), r['roundness'], s=14,
                   color='tab:purple', alpha=0.35, linewidths=0)
    ax.errorbar(values, [np.nanmean(r['roundness']) for r in rows], yerr=[np.nanstd(r['roundness']) for r in rows],
                fmt='o-', color='tab:purple', ms=7, capsize=4, label=f'mean ± std ({cfg.sweep_repeats} seeds)')
    ax.set_ylim(0, 1)
    ax.set_xlabel(name)
    ax.set_ylabel('blob roundness (16·area / perimeter²)')
    design = (f"target Λ_m = {cfg.target_mode}, λ_m = {cfg.target_growth}" if MODELS[cfg.model].diffusion is None
              else f"Du = {MODELS[cfg.model].diffusion[0]:g}, Dv = {MODELS[cfg.model].diffusion[1]:g}")
    ax.set_title(f"{cfg.model}: pattern type vs {name}  ({design})")
    ax.legend(fontsize=9, loc='upper left')
    plt.show()
    return fig


if MODELS[cfg.model].pattern_sweep is None:
    print(f"No pattern_sweep set for '{cfg.model}' in MODELS - skipping the pattern-type sweep.")
else:
    pattern_param, pattern_rows = sweep_pattern_type(cfg, net, n_steps)
    pattern_fig = plot_pattern_sweep(cfg, pattern_param, pattern_rows)

#%% 14. SEEDED START: inject a patch of 'red' and watch it develop
# No noise: the uniform state plus a disc of raised u in the centre (cfg.start_seed, default below).
#   bistable (fitzhugh_nagumo): the patch settles into a localised spot of a set size, or vanishes
#       when Dv = 0 - the front stops.
#   Turing (schnakenberg, ...): the uniform state is itself unstable, so the patch launches a front
#       that lays the pattern down as it spreads (pattern invasion).
#   gray_scott: the uniform state is stable, yet the patch splits into spots that keep dividing
#       (self-replication) - watch the spot count double.
# Kymograph = u along the horizontal line through the centre against time:
#   a widening cone is an invading front (edge slope = front speed), a band of fixed width is a stable spot.
# Set cfg.start_seed to make the main run (cells 3-7) seeded too; otherwise this cell runs its own.

DEFAULT_SEED = (3, None)  # (radius, amplitude)


def seed_development(cfg, model, net, sim, n_times=400, fraction=0.1):
    """Front radius and disturbed area over time: nodes with |u - background| > fraction · max deviation.
    Front radius = 95th percentile of their distance from the centre, so a few nodes lighting up at the
    edges (fewer neighbours there) don't count as the front."""
    background = seed_background(model)[0][0]
    centre = (cfg.grid_size + 1) / 2
    distance = np.hypot(*(net['xy'] - centre).T)

    idx = np.unique(np.linspace(0, len(sim['t']) - 1, n_times).astype(int))
    deviation = np.abs(sim['u_log'][:, idx] - background)
    disturbed = deviation > fraction * max(deviation.max(), 1e-12)
    radius = np.array([np.percentile(distance[d], 95) if d.any() else np.nan for d in disturbed.T])
    t = sim['t'][idx]

    late = t > 0.8 * t[-1]
    if not disturbed[:, -1].any():
        outcome = 'patch vanished (back to uniform)'
    elif radius[-1] > 0.7 * distance.max():
        outcome = 'invaded the whole domain'
    elif np.nanmax(radius[late]) - np.nanmin(radius[late]) < 1:
        outcome = f'localised: front stopped at radius {radius[-1]:.1f}'
    else:
        outcome = 'still spreading'

    # number of separate spots: blobs raised by more than half of that frame's peak (so spots that dim
    # while pinching apart still count); frames with almost no rise left count as 0
    rise = sim['u_log'][:, idx] - background
    peak = rise.max(axis=0)
    spots = np.array([ndimage.label(to_grid(r > 0.5 * p, net['xy'], cfg.grid_size) == 1)[1]
                      if p > 0.1 * max(peak.max(), 1e-12) else 0 for r, p in zip(rise.T, peak)])
    if disturbed[:, -1].any() and (spots > 0).any():
        outcome += f"  ({spots[spots > 0][0]} → {spots[-1]} spots)"

    # front speed from the spreading phase: 20-80% of the way to the furthest radius reached
    r0 = radius[np.isfinite(radius)][0] if np.isfinite(radius).any() else np.nan
    span = np.nanmax(radius) - r0 if np.isfinite(r0) else 0
    growing = (radius > r0 + 0.2 * span) & (radius < r0 + 0.8 * span)
    speed = np.polyfit(t[growing], radius[growing], 1)[0] if span > 3 and growing.sum() > 2 else np.nan

    row = net['xy'][:, 1] == int(centre)
    order = np.argsort(net['xy'][row, 0])
    return {'t': t, 'radius': radius, 'area': disturbed.mean(axis=0), 'spots': spots, 'speed': speed, 'outcome': outcome,
            'kymograph': sim['u_log'][row][order][:, idx], 'kymo_x': net['xy'][row, 0][order] - centre,
            'max_distance': distance.max()}


def plot_seed_development(cfg, model, net, sim, dev):
    u_mid = model['steady_state'][0]  # colour midpoint: steady state (Turing) or saddle (bistable)
    lim = np.nanmax(np.abs(sim['u_log'] - u_mid)) or 1
    style = dict(cmap='bwr', vmin=u_mid - lim, vmax=u_mid + lim, interpolation='nearest')

    fig = plt.figure(figsize=(17, 8.5), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5])
    top, bottom = gs[0].subgridspec(1, cfg.n_snapshots), gs[1].subgridspec(1, 4)
    for k, i in enumerate(np.linspace(0, len(sim['t']) - 1, cfg.n_snapshots).astype(int)):
        ax = fig.add_subplot(top[k])
        ax.imshow(to_grid(sim['u_log'][:, i], net['xy'], cfg.grid_size), origin='lower', **style)
        ax.set_title(f"t = {sim['t'][i]:.3g}", fontsize=10)
        ax.set_axis_off()

    ax = fig.add_subplot(bottom[0])
    t, x = dev['t'], dev['kymo_x']
    ax.imshow(dev['kymograph'], origin='lower', aspect='auto', extent=[t[0], t[-1], x[0], x[-1]], **style)
    ax.set_xlabel('time')
    ax.set_ylabel('distance from centre along x')
    ax.set_title('Kymograph: u through the centre', fontsize=10)

    ax = fig.add_subplot(bottom[1])
    ax.plot(t, dev['radius'], color='tab:purple', lw=2)
    ax.axhline(dev['max_distance'], color='0.6', ls='--', lw=1)
    ax.text(t[0], dev['max_distance'], ' domain corner', va='bottom', fontsize=9, color='0.4')
    if np.isfinite(dev['speed']):
        ax.text(0.97, 0.05, f"front speed ≈ {dev['speed']:.3g} grid units / time", transform=ax.transAxes,
                ha='right', fontsize=9, color='tab:purple')
    ax.set_ylim(0, dev['max_distance'] * 1.1)
    ax.set_xlabel('time')
    ax.set_ylabel('front radius (95% of disturbed nodes inside)')
    ax.set_title('How far the patch has spread', fontsize=10)

    ax = fig.add_subplot(bottom[2])
    ax.plot(t, dev['area'], color='tab:purple', lw=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('time')
    ax.set_ylabel('fraction of nodes disturbed')
    ax.set_title('How much of the domain is patterned', fontsize=10)

    ax = fig.add_subplot(bottom[3])
    ax.step(t, dev['spots'], where='post', color='tab:purple', lw=2)
    ax.set_ylim(0, max(dev['spots'].max(), 1) * 1.1)
    ax.set_xlabel('time')
    ax.set_ylabel('number of spots')
    ax.set_title('Spot count (doubling = self-replication)', fontsize=10)

    radius, amplitude, *seed_shape = cfg.start_seed
    shape_name = seed_shape[0].__name__.replace('_outline_check', '').replace('_check', '') if seed_shape else 'disc'
    fig.suptitle(f"{model['name']}: seeded {shape_name} of radius {radius}"
                 f"{'' if amplitude is None else f', amplitude {amplitude:g}'}  →  {dev['outcome']}", fontsize=13)
    plt.show()
    return fig


if cfg.start_seed is not None:  # the main run was already seeded
    seeded_cfg, seeded_net, seeded_sim = cfg, net, sim
else:
    seeded_cfg = replace(cfg, start_seed=DEFAULT_SEED)
    seeded_net = {**net, 'initial': initial_state(seeded_cfg, model)}
    seeded_sim = run(seeded_cfg, model, diffusion, seeded_net, n_steps)
seed_dev = seed_development(seeded_cfg, model, seeded_net, seeded_sim)
print(f"Seeded start: {seed_dev['outcome']}"
      + (f", front speed ≈ {seed_dev['speed']:.3g}" if np.isfinite(seed_dev['speed']) else ''))
seed_fig = plot_seed_development(seeded_cfg, model, seeded_net, seeded_sim, seed_dev)

#%% 15. ANIMATION: the main run (cells 3-7) over time
# Left: u on the grid (same colours as the dashboard). Right: pattern amplitude std(u), with a dot
# marking the current frame - so you can see which phase (growth, saturation, rearranging) you're in.
# Same approach as animation.py (FuncAnimation + ffmpeg). Shown inline in the VS Code interactive
# window; set cfg.animation_path to also save an .mp4 (ffmpeg) or .gif (pillow).

import matplotlib.animation as animation


def animate_pattern(cfg, model, net, sim, res):
    u0 = model['steady_state'][0]
    t_start, t_end = cfg.animation_window or (sim['t'][0], sim['t'][-1])
    first, last = np.searchsorted(sim['t'], t_start), np.searchsorted(sim['t'], t_end, side='right') - 1
    frames = np.unique(np.linspace(first, last, cfg.n_frames).astype(int))
    print(f"Animation: {len(frames)} frames, one every {(sim['t'][last] - sim['t'][first]) / len(frames):.3g} time units")
    lim = max(np.nanmax(np.abs(sim['u_log'][:, frames] - u0)), 1e-3 * max(abs(u0), 1))

    fig, (ax_u, ax_amp) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True,
                                       gridspec_kw={'width_ratios': [1, 1.3]})
    im = ax_u.imshow(to_grid(sim['u_log'][:, frames[0]], net['xy'], cfg.grid_size), cmap='bwr',
                     vmin=u0 - lim, vmax=u0 + lim, origin='lower', interpolation='nearest')
    ax_u.set_axis_off()
    fig.colorbar(im, ax=ax_u, shrink=0.8)
    title = ax_u.set_title('')

    ax_amp.semilogy(sim['t'], res['amplitude'], color='k', lw=1.5)
    if cfg.animation_window is not None:
        ax_amp.axvspan(sim['t'][first], sim['t'][last], color='tab:purple', alpha=0.1, label='animated window')
        ax_amp.legend(fontsize=9, loc='lower right')
    dot, = ax_amp.semilogy([], [], 'o', color='tab:purple', ms=8)
    ax_amp.set_xlabel('time')
    ax_amp.set_ylabel('pattern amplitude std(u)')
    ax_amp.set_title(f"→ {res['observed']}, {res['shape']['type']}", fontsize=10)

    def update(k):
        i = frames[k]
        im.set_data(to_grid(sim['u_log'][:, i], net['xy'], cfg.grid_size))
        dot.set_data([sim['t'][i]], [res['amplitude'][i]])
        title.set_text(f"{model['name']}   t = {sim['t'][i]:.4g}")
        return im, dot, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / cfg.animation_fps,
                                  blit=False, repeat=True)

    if cfg.animation_path is not None:
        writer = 'pillow' if cfg.animation_path.endswith('.gif') else 'ffmpeg'
        ani.save(cfg.animation_path, writer=writer, fps=cfg.animation_fps)
        print(f"Saved animation to {cfg.animation_path}")

    try:  # inline player in the VS Code interactive window / Jupyter
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            display(HTML(ani.to_jshtml()))
            plt.close(fig)
            return ani
    except ImportError:
        pass
    plt.show()  # plain `python run_turing.py`: opens a window that plays the animation
    return ani


pattern_animation = animate_pattern(cfg, model, net, sim, res)

#%% 16. BEHAVIOUR MAP: simulate a grid of kinetic parameters and sort what happens (Pearson-style)
# Uses MODELS[cfg.model].behaviour_map = (x_param, x_values, y_param, y_values). Every point is a
# seeded run (cfg.start_seed, or MAP_SEED if that's None) of length cfg.T, sorted into:
#   dies             patch decays back to the uniform background
#   uniform red      the whole domain switches to another uniform state
#   oscillating      late pattern amplitude still swings by > 5%  (breathing spots, chaos)
#   moving           amplitude steady but the pattern keeps changing  (gliders, rearranging spots)
#   spots / holes / stripes & mixed   settled pattern, by blob roundness (cell 5)
# Linear theory can't predict any of this for Gray-Scott, so this is the only way to see it.
# Gray-Scott also gets the saddle-node line b = √a/2 - a: below it a second (red) uniform state exists.
# Cost: len(x_values) × len(y_values) runs - about 10 min for the 6×6 Gray-Scott map on a 50×50 grid.

MAP_SEED = (8, None, shapes.small_grid_check)
MAP_CLASSES = {'dies': '#8a8a85', 'uniform red': '#c9a227', 'oscillating': '#eb6834', 'moving': '#8b5fbf',
               'spots': '#2a78d6', 'holes': '#1baf7a', 'stripes & mixed': '#d6457a'}


def classify_run(model, sim, res, background):
    if res['observed'] == 'uniform (no pattern)':
        return 'dies' if abs(sim['u'].mean() - background) < 1e-2 * max(abs(background), 1) else 'uniform red'
    late = res['amplitude'][int(0.7 * len(res['amplitude'])):]
    if late.std() / late.mean() > 0.05:
        return 'oscillating'
    if res['observed'] != 'stationary pattern':
        return 'moving'
    return {'spots': 'spots', 'holes (inverse spots)': 'holes'}.get(res['shape']['type'], 'stripes & mixed')


def behaviour_map(cfg, net, n_steps):
    x_name, x_values, y_name, y_values = MODELS[cfg.model].behaviour_map
    cfg_m = replace(cfg, start_seed=cfg.start_seed or MAP_SEED)
    rows = []
    for y in y_values:
        for x in x_values:
            cfg_p = replace(cfg_m, model_params={**cfg.model_params, x_name: x, y_name: y}, Du=None, Dv=None)
            try:
                model_p = build_model(cfg_p)
                diffusion_p = design_diffusion(cfg_p, model_p)
            except ValueError as err:
                print(f"  {x_name}={x:g}, {y_name}={y:g}: skipped - {err}")
                continue
            growth_p = growth_rates(model_p, diffusion_p, net['eigvals'])
            net_p = {**net, 'initial': initial_state(cfg_p, model_p)}
            sim_p = run(cfg_p, model_p, diffusion_p, net_p, n_steps)
            res_p = analyse(cfg_p, model_p, net_p, sim_p, growth_p)
            kind = classify_run(model_p, sim_p, res_p, seed_background(model_p)[0][0])
            rows.append({'x': x, 'y': y, 'class': kind, 'u_grid': res_p['u_grid'], 'u0': model_p['steady_state'][0],
                         'n_blobs': res_p['shape']['n_blobs']})
            print(f"  {x_name}={x:g}, {y_name}={y:g}: {kind}"
                  + (f" ({res_p['shape']['n_blobs']} blobs)" if kind in ('spots', 'holes') else ''))
    return (x_name, x_values, y_name, y_values), rows


def plot_behaviour_map(cfg, axes_spec, rows):
    x_name, x_values, y_name, y_values = axes_spec
    lookup = {(r['x'], r['y']): r for r in rows}

    # thumbnails: one per run, framed in its class colour, highest y at the top
    fig = plt.figure(figsize=(1.7 * len(x_values) + 2.5, 1.7 * len(y_values) + 0.8), constrained_layout=True)
    gs = GridSpec(len(y_values), len(x_values), figure=fig)
    for i, y in enumerate(reversed(y_values)):
        for j, x in enumerate(x_values):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == len(y_values) - 1:
                ax.set_xlabel(f"{x_name} = {x:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{y_name} = {y:g}", fontsize=9)
            r = lookup.get((x, y))
            if r is None:
                continue
            lim = max(np.nanmax(np.abs(r['u_grid'] - r['u0'])), 1e-3 * max(abs(r['u0']), 1))
            ax.imshow(r['u_grid'], cmap='bwr', vmin=r['u0'] - lim, vmax=r['u0'] + lim,
                      origin='lower', interpolation='nearest')
            for spine in ax.spines.values():
                spine.set_color(MAP_CLASSES[r['class']])
                spine.set_linewidth(3)
            ax.set_title(r['class'], fontsize=8, color='0.2')
    from matplotlib.patches import Patch
    present = [c for c in MAP_CLASSES if any(r['class'] == c for r in rows)]
    fig.legend(handles=[Patch(facecolor='none', edgecolor=MAP_CLASSES[c], lw=3, label=c) for c in present],
               loc='outside right upper', frameon=False, fontsize=9)
    fig.suptitle(f"{cfg.model}: final pattern at t = {cfg.T:g} for each ({x_name}, {y_name})", fontsize=12)
    plt.show()

    # class map in parameter space
    fig2, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for c in present:
        pts = np.array([(r['x'], r['y']) for r in rows if r['class'] == c])
        ax.scatter(pts[:, 0], pts[:, 1], s=260, marker='s', color=MAP_CLASSES[c], label=c, zorder=3)
    if cfg.model == 'gray_scott':  # second uniform state exists below this line
        a = np.linspace(min(y_values), max(y_values), 200)
        ax.plot(np.sqrt(a) / 2 - a, a, color='k', lw=1.5, ls='--', label='saddle-node: b = √a/2 - a')
    current = {**MODELS[cfg.model].params, **cfg.model_params}
    ax.plot(current[x_name], current[y_name], marker='*', ms=16, color='k', mec='w', mew=1.5, zorder=5,
            label='current')
    ax.set_xlim(min(x_values) - 0.6 * np.ptp(x_values) / max(len(x_values) - 1, 1),
                max(x_values) + 0.6 * np.ptp(x_values) / max(len(x_values) - 1, 1))
    ax.set_ylim(min(y_values) - 0.6 * np.ptp(y_values) / max(len(y_values) - 1, 1),
                max(y_values) + 0.6 * np.ptp(y_values) / max(len(y_values) - 1, 1))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{cfg.model}: behaviour map ({cfg.grid_size}×{cfg.grid_size} grid, T = {cfg.T:g})")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    plt.show()
    return fig, fig2


if MODELS[cfg.model].behaviour_map is None:
    print(f"No behaviour_map set for '{cfg.model}' in MODELS - skipping the behaviour map.")
else:
    map_axes, map_rows = behaviour_map(cfg, net, n_steps)
    map_figs = plot_behaviour_map(cfg, map_axes, map_rows)

# %%
