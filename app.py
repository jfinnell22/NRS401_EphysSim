import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(page_title="Ion Channels & Excitability", layout="wide")
st.title("Workshop 3:Ion Channels, Excitability & Channelopathies")

# ------------------------------------------------------------
# Colorblind-friendly palette (Okabe–Ito inspired)
# ------------------------------------------------------------
COL_HEALTHY = "#6B7280"    # gray
COL_EPILEPTIC = "#E69F00"  # orange
COL_TREATED = "#0072B2"    # blue
COL_SHADE = "#E5E7EB"      # light gray shading

# ------------------------------------------------------------
# Shared plot helpers
# ------------------------------------------------------------

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_grid_major_minor(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.14)


def sigmoid(x):
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def mode_badge(text):
    st.markdown(
        f"<div style='display:inline-block; padding:6px 10px; border-radius:10px; "
        f"background:#eef2ff; border:1px solid #c7d2fe; font-weight:600;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _shade_mask(ax, t, mask, alpha=0.55, color=COL_SHADE):
    """Shade contiguous regions where mask is True."""
    m = mask.astype(int)
    if m.sum() == 0:
        return
    starts = np.where((m[1:] == 1) & (m[:-1] == 0))[0] + 1
    ends = np.where((m[1:] == 0) & (m[:-1] == 1))[0] + 1
    if m[0] == 1:
        starts = np.r_[0, starts]
    if m[-1] == 1:
        ends = np.r_[ends, len(m) - 1]
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=color, alpha=alpha, lw=0)


def _raster(ax, spike_times, y, color, lw=1.2, alpha=0.95, half_height=0.18, x_shift_ms=0.0):
    """Draw spike raster lines at spike_times.

    half_height: vertical half-length of each raster line (smaller -> less overlap).
    x_shift_ms: small horizontal shift (ms) added to spike_times for plotting only
                (does not modify underlying data) so overlaid conditions separate.
    """
    if spike_times.size == 0:
        return
    if x_shift_ms != 0.0:
        spike_times = spike_times + float(x_shift_ms)
    ax.vlines(spike_times, y - half_height, y + half_height, color=color, lw=lw, alpha=alpha)


# add missing helpers used by Module 2B
def _poisson_spikes(rate_hz, t_ms, rng):
    """Return boolean array of Poisson spikes for times t_ms (ms)."""
    if len(t_ms) < 2:
        dt_ms = 1.0
    else:
        dt_ms = float(t_ms[1] - t_ms[0])
    p = rate_hz * (dt_ms / 1000.0)  # probability per timestep
    p = float(np.clip(p, 0.0, 1.0))
    return rng.random(size=len(t_ms)) < p

def _make_windows_mask(t_ms, spike_bool_array, win_ms):
    """Return boolean mask with a window of length win_ms after each spike."""
    mask = np.zeros_like(t_ms, dtype=bool)
    if not np.any(spike_bool_array):
        return mask
    dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0
    steps = max(1, int(np.round(win_ms / dt)))
    spike_idx = np.where(spike_bool_array)[0]
    for idx in spike_idx:
        end = min(idx + steps, len(mask))
        mask[idx:end] = True
    return mask

# ============================================================
# MODULE 1 — Voltage‑gated Na⁺ (templates + trains)
# (kept from the previous consolidated builds)
# ============================================================

def spike_only_waveform(
    t,
    V_start=-40.0,
    V_peak=40.0,
    V_rest=-65.0,
    V_ahp=-78.0,
    t_up=0.14,
    k_up=0.016,
    t_down=0.40,
    k_down=0.030,
    t_ahp_on=0.48,
    k_ahp_on=0.030,
    t_ahp_off=1.35,
    k_ahp_off=0.160,
    t_rec=1.05,
    k_rec=0.30,
):
    up = sigmoid((t - t_up) / (k_up + 1e-12))
    down = sigmoid((t - t_down) / (k_down + 1e-12))
    spike = (V_peak - V_start) * (up - down)

    ahp_on = sigmoid((t - t_ahp_on) / (k_ahp_on + 1e-12))
    ahp_off = sigmoid((t - t_ahp_off) / (k_ahp_off + 1e-12))
    ahp = (V_ahp - V_rest) * (ahp_on - ahp_off)

    V = V_start + spike + ahp
    rec = sigmoid((t - t_rec) / (k_rec + 1e-12))
    return (1 - rec) * V + rec * V_rest


def na_current_for_spike(t, t_up=0.14, amp=1.0, sigma=0.085, tail_amp=0.10, tail_tau=0.45):
    g = np.exp(-0.5 * ((t - t_up) / (sigma + 1e-12)) ** 2)
    g = g / (np.max(g) + 1e-12)
    Ina = -amp * g
    tail = np.zeros_like(t)
    idx = t >= t_up
    tail[idx] = np.exp(-(t[idx] - t_up) / (tail_tau + 1e-12))
    Ina += -(tail_amp * amp) * tail
    return Ina


def base_params():
    return dict(
        dt_ms=0.05,
        tstop_ms=120.0,
        stim_on=20.0,
        stim_off=110.0,
        V_rest=-65.0,
        V_reset=-72.0,
        tau_m=12.0,
        drive=3.35,
        refractory_ms=3.8,
        V_thr=-40.0,
        V_peak=40.0,
        V_ahp=-78.0,
        t_up=0.14,
        k_up=0.016,
        t_down=0.40,
        k_down=0.030,
        t_ahp_on=0.48,
        k_ahp_on=0.030,
        t_ahp_off=1.35,
        k_ahp_off=0.160,
        t_rec=1.05,
        k_rec=0.30,
        ina_amp=1.0,
        ina_sigma=0.085,
        ina_tail_amp=0.10,
        ina_tail_tau=0.45,
        spike_template_ms=2.0,
    )


def apply_voltage_sensitivity(p, direction):
    if direction == "Increase":
        p["V_thr"] -= 15.0
    elif direction == "Decrease":
        p["V_thr"] += 10.0


def apply_inactivation(p, direction):
    if direction == "Slower":
        p["V_peak"] = 52.0
        p["t_down"] = 1.20
        p["k_down"] = 0.18
        p["t_ahp_on"] = 1.55
        p["k_ahp_on"] = 0.10
        p["t_ahp_off"] = 3.20
        p["k_ahp_off"] = 0.35
        p["t_rec"] = 2.40
        p["k_rec"] = 0.55
        p["spike_template_ms"] = 4.0
        p["ina_sigma"] = 0.18
        p["ina_tail_amp"] = 0.55
        p["ina_tail_tau"] = 1.55
    elif direction == "Faster":
        p["V_peak"] = 36.0
        p["t_down"] = 0.22
        p["k_down"] = 0.010
        p["t_ahp_on"] = 0.30
        p["k_ahp_on"] = 0.016
        p["t_ahp_off"] = 0.85
        p["k_ahp_off"] = 0.085
        p["t_rec"] = 0.62
        p["k_rec"] = 0.16
        p["spike_template_ms"] = 1.15
        p["ina_sigma"] = 0.045
        p["ina_tail_amp"] = 0.03
        p["ina_tail_tau"] = 0.18


def apply_conductance(p, direction):
    if direction == "Increase":
        p["ina_amp"] *= 1.5
        p["V_peak"] = max(p["V_peak"], 48.0)
    elif direction == "Decrease":
        p["ina_amp"] *= 0.45
        p["V_peak"] = min(p["V_peak"], 12.0)
        p["V_thr"] += 5.0


def params_from_mechanisms(vs_choice=None, inact_choice=None, g_choice=None):
    p = base_params()
    if vs_choice:
        apply_voltage_sensitivity(p, vs_choice)
    if inact_choice:
        apply_inactivation(p, inact_choice)
    if g_choice:
        apply_conductance(p, g_choice)
    return p


def simulate_spike_train_from_params(p):
    dt = p["dt_ms"]
    t = np.arange(0.0, p["tstop_ms"] + dt, dt)
    V = np.ones_like(t) * p["V_rest"]
    INa = np.zeros_like(t)

    t_sp = np.arange(0.0, p["spike_template_ms"] + dt, dt)
    V_sp = spike_only_waveform(
        t_sp,
        V_start=p["V_thr"],
        V_peak=p["V_peak"],
        V_rest=p["V_rest"],
        V_ahp=p["V_ahp"],
        t_up=p["t_up"],
        k_up=p["k_up"],
        t_down=p["t_down"],
        k_down=p["k_down"],
        t_ahp_on=p["t_ahp_on"],
        k_ahp_on=p["k_ahp_on"],
        t_ahp_off=p["t_ahp_off"],
        k_ahp_off=p["k_ahp_off"],
        t_rec=p["t_rec"],
        k_rec=p["k_rec"],
    )
    INa_sp = na_current_for_spike(
        t_sp,
        t_up=p["t_up"],
        amp=p["ina_amp"],
        sigma=p["ina_sigma"],
        tail_amp=p["ina_tail_amp"],
        tail_tau=p["ina_tail_tau"],
    )

    sp_len = len(t_sp)
    refrac_steps = int(np.round(p["refractory_ms"] / dt))

    i = 1
    while i < len(t):
        in_stim = (p["stim_on"] <= t[i] <= p["stim_off"])
        Ieff = p["drive"] if in_stim else 0.0
        dV = ((p["V_rest"] - V[i - 1]) / p["tau_m"] + Ieff) * dt
        V[i] = V[i - 1] + dV

        if in_stim and V[i] >= p["V_thr"]:
            end = min(i + sp_len, len(t))
            nfit = end - i
            offset = V[i] - V_sp[0]
            V[i:end] = V_sp[:nfit] + offset
            INa[i:end] = INa_sp[:nfit]
            j = end
            j_end = min(j + refrac_steps, len(t))
            if j < len(t):
                V[j:j_end] = p["V_reset"]
            i = j_end
            continue
        i += 1

    return t, V, INa


def simulate_spike_train(condition_key):
    if condition_key == "Baseline":
        p = base_params()
    elif condition_key == "Voltage sensitivity increased":
        p = params_from_mechanisms(vs_choice="Increase")
    elif condition_key == "Inactivation slowed":
        p = params_from_mechanisms(inact_choice="Slower")
    elif condition_key == "Na conductance reduced":
        p = params_from_mechanisms(g_choice="Decrease")
    else:
        p = base_params()
    return simulate_spike_train_from_params(p)


def _first_spike_time(t, V, stim_on=0.0):
    idx0 = np.searchsorted(t, stim_on)
    Vw = V[idx0:]
    tw = t[idx0:]
    if Vw.size < 3:
        return None
    above = Vw > 0.0
    crossings = np.where((~above[:-1]) & (above[1:]))[0]
    if crossings.size == 0:
        pk = np.argmax(Vw)
        return float(tw[pk])
    return float(tw[crossings[0] + 1])


def plot_module1(t, V0, INa0, V1, INa1, label1, color1, stim_on=20.0):
    fig, (ax_zoom, ax_na, ax_vm) = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=False)

    t_first = _first_spike_time(t, V1, stim_on=stim_on)
    if t_first is None:
        t_first = stim_on + 10.0
    z0 = max(0.0, t_first - 4.0)
    z1 = min(float(t[-1]), t_first + 12.0)

    ax_zoom.plot(t, V0, color="#9ca3af", lw=2.4, label="Baseline")
    ax_zoom.plot(t, V1, color=color1, lw=2.8, label=label1)
    ax_zoom.set_xlim(z0, z1)
    ax_zoom.set_ylim(-90, 70)
    ax_zoom.set_ylabel("Vm (mV)")
    ax_zoom.set_title("Single action potential (zoomed)")
    add_grid_major_minor(ax_zoom)
    ax_zoom.legend(frameon=False, loc="upper right")
    style_axes(ax_zoom)

    ax_na.plot(t, INa0, color="#9ca3af", lw=2.2, alpha=0.95)
    ax_na.plot(t, INa1, color="#7c3aed", lw=2.2, alpha=0.95)
    ax_na.axhline(0, color="#9ca3af", lw=1)
    ax_na.set_xlim(z0, z1)
    ax_na.set_ylabel("Na⁺ current (a.u.)")
    ax_na.set_title("Na⁺ current")
    add_grid_major_minor(ax_na)
    style_axes(ax_na)

    vals = np.concatenate([INa0[(t >= z0) & (t <= z1)], INa1[(t >= z0) & (t <= z1)]])
    vals = vals[np.isfinite(vals)]
    m = max(np.max(np.abs(vals)), 0.2) if vals.size else 1.0
    ax_na.set_ylim(-1.25 * m, 0.25 * m)

    ax_vm.plot(t, V0, color="#9ca3af", lw=2.6, label="Baseline")
    ax_vm.plot(t, V1, color=color1, lw=2.8, label=label1)
    ax_vm.set_ylabel("Vm (mV)")
    ax_vm.set_xlabel("Time (ms)")
    ax_vm.set_title("Spike train")
    ax_vm.set_ylim(-90, 60)
    add_grid_major_minor(ax_vm)
    style_axes(ax_vm)

    fig.tight_layout()
    return fig


# ============================================================
# MODULE 2 — Ligand‑gated channels (merged baseline + one change)
# ============================================================

def syn_g(t_ms, g_peak_nS=1.0, tau_rise_ms=0.5, tau_decay_ms=5.0, t_on_ms=10.0):
    g = np.zeros_like(t_ms, dtype=float)
    tt = t_ms - t_on_ms
    idx = tt >= 0
    tt2 = tt[idx]
    raw = np.exp(-tt2 / tau_decay_ms) - np.exp(-tt2 / tau_rise_ms)
    if raw.size > 0:
        peak = raw.max()
        if peak > 0:
            raw = raw / peak
        g[idx] = g_peak_nS * raw
    return g


def synapse_baseline_params(receptor: str):
    if receptor.startswith("Excitatory"):
        return dict(Erev=0.0, g_peak=1.5, tau_rise=0.4, tau_decay=4.0, title="EPSC")
    return dict(Erev=-70.0, g_peak=2.0, tau_rise=0.6, tau_decay=10.0, title="IPSC")


def apply_synapse_change(base: dict, mechanism: str, direction: str):
    p = dict(base)
    if mechanism == "None (baseline only)":
        return p
    if mechanism == "Ligand binding efficacy":
        p["g_peak"] *= 0.6 if direction == "Decrease" else 1.6
    elif mechanism == "Channel kinetics (rise/decay)":
        fac = 1.8 if direction == "Slower" else 0.55
        p["tau_rise"] *= fac
        p["tau_decay"] *= fac
    return p


def voltage_clamp_IV_from_params(params: dict):
    V_steps = np.arange(-90, 51, 10)
    t = np.arange(0.0, 40.0 + 0.05, 0.05)
    g = syn_g(t, g_peak_nS=params["g_peak"], tau_rise_ms=params["tau_rise"], tau_decay_ms=params["tau_decay"], t_on_ms=10.0)
    I_peaks = []
    for V in V_steps:
        I_t = g * (V - params["Erev"])
        I_peak = I_t.min() if np.abs(I_t.min()) > np.abs(I_t.max()) else I_t.max()
        I_peaks.append(I_peak)
    return V_steps, np.array(I_peaks)


def current_clamp_syn_current_from_params(params: dict, Vm_hold=-65.0):
    t = np.arange(0.0, 60.0 + 0.05, 0.05)
    gsyn = syn_g(t, g_peak_nS=params["g_peak"], tau_rise_ms=params["tau_rise"], tau_decay_ms=params["tau_decay"], t_on_ms=10.0)
    I_syn = gsyn * (Vm_hold - params["Erev"])
    return t, I_syn, params["title"]


def plot_module2_current_and_iv(base: dict, cond: dict, show_condition: bool):
    fig, (ax_cur, ax_iv) = plt.subplots(2, 1, figsize=(11.0, 9.0))

    t0, I0, title0 = current_clamp_syn_current_from_params(base)
    ax_cur.plot(t0, I0, color=COL_HEALTHY, lw=2.6, label="Baseline")
    if show_condition:
        t1, I1, _ = current_clamp_syn_current_from_params(cond)
        ax_cur.plot(t1, I1, color=COL_TREATED, lw=2.8, label="Condition")
    ax_cur.axhline(0, color=COL_HEALTHY, lw=1)
    ax_cur.set_xlabel("Time (ms)")
    ax_cur.set_ylabel("Synaptic current (pA)")
    ax_cur.set_title(f"{title0} trace")
    add_grid_major_minor(ax_cur)
    ax_cur.legend(frameon=False)
    style_axes(ax_cur)

    V0, IV0 = voltage_clamp_IV_from_params(base)
    ax_iv.plot(V0, IV0, marker="o", lw=2.4, color=COL_HEALTHY, label="Baseline")
    if show_condition:
        V1, IV1 = voltage_clamp_IV_from_params(cond)
        ax_iv.plot(V1, IV1, marker="o", lw=2.8, color=COL_TREATED, label="Condition")
    ax_iv.axhline(0, color=COL_HEALTHY, lw=1)
    ax_iv.set_xlabel("Clamp voltage (mV)")
    ax_iv.set_ylabel("Peak synaptic current (pA)")
    ax_iv.set_title("I–V curve")
    add_grid_major_minor(ax_iv)
    ax_iv.legend(frameon=False)
    style_axes(ax_iv)

    fig.tight_layout()
    return fig


# ============================================================
# MODULE 2B — Epilepsy model (cleaner, obvious)
# ============================================================

def simulate_epilepsy_condition(T_ms=450.0, dt_ms=0.2, seed=5, state="Healthy", drug="None"):
    """Return a condition dict for Module 2B.

    Output:
      - interneuron spike times
      - pyramidal spike times
      - inhibitory window mask
      - GABA conductance proxy

    Pedagogical behavior:
      Healthy: strong/long inhibitory windows → pyramidal spikes largely outside windows.
      Epileptic: weak/short windows → pyramidal spikes increase and occur inside windows.
      Drug: rescues windows; barbiturate rescues more than benzodiazepine but not to silence.

    Drug effects also apply in Healthy (small shift toward fewer spikes).
    """

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_ms + dt_ms, dt_ms)

    # Poisson excitation is IDENTICAL across states to emphasize that inhibition changes output.
    exc_IN = _poisson_spikes(220.0, t, rng)
    exc_PY = _poisson_spikes(300.0, t, rng)

    # LIF-ish voltages
    Vrest, Vreset = -65.0, -70.0
    Vthr_IN, Vthr_PY = -47.0, -54.0
    tau_IN, tau_PY = 12.0, 18.0

    # excitatory conductance proxy
    gE_IN = np.zeros_like(t)
    gE_PY = np.zeros_like(t)
    tau_e = 3.0
    gE_IN_jump = 1.10
    gE_PY_jump = 1.40

    g_to_I = 0.030
    Eexc = 0.0

    # --- state & drug specific excitability adjustments (pyramidal) ---
    # Make pyramidal cells much more excitable in the untreated Epileptic state
    # so the increase in AP count is visually obvious for students.
    if state == "Epileptic":
        Vthr_PY = -50.0        # easier to reach threshold
        tau_PY = 14.0          # slightly faster membrane -> higher rates possible
        gE_PY_jump *= 2.2      # larger effective EPSP per excitatory event

    # Drug-specific rescues (only modify PY excitability, keep IN behavior consistent)
    if state == "Epileptic" and drug == "Benzodiazepine":
        # partial rescue: raise threshold a little, modestly reduce drive
        Vthr_PY = -52.0
        gE_PY_jump *= 0.9
        # benzodiazepine boosts phasic inhibition via gI_amp (set earlier), leave tonic small

    if state == "Epileptic" and drug == "Barbiturate":
        # stronger rescue: restore threshold near healthy and substantially reduce effective EPSP
        Vthr_PY = -54.0
        gE_PY_jump *= 0.35
        # increase tonic-like stabilization and phasic gain (gI_amp from earlier will amplify effect)
        # (tonic_ref already set for barbiturate above)

    V_IN = np.ones_like(t) * Vrest
    V_PY = np.ones_like(t) * Vrest

    spk_IN = np.zeros_like(t, dtype=bool)
    spk_PY = np.zeros_like(t, dtype=bool)

    # --- Inhibitory window settings (ms) ---
    # Healthy baseline
    win_base = 14.0
    # Epileptic baseline
    win_epi = 3.0

    # Drug adjustments (apply in both states)
    if drug == "None":
        win_drug = 0.0
        gI_amp = 1.0
        tonic_ref = 0.0
    elif drug == "Benzodiazepine":
        win_drug = 5.0
        gI_amp = 1.35
        tonic_ref = 0.0
    else:  # Barbiturate
        win_drug = 8.0
        gI_amp = 1.55
        tonic_ref = 2.0  # small additional refractory-like stabilization (not full silence)

    # final window
    if state == "Healthy":
        win = win_base + 0.25 * win_drug
    else:
        win = win_epi + win_drug

    win = max(1.0, win)

    # GABA conductance proxy (for plotting)
    gI_peak_healthy = 2.0
    gI_peak_epi = 0.45
    gI_peak = (gI_peak_healthy if state == "Healthy" else gI_peak_epi) * gI_amp

    # Refractory timers
    ref_IN = 0.0
    ref_PY = 0.0

    for k in range(1, len(t)):
        # decay excitation
        gE_IN[k] = gE_IN[k-1] * np.exp(-dt_ms / tau_e)
        gE_PY[k] = gE_PY[k-1] * np.exp(-dt_ms / tau_e)

        if exc_IN[k]:
            gE_IN[k] += gE_IN_jump
        if exc_PY[k]:
            gE_PY[k] += gE_PY_jump

        # interneuron
        if ref_IN > 0:
            V_IN[k] = Vreset
            ref_IN -= dt_ms
        else:
            I_exc = g_to_I * (gE_IN[k] * (Eexc - V_IN[k-1]))
            V_IN[k] = V_IN[k-1] + ((Vrest - V_IN[k-1]) / tau_IN + I_exc) * dt_ms
            if V_IN[k] >= Vthr_IN:
                spk_IN[k] = True
                V_IN[k] = 40.0
                ref_IN = 2.0
                # inhibition window silences PY (pedagogical)
                ref_PY = max(ref_PY, win + tonic_ref)

        # pyramidal
        if ref_PY > 0:
            V_PY[k] = Vreset
            ref_PY -= dt_ms
        else:
            I_exc = g_to_I * (gE_PY[k] * (Eexc - V_PY[k-1]))
            V_PY[k] = V_PY[k-1] + ((Vrest - V_PY[k-1]) / tau_PY + I_exc) * dt_ms
            if V_PY[k] >= Vthr_PY:
                spk_PY[k] = True
                V_PY[k] = 35.0
                ref_PY = 2.0

    # inhibition mask: windows after interneuron spikes
    inh_mask = _make_windows_mask(t, spk_IN, win)

    # GABA conductance proxy trace from IN spikes
    gI = np.zeros_like(t)
    tau_i_plot = 12.0
    for idx in np.where(spk_IN)[0]:
        gI[idx] += gI_peak
    for k in range(1, len(t)):
        gI[k] += gI[k-1] * np.exp(-dt_ms / tau_i_plot)

    return {
        "t": t,
        "spk_IN": spk_IN,
        "spk_PY": spk_PY,
        "inh_mask": inh_mask,
        "gI": gI,
        "win": win,
    }


def schematic_fallback():
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.6, "IN → GABA_A → PY", ha="center", va="center", fontsize=14, weight="600")
    ax.text(0.5, 0.35, "Interneuron spikes trigger inhibitory windows\nthat gate pyramidal firing", ha="center", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def plot_module2B_grid(baseline, selection, sel_title, sel_color):
    """Overlay baseline and selection similar to Modules 1 & 2.

    Rows (overlaid):
      1) interneuron spike raster (baseline gray, selection colored)
      2) GABA conductance proxy (overlaid traces) — plotted on its own amplitude scale
      3) pyramidal spike raster with inhibitory windows (both shaded)
    """
    t = baseline["t"]
    T = t[-1]

    # ensure we index times from boolean spike masks
    tIN_base = t[np.where(baseline["spk_IN"])]
    tPY_base = t[np.where(baseline["spk_PY"])]

    tIN_sel = t[np.where(selection["spk_IN"])]
    tPY_sel = t[np.where(selection["spk_PY"])]

    # common gI scale so overlays are interpretable
    g0 = baseline["gI"]
    g1 = selection["gI"]
    gmax = max(np.nanmax(g0), np.nanmax(g1), 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(13.6, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 0.8, 1.0]})

    # Row 1: IN raster (overlay) - keep compact vertical space
    _raster(axes[0], tIN_base, y=1.2, color=COL_HEALTHY, lw=1.2)
    _raster(axes[0], tIN_sel,  y=0.6, color=sel_color, lw=1.6)
    axes[0].set_ylim(0, 1.6)
    axes[0].set_yticks([])
    axes[0].set_ylabel("IN")
    axes[0].set_title("Interneuron spikes (baseline vs selection)")
    add_grid_major_minor(axes[0])
    style_axes(axes[0])
    p0 = axes[0].plot([], [], color=COL_HEALTHY, lw=2.4, label="Healthy baseline")[0]
    p1 = axes[0].plot([], [], color=sel_color, lw=2.8, label=sel_title)[0]
    axes[0].legend(handles=[p0, p1], frameon=False, loc="upper right")

    # Row 2: gI trace (overlay) with clear amplitude axis
    axes[1].plot(t, g0, color=COL_HEALTHY, lw=2.4, label="Healthy baseline")
    axes[1].plot(t, g1, color=sel_color, lw=2.6, alpha=0.95, label=sel_title)
    axes[1].set_ylabel("g_GABA (a.u.)")
    axes[1].set_title("GABA_A output (proxy)")
    axes[1].set_ylim(0, gmax * 1.08)
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    add_grid_major_minor(axes[1])
    style_axes(axes[1])
    axes[1].legend(frameon=False, loc="upper right")

    # Row 3: PY raster + both inhibition window shadings
    _shade_mask(axes[2], t, baseline["inh_mask"], alpha=0.22, color=COL_SHADE)
    _shade_mask(axes[2], t, selection["inh_mask"], alpha=0.28, color=sel_color)

    # add small horizontal offsets so baseline vs selection don't overlap
    _raster(axes[2], tPY_base, y=1.2, color=COL_HEALTHY, lw=1.4, x_shift_ms=-0.4)
    _raster(axes[2], tPY_sel,  y=0.6, color=sel_color, lw=1.6, x_shift_ms=0.4)

    axes[2].set_ylim(0, 1.6)
    axes[2].set_yticks([])
    axes[2].set_ylabel("PY")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Pyramidal spikes (shaded = inhibitory windows)")
    add_grid_major_minor(axes[2])
    style_axes(axes[2])

    for ax in axes:
        ax.set_xlim(0, float(T))

    fig.suptitle("Module 2B: Interneuron gating of pyramidal firing — baseline vs selection", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_module2B_trio(baseline, epileptic, treated, drug_label):
    """Three-row overlaid comparison: Healthy, Epileptic, Epileptic+Drug.
    Rows:
      1) IN spike rasters (overlaid)
      2) GABA conductance proxy traces (overlaid, same amplitude scale)
      3) PY spike rasters with inhibitory window shadings (overlaid)
    """
    t = baseline["t"]
    T = t[-1]

    tIN_base = t[np.where(baseline["spk_IN"])]
    tPY_base = t[np.where(baseline["spk_PY"])]

    tIN_epi = t[np.where(epileptic["spk_IN"])]
    tPY_epi = t[np.where(epileptic["spk_PY"])]

    tIN_tr = t[np.where(treated["spk_IN"])]
    tPY_tr = t[np.where(treated["spk_PY"])]

    # common gI scale for fair visual comparison
    gvals = np.concatenate([baseline["gI"], epileptic["gI"], treated["gI"]])
    gmax = max(np.nanmax(gvals), 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 0.8, 1.0]})

    # Row 1: IN rasters (overlaid; vertical offsets for clarity)
    _raster(axes[0], tIN_base, y=1.2, color=COL_HEALTHY, lw=1.4)
    _raster(axes[0], tIN_epi,  y=0.8, color=COL_EPILEPTIC, lw=1.6)
    _raster(axes[0], tIN_tr,   y=0.4, color=COL_TREATED, lw=1.6)
    axes[0].set_ylim(0, 1.6)
    axes[0].set_yticks([])
    axes[0].set_ylabel("IN")
    axes[0].set_title("Interneuron spikes: Healthy vs Epileptic vs Epileptic+Drug")
    add_grid_major_minor(axes[0])
    style_axes(axes[0])
    p0 = axes[0].plot([], [], color=COL_HEALTHY, lw=2.6, label="Healthy")[0]
    p1 = axes[0].plot([], [], color=COL_EPILEPTIC, lw=2.6, label="Epileptic")[0]
    p2 = axes[0].plot([], [], color=COL_TREATED, lw=2.6, label=f"Epileptic + {drug_label}")[0]
    axes[0].legend(handles=[p0, p1, p2], frameon=False, loc="upper right")

    # Row 2: gI conductance proxies (overlaid, same scale)
    axes[1].plot(t, baseline["gI"], color=COL_HEALTHY, lw=2.6, label="Healthy")
    axes[1].plot(t, epileptic["gI"], color=COL_EPILEPTIC, lw=2.6, label="Epileptic", alpha=0.95)
    axes[1].plot(t, treated["gI"], color=COL_TREATED, lw=2.8, label=f"Epileptic + {drug_label}", alpha=0.98)
    axes[1].set_ylabel("g_GABA (a.u.)")
    axes[1].set_title("GABA_A output (proxy)")
    axes[1].set_ylim(0, gmax * 1.08)
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    add_grid_major_minor(axes[1])
    style_axes(axes[1])
    axes[1].legend(frameon=False, loc="upper right")

    # Row 3: PY rasters + inhibitory window shading (overlaid)
    _shade_mask(axes[2], t, baseline["inh_mask"], alpha=0.18, color=COL_SHADE)
    _shade_mask(axes[2], t, epileptic["inh_mask"], alpha=0.28, color=COL_EPILEPTIC)
    _shade_mask(axes[2], t, treated["inh_mask"], alpha=0.30, color=COL_TREATED)

    # add small horizontal offsets so baseline vs selection don't overlap
    _raster(axes[2], tPY_base, y=1.2, color=COL_HEALTHY, lw=1.4, x_shift_ms=-0.6)
    _raster(axes[2], tPY_epi,  y=0.8, color=COL_EPILEPTIC, lw=1.6, x_shift_ms=0.0)
    _raster(axes[2], tPY_tr,   y=0.4, color=COL_TREATED, lw=1.6, x_shift_ms=0.6)

    axes[2].set_ylim(0, 1.6)
    axes[2].set_yticks([])
    axes[2].set_ylabel("PY")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Pyramidal spikes (shaded = inhibitory windows)")
    add_grid_major_minor(axes[2])
    style_axes(axes[2])

    for ax in axes:
        ax.set_xlim(0, float(T))

    fig.suptitle("Module 2B: Healthy vs Epileptic vs Epileptic+Drug (overlaid)", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
# Sidebar navigation
# ============================================================

with st.sidebar:
    st.markdown("### Navigation")
    module = st.radio(
        "Choose module",
        [
            "MODULE 1A — Voltage-gated Na⁺ channels",
            "MODULE 1B — Channelopathies",
            "MODULE 2 — Ligand-gated channels",
            "MODULE 2B — Case Study: Epilepsy",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Qualitative traces • Designed for visual reasoning")


# ============================================================
# MODULE 1A
# ============================================================

if module.startswith("MODULE 1A"):
    section_header(
        "MODULE 1A — Exploring Structural/Functional Relationships",
        "Visualizing structural changes in action potentials, current, and spike trains.",
    )
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        manipulation = st.radio(
            "Structural feature",
            [
                "Normal (baseline)",
                "Reduced Threshold",
                "Slowed Inactivation",
                "Reduced Conductance",
            ],
            index=0,
        )
        # Map UI labels to the internal keys expected by simulate_spike_train(...)
        if manipulation == "Normal (baseline)":
            cond_key = "Baseline"
        elif manipulation == "Reduced Threshold":
            cond_key = "Voltage sensitivity increased"
        elif manipulation == "Slowed Inactivation":
            cond_key = "Inactivation slowed"
        elif manipulation == "Reduced Conductance":
            cond_key = "Na conductance reduced"
        else:
            cond_key = "Baseline"
        st.markdown("---")
        st.caption("Stimulus strength is held constant across conditions.")

    with col2:
        t0, V0, INa0 = simulate_spike_train("Baseline")
        t1, V1, INa1 = simulate_spike_train(cond_key)
        fig = plot_module1(t0, V0, INa0, V1, INa1, label1="Condition", color1="#2563eb", stim_on=base_params()["stim_on"])
        st.pyplot(fig, clear_figure=True, use_container_width=True)


# ============================================================
# MODULE 1B
# ============================================================

elif module.startswith("MODULE 1B"):
    section_header(
        "MODULE 1B — Channelopathies",
        "Use the vignette to build the appropriate channelopathy",
    )
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        st.markdown("**Step 1 — How many properties change?**")
        n_mech = st.radio("Select one", ["One mechanism", "Two mechanisms"], index=0)
        st.markdown("---")
        st.markdown("**Step 2 — Choose mechanism(s)**")
        mech_options = ["Voltage sensitivity", "Inactivation kinetics", "Na⁺ conductance"]
        if n_mech == "One mechanism":
            mechs = [st.selectbox("Mechanism", mech_options, index=0)]
        else:
            m1 = st.selectbox("Mechanism 1", mech_options, index=0)
            remaining = [m for m in mech_options if m != m1]
            m2 = st.selectbox("Mechanism 2", remaining, index=0)
            mechs = [m1, m2]
        st.markdown("---")
        st.markdown("**Step 3 — Direction**")
        vs_choice = None
        inact_choice = None
        g_choice = None
        if "Voltage sensitivity" in mechs:
            vs_choice = st.radio("Voltage sensitivity", ["Increase", "Decrease"], index=0)
            vs_choice = "Increase" if vs_choice.startswith("Increase") else "Decrease"
        if "Inactivation kinetics" in mechs:
            inact_choice = st.radio("Inactivation kinetics", ["Slower", "Faster"], index=0)
            inact_choice = "Slower" if inact_choice.startswith("Slower") else "Faster"
        if "Na⁺ conductance" in mechs:
            g_choice = st.radio("Na⁺ conductance", ["Increase", "Decrease"], index=1)
            g_choice = "Increase" if g_choice.startswith("Increase") else "Decrease"
        st.markdown("---")
        st.caption("Stimulus strength is held constant.")

    with col2:
        p0 = base_params()
        t0, V0, INa0 = simulate_spike_train_from_params(p0)
        p1 = params_from_mechanisms(vs_choice=vs_choice, inact_choice=inact_choice, g_choice=g_choice)
        t1, V1, INa1 = simulate_spike_train_from_params(p1)
        fig = plot_module1(t0, V0, INa0, V1, INa1, label1="Student model", color1="#dc2626", stim_on=p0["stim_on"])
        st.pyplot(fig, clear_figure=True, use_container_width=True)


# ============================================================
# MODULE 2
# ============================================================

elif module.startswith("MODULE 2A —"):
    section_header(
        "MODULE 2A — Ligand‑Gated Ion Channels",
        "Exploring functional changes in ligand-gated ion channels",
    )
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        receptor = st.radio("Receptor type", ["Excitatory (AMPA-like)", "Inhibitory (GABA_A-like)"], index=0)
        st.markdown("---")
        mechanism = st.radio(
            "Mechanistic change",
            ["None (baseline only)", "Ligand binding efficacy", "Channel kinetics (rise/decay)"],
            index=0,
        )
        direction = ""
        if mechanism == "Ligand binding efficacy":
            direction = st.radio("Direction", ["Decrease", "Increase"], index=0)
        elif mechanism == "Channel kinetics (rise/decay)":
            direction = st.radio("Direction", ["Slower", "Faster"], index=0)
            st.caption("Kinetics affects time course; I–V is largely unchanged here.")
        st.markdown("---")
        mode_badge("Current trace (top) + I–V curve (bottom)")

    with col2:
        base = synapse_baseline_params(receptor)
        cond = apply_synapse_change(base, mechanism=mechanism, direction=direction)
        fig = plot_module2_current_and_iv(base, cond, show_condition=(mechanism != "None (baseline only)"))
        st.pyplot(fig, clear_figure=True, use_container_width=True)


# ============================================================
# MODULE 2B
# ============================================================

else:
    section_header(
        "MODULE 2B — Case Study: Epilepsy",
        "Exploring how changes in inhibitory GABAergic tone: Epilepsy and Treatments",
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        # Healthy default
        state = st.radio("State", ["Healthy", "Epileptic"], index=0)
        drug = st.radio("Drug", ["None", "Benzodiazepine", "Barbiturate"], index=0)

        st.markdown("---")
        st.markdown("**How to read the figure**")
        st.write(
            "• Row 1: Interneuron spike train.\n"
            "• Row 2: GABA_A Current in Pyramidal Neurons.\n"
            "• Row 3: Pyramidal spike train.\n\n"
            "Shaded regions mark the inhibitory window.\n"
        )

        

    with right:
        baseline = simulate_epilepsy_condition(state="Healthy", drug="None")

        # Display logic: compute and show only what's needed
        if state == "Healthy":
            if drug == "None":
                # show healthy baseline only
                fig = plot_module2B_grid(baseline, baseline, sel_title="Healthy baseline", sel_color=COL_HEALTHY)
            else:
                # show how drug affects healthy baseline
                treated = simulate_epilepsy_condition(state="Healthy", drug=drug)
                fig = plot_module2B_grid(baseline, treated, sel_title=f"Healthy + {drug}", sel_color=COL_TREATED)
        else:  # state == "Epileptic"
            epileptic = simulate_epilepsy_condition(state="Epileptic", drug="None")
            if drug == "None":
                fig = plot_module2B_grid(baseline, epileptic, sel_title="Epileptic (no drug)", sel_color=COL_EPILEPTIC)
            else:
                treated = simulate_epilepsy_condition(state="Epileptic", drug=drug)
                fig = plot_module2B_trio(baseline, epileptic, treated, drug_label=drug)

        st.pyplot(fig, clear_figure=True, use_container_width=True)