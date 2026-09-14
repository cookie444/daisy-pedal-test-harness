# Daisy Pedal HIL Test Rig

Hardware-in-the-loop audio test harness for Electrosmith Daisy pedals (Pod / Seed).

The rig plays a **known signal** into the pedal, captures what comes back, and
**asserts measurable specs** with pass/fail thresholds — delay vs tap tempo,
repeat decay, THD/THD+N, frequency response, stutter slice timing, noise floor,
latency. It emits **JUnit XML** for CI and an **HTML report** with the waveforms.

It runs the same suite against three backends:

| backend   | what it does                                            | where      |
|-----------|---------------------------------------------------------|------------|
| `sim`     | a reference DSP model of the pedal (a *plant*, not a mock) | GitHub Actions, no hardware |
| `audio`   | plays through a real Daisy Pod via your audio interface | self-hosted runner with a Pod attached |
| `wavfile` | analyses WAV captures recorded earlier (triage, bench rigs) | anywhere  |

The point is the harness, not the test cases: **every spec has a measurable
threshold** in `config/rig.yaml`, and the rig is proven to *fail* when the pedal
regresses (see [Fault injection](#fault-injection)).

---

## Layout

```
config/rig.yaml          sample rate, pedal settings, tolerances, faults
src/daisyhil/
  signals.py             stimulus generators (click, sine, sweep, noise, silence)
  measure.py             measurement algorithms (pure, unit-tested)
  suite.py               stimulus -> pedal -> measurement -> verdict
  results.py             SpecResult / RunReport / threshold evaluation
  reporting/             JUnit XML writer + HTML report (matplotlib figures)
  backends/
    simulator.py         the reference DSP plant with fault injection
    audio.py             sounddevice play/record loopback
    wavfile.py           offline captures
    control.py           tap / bypass / stutter transport (serial, scheduled)
tests/
  unit/test_measurements.py    algorithms verified against synthetic ground truth
  unit/test_fault_detection.py the rig must fail when the plant regresses
  unit/test_suite_contract.py  every spec has a threshold and vice versa
  hil/test_pedal_specs.py      the suite as pytest cases (JUnit via --junit-xml)
.github/workflows/hil.yml
```

## Quick start

```bash
pip install -e '.[dev]'          # CI path: no audio deps
python -m daisyhil.cli run --backend sim --out build
python -m daisyhil.cli run --backend sim --out build --fault delay_drift_pct=25   # proves a fail
```

or with pytest (produces `pytest --junit-xml=build/junit.xml` per spec):

```bash
pytest tests -q
pytest tests/hil -q --junit-xml=build/hil-junit.xml
```

Run the whole thing and open the report:

```bash
daisy-hil run --backend sim --out build
# build/report.html  <- waveforms + verdicts
# build/junit.xml    <- CI-consumable
```

## The specs

| group             | spec                          | stimulus                | measured how                                  | threshold window                     |
|-------------------|-------------------------------|-------------------------|-----------------------------------------------|--------------------------------------|
| Delay             | `delay_time_ms`               | click                    | median spacing of detected repeats           | knob value ± 2%                      |
| Delay             | `tap_delay_ms`                | click (wet only)         | repeat spacing after two footswitch taps     | tap interval ± 3%                    |
| Delay             | `processing_latency_ms`       | click                    | arrival through pedal minus bypassed arrival | ≤ 3 ms                               |
| Repeats           | `decay_db_per_repeat`         | click                    | level slope of the repeat train              | 20·log₁₀(feedback) ± 1.5 dB          |
| Repeats           | `decay_time_ms`               | click                    | time for repeats to fall 60 dB               | theoretical ± 25%                    |
| Distortion        | `thd_pct`, `thd_n_pct`, `snr_db` | 1 kHz sine @ -12 dBFS    | Blackman-Harris FFT, harmonics 2–10          | ≤ 1 % / ≤ 2 % / ≥ 60 dB              |
| FrequencyResponse | `gain_1khz_db`                | swept sine (Farina)      | deconvolved IR, reference-normalised         | ± 1 dB                               |
| FrequencyResponse | `fr_ripple_db`                | swept sine               | peak-to-peak 100 Hz–10 kHz                   | ≤ 3 dB                               |
| FrequencyResponse | `fr_corner_low_hz` / `fr_corner_high_hz` | swept sine | −3 dB crossings                     | ≤ 60 Hz / ≥ 12 kHz                   |
| Stutter           | `stutter_slice_ms`            | decorrelated noise       | autocorrelation period of the gated region  | request ± 5%                         |
| Stutter           | `stutter_gate_ms` / `stutter_repeats` | noise             | sample-accurate gate length / slice          | request ± 10% / ± 0.35               |
| Noise             | `noise_floor_dbfs`            | silence                  | RMS of the pedal output                      | ≤ −80 dBFS                           |

The delay/tap/stutter values are chosen to be an exact number of samples at
48 kHz so the rig measures the pedal, not a fractional-delay interpolator.

## Fault injection

The simulator is a *plant*, and every regression the rig is meant to catch can
be injected into it. The fault tests assert the corresponding spec **fails** — a
green harness is unverifiable, a harness that catches injected faults is real:

```bash
pytest tests/unit/test_fault_detection.py -v
```

```
fault delay_drift_pct=25.0    -> delay_time_ms FAILS
fault decay_error_pct=40.0    -> decay_db_per_repeat FAILS
fault thd_add_pct=100.0       -> thd_pct FAILS
fault noise_floor_dbfs=-60.0  -> noise_floor_dbfs FAILS
fault stutter_slice_error_pct=20.0 -> stutter_slice_ms FAILS
fault latency_extra_ms=5.0    -> processing_latency_ms FAILS
fault high_corner_hz=8000.0   -> fr_corner_high_hz FAILS
```

If one of those tests starts *passing*, someone loosened a threshold past the
point where it catches that bug.

## Hardware setup (audio backend)

Wire the Pod between an audio interface's output and input:

```
interface out ──► Pod input     Pod output ──► interface in
```

```bash
daisy-hil devices                       # find your interface's index
daisy-hil run --backend audio --out build-hw --save-captures
```

Point `config/rig.yaml` → `audio:` at your device (`input_device` /
`output_device` accept a name substring or index), and if the pedal exposes a
debug build over USB CDC (taps / bypass / stutter), add it under `plugins.serial`
— the harness uses it for tap tempo and bypass so those specs run end to end.
Stutter is triggered via a scheduled transport command.

Every spec tolerates the interface's own round-trip latency because it is
*calibrated* on the bypassed path first, then subtracted.

## Adding a spec

1. Add a threshold to `config/rig.yaml`.
2. Add one method `_spec_<name>` in `suite.py`.
3. Wire it into `run()`.
The contract test (`tests/unit/test_suite_contract.py`) refuses to merge a spec
with no threshold or a threshold nothing measures.

## GitHub Actions

`.github/workflows/hil.yml` runs, on every push/PR:

1. **unit** — measurement self-verification, suite contract, fault detection
   across Python 3.10–3.12;
2. **simulated** — the full spec suite against the reference DSP, artifacts
   uploaded and JUnit results published to the checks panel;
3. **hardware** — dispatch-only (`workflow_dispatch` with `backend: audio`),
   runs on a `[self-hosted, hil, daisy]` runner with a Pod attached, uploads the
   report **and the WAV captures** so a failing build can be triaged offline
   with the `wavfile` backend.