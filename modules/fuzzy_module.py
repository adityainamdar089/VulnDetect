"""
modules/fuzzy_module.py – scikit-fuzzy risk assessment module.

Builds a Mamdani fuzzy inference system with triangular membership functions
(low, medium, high) across all 5 input features and defuzzifies via the
centroid method to produce a single risk score in [0, 1].

Public API
----------
compute_fuzzy_risk(features: dict) -> float
run_fuzzy_on_dataset(df: pd.DataFrame) -> pd.DataFrame
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

# ─── scikit-fuzzy import (optional) ───────────────────────────────────────────
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    _SKFUZZY_OK = True
    logger.info("scikit-fuzzy is available.")
except ImportError:
    _SKFUZZY_OK = False
    logger.warning(
        "scikit-fuzzy not found – weighted arithmetic fallback will be used."
    )

# ─── Fuzzy system singleton ────────────────────────────────────────────────────
_fuzzy_sim = None


def _build_fuzzy_system() -> "ctrl.ControlSystemSimulation":
    """
    Construct the full fuzzy inference system.

    Antecedents (all universe [0, 1]):
        input_validation, sensitive_data_exposure, access_control,
        resource_management, control_flow_complexity

    Consequent:
        risk  (universe [0, 1])

    Membership functions: triangular – low, medium, high.
    Rules: 10 semantic rules covering all major combinations.
    Defuzzification: centroid method (default in skfuzzy).
    """
    universe = np.arange(0.0, 1.01, 0.01)

    # Antecedents ──────────────────────────────────────────────────────────────
    iv  = ctrl.Antecedent(universe, "input_validation")
    sde = ctrl.Antecedent(universe, "sensitive_data")
    ac  = ctrl.Antecedent(universe, "access_control")
    rm  = ctrl.Antecedent(universe, "resource_management")
    cfc = ctrl.Antecedent(universe, "control_flow_complexity")

    # Consequent ───────────────────────────────────────────────────────────────
    risk = ctrl.Consequent(universe, "risk")

    # Membership functions ─────────────────────────────────────────────────────
    # "Positive" antecedents (high value = safer)
    for var in [iv, ac, rm]:
        var["low"]    = fuzz.trimf(var.universe, [0.00, 0.00, 0.50])
        var["medium"] = fuzz.trimf(var.universe, [0.00, 0.50, 1.00])
        var["high"]   = fuzz.trimf(var.universe, [0.50, 1.00, 1.00])

    # "Negative" antecedents (high value = riskier)
    for var in [sde, cfc]:
        var["low"]    = fuzz.trimf(var.universe, [0.00, 0.00, 0.50])
        var["medium"] = fuzz.trimf(var.universe, [0.00, 0.50, 1.00])
        var["high"]   = fuzz.trimf(var.universe, [0.50, 1.00, 1.00])

    # Risk output
    risk["low"]    = fuzz.trimf(risk.universe, [0.00, 0.00, 0.50])
    risk["medium"] = fuzz.trimf(risk.universe, [0.00, 0.50, 1.00])
    risk["high"]   = fuzz.trimf(risk.universe, [0.50, 1.00, 1.00])

    # Rules ────────────────────────────────────────────────────────────────────
    rules = [
        # 1. Low validation  + high exposure            → high risk
        ctrl.Rule(iv["low"] & sde["high"],                       risk["high"]),
        # 2. Low access ctrl + low resource management  → medium risk
        ctrl.Rule(ac["low"] & rm["low"],                         risk["medium"]),
        # 3. High complexity + low validation           → high risk
        ctrl.Rule(cfc["high"] & iv["low"],                       risk["high"]),
        # 4. All positive indicators high               → low risk
        ctrl.Rule(iv["high"] & ac["high"] & rm["high"]
                  & cfc["low"] & sde["low"],                     risk["low"]),
        # 5. High exposure  + low access control        → high risk
        ctrl.Rule(sde["high"] & ac["low"],                       risk["high"]),
        # 6. Medium validation + medium complexity      → medium risk
        ctrl.Rule(iv["medium"] & cfc["medium"],                  risk["medium"]),
        # 7. Low resource mgmt + high complexity        → high risk
        ctrl.Rule(rm["low"] & cfc["high"],                       risk["high"]),
        # 8. High access ctrl + high resource mgmt      → low risk
        ctrl.Rule(ac["high"] & rm["high"],                       risk["low"]),
        # 9. Medium exposure  + medium access control   → medium risk
        ctrl.Rule(sde["medium"] & ac["medium"],                  risk["medium"]),
        # 10. Low complexity + high validation + low exposure → low risk
        ctrl.Rule(cfc["low"] & iv["high"] & sde["low"],          risk["low"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def _get_sim():
    """Return the singleton fuzzy simulation, building it on first call."""
    global _fuzzy_sim
    if _fuzzy_sim is None and _SKFUZZY_OK:
        try:
            _fuzzy_sim = _build_fuzzy_system()
        except Exception as exc:
            logger.error("Failed to build fuzzy system: %s", exc)
    return _fuzzy_sim


def _fallback_risk(features: Dict[str, float]) -> float:
    """
    Simple weighted arithmetic fallback when scikit-fuzzy is unavailable.
    Weights reflect relative importance of each feature on overall risk.
    """
    iv  = features.get("input_validation_score",    0.5)
    sde = features.get("sensitive_data_exposure",   0.5)
    ac  = features.get("access_control_strength",   0.5)
    rm  = features.get("resource_management_score", 0.5)
    cfc = features.get("control_flow_complexity",   0.5)

    # "Positive" features reduce risk; "negative" features increase risk
    risk = (
        sde * 0.30
        + cfc * 0.20
        + (1.0 - iv)  * 0.20
        + (1.0 - ac)  * 0.20
        + (1.0 - rm)  * 0.10
    )
    return float(np.clip(risk, 0.0, 1.0))


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_fuzzy_risk(features: Dict[str, float]) -> float:
    """
    Compute a fuzzy risk score for one code snippet.

    Parameters
    ----------
    features : dict with keys:
        input_validation_score, sensitive_data_exposure,
        access_control_strength, resource_management_score,
        control_flow_complexity

    Returns
    -------
    float in [0, 1]  (higher = riskier)
    """
    sim = _get_sim()
    if sim is None:
        return _fallback_risk(features)

    # Clip to (0.01, 0.99) to avoid MF boundary issues in skfuzzy
    def _clip(key: str) -> float:
        return float(np.clip(features.get(key, 0.5), 0.01, 0.99))

    try:
        sim.input["input_validation"]       = _clip("input_validation_score")
        sim.input["sensitive_data"]         = _clip("sensitive_data_exposure")
        sim.input["access_control"]         = _clip("access_control_strength")
        sim.input["resource_management"]    = _clip("resource_management_score")
        sim.input["control_flow_complexity"] = _clip("control_flow_complexity")
        sim.compute()
        return float(np.clip(sim.output["risk"], 0.0, 1.0))
    except Exception as exc:
        logger.warning("Fuzzy compute failed (%s) – using fallback.", exc)
        return _fallback_risk(features)


def run_fuzzy_on_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fuzzy_risk_score for every row in *df*.

    Parameters
    ----------
    df : DataFrame containing the 5 feature columns.

    Returns
    -------
    Copy of *df* with an added ``fuzzy_risk_score`` column.
    """
    logger.info("Computing fuzzy risk scores for %d samples …", len(df))
    scores = []
    for _, row in df.iterrows():
        feat = {
            "input_validation_score":    row.get("input_validation_score",    0.5),
            "sensitive_data_exposure":   row.get("sensitive_data_exposure",   0.5),
            "access_control_strength":   row.get("access_control_strength",   0.5),
            "resource_management_score": row.get("resource_management_score", 0.5),
            "control_flow_complexity":   row.get("control_flow_complexity",   0.5),
        }
        scores.append(compute_fuzzy_risk(feat))

    out = df.copy()
    out["fuzzy_risk_score"] = scores
    logger.info("Fuzzy scoring complete. Mean score: %.4f", float(np.mean(scores)))
    return out
