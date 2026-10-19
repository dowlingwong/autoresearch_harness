"""Declared, frozen action versions (PLAN §10, D4 amendment adopted 18 Oct 2026).

D4 allows the keep rule to change only through a *declared, frozen* action
version.  This module names those versions and holds the Gaussian arithmetic
they use (T2 §1–§3, AI-assisted and unchecked by the author):

* keep rule ``default/v1`` — keep iff Δ̂ > 0 (``decide_trial``; the historical rule);
* keep rule ``bayes_threshold/v1`` — run-free: keep iff Δ̂ > t*, t* = −μσ²/τ²,
  with (μ, τ, σ) supplied per decision by the control plane (a constant
  :class:`KeepPrior` or a per-proposal provider) and recorded in the decision
  event;
* verify response ``mean_delta/v1`` — keep iff D̄ > 0 (the frozen response of
  ``_run_verify``; kept for continuity);
* verify response ``posterior_mean/v1`` — keep iff E[Δ | Δ̂, D̄] > 0 under the
  Gaussian model of T2 §1 with the campaign's verify split policy;
* verify-failed policy ``veto/v1`` — historical: when every verify pair fails
  the response is ``reject`` (which vetoes a keep);
* verify-failed policy ``original_stands/v1`` — the original decision stands and
  ``verify_failed`` is recorded (PLAN default for new campaigns).

Backward compatibility (the D24 pattern): a manifest or caller that does not
declare a field means the historical value.  ``ActionVersions.legacy()`` is
exactly the pre-amendment behaviour (``default/v1``, ``mean_delta/v1``,
``veto/v1``, split ``"fixed"``); ``ActionVersions.new_campaign()`` holds the
PLAN defaults for new campaigns (split ``"vary"``, ``original_stands/v1``; the
keep rule stays ``default/v1`` — the Bayes rule is off unless declared).
``run_real_campaign`` without ``action_versions`` behaves exactly as before.

Everything here is ENGINEERING_ONLY.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Union

KEEP_RULE_DEFAULT = "default/v1"
KEEP_RULE_BAYES = "bayes_threshold/v1"
KEEP_RULES = (KEEP_RULE_DEFAULT, KEEP_RULE_BAYES)

VERIFY_RESPONSE_MEAN = "mean_delta/v1"
VERIFY_RESPONSE_POSTERIOR = "posterior_mean/v1"
VERIFY_RESPONSES = (VERIFY_RESPONSE_MEAN, VERIFY_RESPONSE_POSTERIOR)

VERIFY_FAILED_VETO = "veto/v1"
VERIFY_FAILED_STANDS = "original_stands/v1"
VERIFY_FAILED_POLICIES = (VERIFY_FAILED_VETO, VERIFY_FAILED_STANDS)

SPLIT_POLICIES = ("fixed", "vary")
# PLAN §10 (18 Oct): "vary" on every node for new campaigns; manifests and
# callers that do not declare a policy keep the historical "fixed".
NEW_CAMPAIGN_VERIFY_SPLIT_POLICY = "vary"
LEGACY_VERIFY_SPLIT_POLICY = "fixed"


def resolve_verify_split_policy(manifest: Mapping[str, Any] | None) -> str:
    """The verify split policy a manifest declares; a missing field means ``"fixed"``."""
    if manifest:
        declared = (manifest.get("action_versions") or {}).get("verify_split_policy")
        if declared is None:
            declared = manifest.get("verify_split_policy")
        if declared is not None:
            policy = str(declared).split(" ")[0]
            if policy not in SPLIT_POLICIES:
                raise ValueError(f"unknown verify_split_policy: {declared!r}")
            return policy
    return LEGACY_VERIFY_SPLIT_POLICY


@dataclass(frozen=True)
class ActionVersions:
    """The declared action versions of one campaign (recorded in its manifest)."""

    keep_rule: str = KEEP_RULE_DEFAULT
    verify_response: str = VERIFY_RESPONSE_MEAN
    verify_failed: str = VERIFY_FAILED_STANDS
    verify_split_policy: str = NEW_CAMPAIGN_VERIFY_SPLIT_POLICY

    def __post_init__(self) -> None:
        if self.keep_rule not in KEEP_RULES:
            raise ValueError(f"unknown keep rule version: {self.keep_rule!r}")
        if self.verify_response not in VERIFY_RESPONSES:
            raise ValueError(f"unknown verify response version: {self.verify_response!r}")
        if self.verify_failed not in VERIFY_FAILED_POLICIES:
            raise ValueError(f"unknown verify-failed policy: {self.verify_failed!r}")
        if self.verify_split_policy not in SPLIT_POLICIES:
            raise ValueError(f"unknown verify_split_policy: {self.verify_split_policy!r}")

    @classmethod
    def legacy(cls) -> "ActionVersions":
        """Exactly the pre-amendment behaviour (historical manifests)."""
        return cls(
            keep_rule=KEEP_RULE_DEFAULT,
            verify_response=VERIFY_RESPONSE_MEAN,
            verify_failed=VERIFY_FAILED_VETO,
            verify_split_policy=LEGACY_VERIFY_SPLIT_POLICY,
        )

    @classmethod
    def new_campaign(cls, **overrides: str) -> "ActionVersions":
        """PLAN defaults for a new campaign, with explicit overrides."""
        return cls(**overrides)

    @property
    def is_legacy(self) -> bool:
        return self == ActionVersions.legacy()

    @property
    def needs_prior(self) -> bool:
        return self.keep_rule == KEEP_RULE_BAYES or self.verify_response == VERIFY_RESPONSE_POSTERIOR

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ActionVersions":
        """Read a declaration; every missing field means its historical value."""
        legacy = cls.legacy()
        payload = payload or {}
        return cls(
            keep_rule=str(payload.get("keep_rule", legacy.keep_rule)),
            verify_response=str(payload.get("verify_response", legacy.verify_response)),
            verify_failed=str(payload.get("verify_failed", legacy.verify_failed)),
            verify_split_policy=str(payload.get("verify_split_policy", legacy.verify_split_policy)),
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any] | None) -> "ActionVersions":
        """Versions declared by a campaign manifest.

        A manifest without ``action_versions`` is historical: legacy versions,
        with its top-level ``verify_split_policy`` honoured (default ``"fixed"``).
        """
        manifest = manifest or {}
        declared = dict(manifest.get("action_versions") or {})
        declared.setdefault("verify_split_policy", resolve_verify_split_policy(manifest))
        return cls.from_dict(declared)


@dataclass(frozen=True)
class KeepPrior:
    """Gaussian prior and noise of one decision, in metric units (direction-adjusted).

    ``mu`` and ``tau`` are the mean and sd of the true increment Δ of the
    proposal; ``sigma`` is σ_d, the sd of one paired campaign comparison Δ̂.
    ``sigma_t``/``sigma_s`` split σ_d into training-seed and split-seed parts;
    they are needed only by the posterior-mean verify response under the
    ``"fixed"`` split policy (give one and the other is derived).
    """

    mu: float
    tau: float
    sigma: float
    sigma_t: float | None = None
    sigma_s: float | None = None
    source: str = "constant"

    def __post_init__(self) -> None:
        for name in ("mu", "tau", "sigma"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"KeepPrior.{name} must be a finite number")
        if self.tau <= 0:
            raise ValueError("KeepPrior.tau must be > 0")
        if self.sigma < 0:
            raise ValueError("KeepPrior.sigma must be >= 0")
        for name in ("sigma_t", "sigma_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or value < 0 or value > self.sigma * (1 + 1e-9)):
                raise ValueError(f"KeepPrior.{name} must lie in [0, sigma]")
        if self.sigma_t is not None and self.sigma_s is not None:
            if not math.isclose(math.hypot(self.sigma_t, self.sigma_s), self.sigma, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError("KeepPrior: hypot(sigma_t, sigma_s) must equal sigma")

    def split_components(self) -> tuple[float, float] | None:
        """(σ_t, σ_s), or ``None`` when the split is not given."""
        if self.sigma_t is not None:
            st = float(self.sigma_t)
            return st, math.sqrt(max(self.sigma ** 2 - st ** 2, 0.0))
        if self.sigma_s is not None:
            ss = float(self.sigma_s)
            return math.sqrt(max(self.sigma ** 2 - ss ** 2, 0.0)), ss
        return None

    @property
    def threshold(self) -> float:
        """t* = −μσ²/τ² (T2 Proposition 1)."""
        return -self.mu * self.sigma ** 2 / self.tau ** 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A per-decision prior source: a constant prior, or a callable receiving the
# leak-safe E2 ``ControlView`` of the decision (per-proposal priors).
KeepPriorProvider = Union[KeepPrior, Callable[[Any], KeepPrior]]


def resolve_keep_prior(provider: KeepPriorProvider | None, view: Any) -> KeepPrior:
    if provider is None:
        raise ValueError("this action version needs a keep prior (mu, tau, sigma); none was supplied")
    prior = provider if isinstance(provider, KeepPrior) else provider(view)
    if not isinstance(prior, KeepPrior):
        raise TypeError(f"keep prior provider returned {type(prior).__name__}, expected KeepPrior")
    return prior


def validate_campaign_versions(
    versions: ActionVersions | None, provider: KeepPriorProvider | None
) -> None:
    """Fail at campaign start, never mid-trial, when a declared version cannot run."""
    if versions is None:
        if provider is not None:
            raise ValueError("a keep prior was supplied without action_versions declaring a rule that uses it")
        return
    if versions.needs_prior and provider is None:
        raise ValueError(
            f"action versions {versions.to_dict()} need a keep prior (mu, tau, sigma); none was supplied"
        )
    if (
        isinstance(provider, KeepPrior)
        and versions.verify_response == VERIFY_RESPONSE_POSTERIOR
        and versions.verify_split_policy == "fixed"
        and provider.split_components() is None
    ):
        raise ValueError("posterior-mean verify under the 'fixed' split policy needs sigma_t or sigma_s")


def bayes_keep(delta_hat: float, prior: KeepPrior) -> bool:
    """Keep iff Δ̂ > t* (equivalently E[Δ | Δ̂] > 0)."""
    return float(delta_hat) > prior.threshold


def _gauss_condition(mu: float, cov_x_y: list[float], cov_y: list[list[float]], y: list[float]) -> float:
    """E[X | Y = y] for jointly Gaussian (X, Y) with E[Y_i] = mu (scalar X, 1 or 2 observations)."""
    if len(y) == 1:
        if cov_y[0][0] <= 0:
            return y[0]
        return mu + cov_x_y[0] / cov_y[0][0] * (y[0] - mu)
    a, b, d = cov_y[0][0], cov_y[0][1], cov_y[1][1]
    det = a * d - b * b
    if det <= 1e-12 * max(a * d, 1e-300):
        # (Δ̂, D̄) degenerate (e.g. "fixed" with σ_t = 0: D̄ carries no information
        # beyond Δ̂).  Condition on the first observation only.
        return _gauss_condition(mu, cov_x_y[:1], [[a]], y[:1])
    inv = [[d / det, -b / det], [-b / det, a / det]]
    r0, r1 = y[0] - mu, y[1] - mu
    w0 = cov_x_y[0] * inv[0][0] + cov_x_y[1] * inv[1][0]
    w1 = cov_x_y[0] * inv[0][1] + cov_x_y[1] * inv[1][1]
    return mu + w0 * r0 + w1 * r1


def verify_posterior_mean(
    delta_hat: float | None,
    dbar: float,
    r: int,
    prior: KeepPrior,
    split_policy: str,
) -> float:
    """E[Δ | Δ̂, D̄] under T2 §1 (direction-adjusted metric units).

    ``"vary"``: D_i = Δ + ξ_i, ξ_i ~ N(0, σ_d²), independent of Δ̂.
    ``"fixed"``: D_i = Δ + e_s + ξ_i, ξ_i ~ N(0, σ_t²), sharing the campaign's
    split error e_s with Δ̂.  *r* is the number of valid pairs; *delta_hat*
    ``None`` (no incumbent comparison) conditions on D̄ alone.
    """
    if r < 1:
        raise ValueError("posterior mean needs at least one valid verify pair")
    tau2 = prior.tau ** 2
    if split_policy == "vary":
        var_hat = tau2 + prior.sigma ** 2
        var_dbar = tau2 + prior.sigma ** 2 / r
        cov_hat_dbar = tau2
    elif split_policy == "fixed":
        parts = prior.split_components()
        if parts is None:
            raise ValueError("posterior-mean verify under the 'fixed' split policy needs sigma_t or sigma_s")
        st2, ss2 = parts[0] ** 2, parts[1] ** 2
        var_hat = tau2 + st2 + ss2
        var_dbar = tau2 + ss2 + st2 / r
        cov_hat_dbar = tau2 + ss2
    else:
        raise ValueError(f"unknown split policy {split_policy!r}")
    if delta_hat is None:
        return _gauss_condition(prior.mu, [tau2], [[var_dbar]], [float(dbar)])
    return _gauss_condition(
        prior.mu,
        [tau2, tau2],
        [[var_hat, cov_hat_dbar], [cov_hat_dbar, var_dbar]],
        [float(delta_hat), float(dbar)],
    )


def declare_new_campaign(manifest: dict[str, Any], **overrides: str) -> ActionVersions:
    """Record the PLAN defaults (verify split ``"vary"``, 18 Oct 2026) in a new manifest.

    Writes both the top-level ``verify_split_policy`` (read by existing
    collectors) and the full ``action_versions`` declaration, and returns the
    versions to pass to ``run_real_campaign``.  Refuses a manifest that already
    declares something else, so a historical manifest is never rewritten.
    """
    versions = ActionVersions.new_campaign(**overrides)
    for key, value in (("verify_split_policy", versions.verify_split_policy),
                       ("action_versions", versions.to_dict())):
        if key in manifest and manifest[key] != value:
            raise ValueError(f"manifest already declares {key}={manifest[key]!r}")
        manifest[key] = value
    return versions
