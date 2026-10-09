"""Expected free energy of every action sequence, rolled out as a tree in batch.

For a policy pi over T steps, with predicted state beliefs q(s_t | pi) and joint outcomes
q(o_t | pi) = sum_s q(s_t | pi) prod_m A_m(o_m | s):

    risk      = sum_t KL[ q(o_t | pi) || prod_m softmax(C_m) ]
    ambiguity = sum_t E_q(s_t | pi) H[ A(. | s_t) ]            (A's modalities are independent given s)
    G         = -(risk + ambiguity)

This is exactly pymdp's ``update_posterior_policies`` G (expected utility plus state information gain, the
latter taken over the joint outcome), computed for all policies at once. pymdp 0.0.7 loops over policies in
Python and takes ~7 ms per policy and step, too slow for 3^6 policies at planner rates;
``tests/test_aif.py`` checks the two agree.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
from numpy.typing import NDArray

from av_core.plan.aif.model import GenerativeModel

_EPS = 1e-16


def preferences(model: GenerativeModel) -> list[NDArray[np.float64]]:
    """softmax(C_m): the preferred distribution over each modality's outcomes."""
    out: list[NDArray[np.float64]] = []
    for c in model.C:
        e = np.exp(np.asarray(c, dtype=np.float64) - np.max(c))
        out.append(e / e.sum())
    return out


def joint_likelihood(A: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """(prod_m O_m, NS): P(o_1, ..., o_M | s), the outer product of the modalities' columns."""
    ns = A[0].shape[1]
    return reduce(lambda acc, a: (acc[:, None, :] * a[None, :, :]).reshape(-1, ns), A[1:], A[0])


def expected_free_energy(qs: NDArray[np.float64], model: GenerativeModel, policy_len: int
                         ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """(policies (P, T) of action indices, risk (P,), ambiguity (P,)) for all ``n_actions ** T`` policies.

    ``qs`` is the current belief over the world factor, shape (NS,).
    """
    B = np.asarray(model.B[0], dtype=np.float64)  # (NS, NS, n_actions)
    n_actions = B.shape[2]
    A = [np.asarray(a, dtype=np.float64) for a in model.A]  # (O_m, NS)
    A_joint = joint_likelihood(A)
    # log of the joint preference: sum of the modalities' log preferences, in the same outcome order
    log_pref = reduce(lambda acc, p: (acc[:, None] + np.log(p + _EPS)[None, :]).reshape(-1),
                      preferences(model)[1:], np.log(preferences(model)[0] + _EPS))
    ent = sum(-(a * np.log(a + _EPS)).sum(axis=0) for a in A)  # (NS,): E H[A(.|s)] adds over modalities
    B_by_action = [B[:, :, a].T for a in range(n_actions)]  # (NS_from, NS_to) for row-vector beliefs

    beliefs = np.asarray(qs, dtype=np.float64)[None, :]  # (n_policies_so_far, NS)
    policies = np.zeros((1, 0), dtype=np.int64)
    risk = np.zeros(1)
    ambiguity = np.zeros(1)
    for _ in range(policy_len):
        n = beliefs.shape[0]
        nxt = np.stack([beliefs @ Ba for Ba in B_by_action], axis=1).reshape(n * n_actions, -1)  # parent-major
        policies = np.concatenate(
            [np.repeat(policies, n_actions, axis=0), np.tile(np.arange(n_actions), n)[:, None]], axis=1)
        qo = nxt @ A_joint.T  # (n * n_actions, prod O_m)
        risk = np.repeat(risk, n_actions) + np.sum(qo * (np.log(qo + _EPS) - log_pref), axis=1)
        ambiguity = np.repeat(ambiguity, n_actions) + nxt @ ent
        beliefs = nxt
    return policies, risk, ambiguity
