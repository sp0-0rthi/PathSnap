"""HMM / Viterbi route reconstruction.

Hidden states: the candidate road arcs of each GPS observation (the
candidate-to-road-segment correspondence is preserved: every state is an
arc of the trip's network).

Emission probabilities: P(observation | state = arc) is the tuned
classifier's predicted probability that this candidate is the correct
segment for this observation. Probabilities are floored at emission_eps
so log-probabilities stay finite.

Transition probabilities: from the road-network connectivity. The
shortest network path distance d_net between the two candidate arcs is
compared with the great-circle distance d_gps between the two GPS
observations (Newson & Krumm style):

    log P(a -> b) = -|d_net(a, b) - d_gps| / beta

so transitions whose network distance agrees with the observed movement
are favoured. Unreachable pairs (disconnected segments) receive
transition_eps instead of zero: the chain never dies and the decoder can
still cross genuinely disconnected observations, just at heavy cost.

Edge cases:
* missing candidates - observations with an empty candidate set are
  excluded from the chain; after decoding they inherit the decoded arc of
  the nearest in-chain observation (and are flagged in the output).
* invalid transitions - floored at transition_eps (see above).
* GPS gaps - when the time delta exceeds gap_seconds or the position jump
  exceeds gap_dist_m, the chain is broken: the next observation starts a
  fresh Viterbi chain, because sequential information is not meaningful
  across a long dropout.
"""

from __future__ import annotations

import numpy as np

from .config import CFG
from .geo import haversine_m
from .road_network import RoadNetwork


def transition_logprob(a: int, b: int, gps_dist_m: float, network: RoadNetwork,
                       beta: float = CFG.transition_beta,
                       eps: float = CFG.transition_eps) -> float:
    cutoff = gps_dist_m * CFG.path_search_factor + CFG.path_search_buffer_m
    d_net = network.path_distance(a, b, cutoff)
    if d_net is None:
        return float(np.log(eps))
    return -abs(d_net - gps_dist_m) / beta


def viterbi_decode(emissions, cand_lists, track, network: RoadNetwork,
                   cfg: CFG = CFG):
    """Decode the most likely arc sequence for one trip.

    Parameters
    ----------
    emissions  : list per observation of arrays with P(correct) per
                 candidate, aligned with cand_lists.
    cand_lists : list per observation of candidate arc indices.
    track      : cleaned track DataFrame (lat, lon, timestamp).
    network    : the trip's RoadNetwork.

    Returns
    -------
    decoded    : int array, one arc index per observation.
    """
    n = len(cand_lists)
    lat = track["lat"].to_numpy(dtype=float)
    lon = track["lon"].to_numpy(dtype=float)
    ts = track["timestamp"].to_numpy(dtype=float)

    # Memoise network path distances per arc pair: consecutive observations
    # share most candidates, so this eliminates nearly all repeated Dijkstra
    # searches.
    dist_cache: dict[tuple[int, int], float | None] = {}

    def t_logp(a, b, d_gps):
        key = (a, b)
        if key not in dist_cache:
            cutoff = d_gps * cfg.path_search_factor + cfg.path_search_buffer_m
            dist_cache[key] = network.path_distance(a, b, cutoff)
        d_net = dist_cache[key]
        if d_net is None:
            return float(np.log(cfg.transition_eps))
        return -abs(d_net - d_gps) / cfg.transition_beta

    usable = [i for i in range(n) if len(cand_lists[i]) > 0]
    if not usable:
        return np.full(n, -1, dtype=int)

    deltas = []        # log-prob table per chain position
    backpointers = []  # predecessor state index per chain position
    prev_obs = None

    for i in usable:
        states = cand_lists[i]
        emit = np.log(np.clip(np.asarray(emissions[i], dtype=float),
                              cfg.emission_eps, 1.0))
        if prev_obs is None:
            deltas.append(emit.copy())
            backpointers.append(np.full(len(states), -1, dtype=int))
        else:
            dt = ts[i] - ts[prev_obs]
            d_gps = haversine_m(lat[prev_obs], lon[prev_obs], lat[i], lon[i])
            broken = (dt > cfg.gap_seconds) or (d_gps > cfg.gap_dist_m)
            if broken:
                # GPS gap edge case: restart the chain
                deltas.append(emit.copy())
                backpointers.append(np.full(len(states), -1, dtype=int))
            else:
                prev_states = cand_lists[prev_obs]
                trans = np.empty((len(prev_states), len(states)))
                stay = cfg.stay_bonus
                for pa, a in enumerate(prev_states):
                    for cb, b in enumerate(states):
                        # persistence blend: reward remaining on the same arc
                        trans[pa, cb] = t_logp(a, b, d_gps) + (stay if a == b else 0.0)
                scores = deltas[-1][:, None] + trans
                best_prev = np.argmax(scores, axis=0)
                deltas.append(emit + scores[best_prev, np.arange(len(states))])
                backpointers.append(best_prev.astype(int))
        prev_obs = i

    # backtrace through chain positions
    n_pos = len(usable)
    choice = [0] * n_pos
    choice[-1] = int(np.argmax(deltas[-1]))
    for pos in range(n_pos - 1, 0, -1):
        bp = backpointers[pos]
        prev_idx = bp[choice[pos]]
        if prev_idx < 0:
            # chain restart marker: the previous position stands alone
            prev_idx = int(np.argmax(deltas[pos - 1]))
        choice[pos - 1] = int(prev_idx)

    decoded = np.full(n, -1, dtype=int)
    for pos, obs_i in enumerate(usable):
        states = cand_lists[obs_i]
        decoded[obs_i] = states[min(choice[pos], len(states) - 1)]

    # missing-candidate edge case: inherit the nearest decoded neighbour
    for i in range(n):
        if decoded[i] >= 0:
            continue
        left = next((j for j in range(i - 1, -1, -1) if decoded[j] >= 0), None)
        right = next((j for j in range(i + 1, n) if decoded[j] >= 0), None)
        if left is None and right is None:
            continue
        if left is None:
            decoded[i] = decoded[right]
        elif right is None:
            decoded[i] = decoded[left]
        else:
            decoded[i] = decoded[left] if (i - left) <= (right - i) else decoded[right]
    return decoded


def trip_emissions(model, feature_df, cand_lists, feature_columns):
    """Per-observation emission probability arrays from the classifier.

    feature_df rows are produced by build_feature_rows in candidate order
    per observation, so probabilities group back into the same order.
    """
    out = [np.array([], dtype=float) for _ in cand_lists]
    if feature_df.empty:
        return out
    probs = model.predict_proba(feature_df[feature_columns])[:, 1]
    obs = feature_df["obs_idx"].to_numpy()
    for i, p in zip(obs, probs):
        out[i] = np.append(out[i], p)
    return out
