import unittest
import numpy as np
import torch

from baselines.priorrl_cc4.prototype_retrieval import (
    FrozenPrototypeRetriever, PrototypeInput, PrototypeLookupMiss, fit_train_prototypes,
    freeze_validation_radius, validate_payload,
)
from shared.formal_state import BLUE_AGENTS
from baselines.priorrl_cc4.policy import PriorRLActorCritic, forward_kl


def rec(agent, value, prior=None, suffix="0"):
    state = np.full(27, value, dtype=np.float32)
    plans = np.asarray([
        [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2],
        [3, 3, 3, 3], [0, 1, 2, 3], [1, 2, 3, 0],
    ], dtype=np.int64)
    preferences = np.asarray([.3, .2, .15, .1, .15, .1], dtype=np.float32)
    return PrototypeInput(state, agent, np.asarray(prior or [.7, .1, .1, .1], dtype=np.float32),
                          suffix * 64, ("f" if suffix == "0" else "e") * 64,
                          plans, preferences.copy(), preferences, ("llm",) * 6)


def frozen():
    rows = []
    for i, agent in enumerate(BLUE_AGENTS):
        rows.extend([rec(agent, i, suffix="0"), rec(agent, i + .1, prior=[.1,.7,.1,.1], suffix="1")])
    train = fit_train_prototypes(rows, source_sha256="a" * 64)
    validation = [(agent, np.full(27, i + .05, np.float32)) for i, agent in enumerate(BLUE_AGENTS)]
    return freeze_validation_radius(train, validation, validation_source_sha256="b" * 64, quantile=1.0)


class TestPriorRLPrototypeRetrieval(unittest.TestCase):
    def test_train_only_agent_local_and_validation_radius(self):
        payload = frozen(); validate_payload(payload)
        self.assertEqual(payload["fit_split"], "train")
        self.assertEqual(payload["radius_selection_split"], "validation")
        self.assertEqual(set(payload["agents"]), set(BLUE_AGENTS))

    def test_agent_local_lookup_and_radius_fail_closed(self):
        retriever = FrozenPrototypeRetriever(frozen())
        prior = retriever.lookup(np.zeros(27, np.float32), agent_name="blue_agent_0")
        self.assertEqual(int(prior.argmax()), 0)
        with self.assertRaisesRegex(PrototypeLookupMiss, "exceeds frozen radius") as caught:
            retriever.lookup(np.full(27, 100, np.float32), agent_name="blue_agent_0")
        self.assertEqual(caught.exception.agent_name, "blue_agent_0")
        self.assertEqual(caught.exception.state.shape, (27,))
        self.assertEqual(len(caught.exception.state_sha256), 64)
        with self.assertRaisesRegex(RuntimeError, "unknown agent"):
            retriever.lookup(np.zeros(27), agent_name="bad")

    def test_tie_break_is_state_sha_then_cache_key(self):
        payload = frozen(); item = payload["agents"]["blue_agent_0"]
        # midpoint is equidistant; sorted state hash/cache key controls winner reproducibly
        query = np.full(27, .05, np.float32)
        expected = sorted(item["prototypes"], key=lambda x: (x["state_sha256"], x["cache_key"]))[0]
        got = FrozenPrototypeRetriever(payload).lookup(query, agent_name="blue_agent_0")
        np.testing.assert_allclose(got.numpy(), expected["action_prior"])

    def test_checksum_and_unfrozen_radius_rejected(self):
        payload = frozen(); payload["weights"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            FrozenPrototypeRetriever(payload)
        train = fit_train_prototypes([rec(a, i) for i, a in enumerate(BLUE_AGENTS)], source_sha256="a"*64)
        with self.assertRaisesRegex(ValueError, "validation only"):
            FrozenPrototypeRetriever(train)

    def test_retrieved_a4_prior_integrates_with_forward_kl_policy(self):
        retriever = FrozenPrototypeRetriever(frozen())
        state = np.zeros(27, np.float32)
        prior = retriever.lookup(state, agent_name="blue_agent_0")
        logits, _ = PriorRLActorCritic()(state)
        loss = forward_kl(logits, prior)
        self.assertEqual(tuple(prior.shape), (4,))
        self.assertTrue(torch.isfinite(loss))

    def test_retrieved_k6_h4_prior_is_exact_and_read_only(self):
        payload = frozen(); retriever = FrozenPrototypeRetriever(payload)
        batch = retriever.lookup_prior_batch(np.zeros(27, np.float32), agent_name="blue_agent_0")
        winner = retriever._winner(np.zeros(27, np.float32), agent_name="blue_agent_0")
        np.testing.assert_array_equal(batch.plans, np.asarray(winner["plans"]))
        np.testing.assert_allclose(batch.prior_preferences, winner["prior_preferences"])
        batch.plans[0, 0] = 3
        self.assertNotEqual(batch.plans[0, 0], retriever.lookup_prior_batch(
            np.zeros(27, np.float32), agent_name="blue_agent_0").plans[0, 0])


if __name__ == "__main__": unittest.main()
