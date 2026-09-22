import tempfile
import unittest

from pathlib import Path

import numpy as np
import torch

from formal_experiments.data_collection.decision_replay import (
    DecisionEpochTransition,
    EXECUTED_DURATION,
)

from formal_experiments.training.bootstrap_world_model import (
    BootstrapProbabilisticWorldModel,
    BootstrapWorldModelConfig,
    StateNormalizer,
    WorldModelDataset,
    build_world_model_dataset,
)

from shared.action_contract import (
    get_action,
)

from shared.formal_state import (
    FORMAL_STATE_DIM,
)


def formal_state(
    value: float,
):
    return np.full(
        (FORMAL_STATE_DIM,),
        value,
        dtype=np.float32,
    )


def transition(
    *,
    requested_action_id: int,
    executed_family: str,
    start_value: float = 0.0,
    next_value: float = 0.1,
    action_completed: bool = True,
):
    requested = get_action(
        requested_action_id
    )

    duration = (
        EXECUTED_DURATION[
            executed_family
        ]
    )

    if action_completed:
        decision_dt = duration
    else:
        decision_dt = max(
            1,
            duration - 1,
        )

    target = (
        None
        if executed_family == "Sleep"
        else "host_a"
    )

    return DecisionEpochTransition(
        episode_seed=42,
        agent_name="blue_agent_0",
        decision_index=0,

        global_tick_start=0,
        global_tick_end=decision_dt,
        decision_dt=decision_dt,

        state=formal_state(
            start_value
        ),

        next_state=formal_state(
            next_value
        ),

        requested_action_id=(
            requested.action_id
        ),

        requested_action_name=(
            requested.name
        ),

        requested_cyborg_family=(
            requested.cyborg_action
        ),

        requested_duration_ticks=(
            requested.duration_ticks
        ),

        executed_index=0,
        executed_label=(
            executed_family
            if target is None
            else (
                f"{executed_family} "
                f"{target}"
            )
        ),

        executed_action_family=(
            executed_family
        ),

        executed_duration_ticks=(
            duration
        ),

        target_host=target,

        fallback=(
            requested.cyborg_action
            != executed_family
        ),

        fallback_reason=(
            "no_valid_observable_target"
            if (
                requested.cyborg_action
                != executed_family
            )
            else None
        ),

        action_completed=(
            action_completed
        ),

        completed_action_success=(
            True
            if action_completed
            else None
        ),

        incident_event_ids=tuple(),
        incident_host_ids=tuple(),

        incident_active_ticks=0,
        incident_host_lwf_count=0,
        incident_host_lwf_raw_penalty=0.0,
        response_reward=0.0,

        official_reward=0.0,

        done=(
            not action_completed
        ),
    )


def synthetic_dataset(
    n: int = 48,
):
    rng = np.random.default_rng(
        12345
    )

    states = rng.uniform(
        low=0.0,
        high=1.0,
        size=(
            n,
            FORMAL_STATE_DIM,
        ),
    ).astype(
        np.float32
    )

    actions = rng.integers(
        low=0,
        high=4,
        size=n,
        dtype=np.int64,
    )

    next_states = (
        states.copy()
    )

    next_states[:, 5] += (
        actions.astype(
            np.float32
        )
        * 0.02
    )

    next_states[:, 15] += 0.01

    return WorldModelDataset(
        states=states,
        actions=actions,
        next_states=next_states,
    )


def tiny_config(
    *,
    target_mode="absolute",
):
    return BootstrapWorldModelConfig(
        ensemble_size=3,
        hidden_dim=16,
        lr=1e-3,
        batch_size=16,
        epochs=2,
        target_mode=target_mode,
        model_seed=100,
        bootstrap_seed=200,
    )


class TestGateABootstrapWorldModel(
    unittest.TestCase
):

    def test_formal_defaults(
        self,
    ):
        config = (
            BootstrapWorldModelConfig()
        )

        self.assertEqual(
            config.ensemble_size,
            5,
        )

        self.assertEqual(
            config.hidden_dim,
            128,
        )

        self.assertEqual(
            config.lr,
            3e-4,
        )

        self.assertEqual(
            config.target_mode,
            "absolute",
        )

    def test_dataset_uses_executed_action(
        self,
    ):
        # requested Restore
        # executed fallback Sleep
        item = transition(
            requested_action_id=3,
            executed_family="Sleep",
        )

        dataset = (
            build_world_model_dataset(
                [item]
            )
        )

        self.assertEqual(
            dataset.actions.tolist(),
            [0],
        )

    def test_incomplete_terminal_is_dropped(
        self,
    ):
        complete = transition(
            requested_action_id=0,
            executed_family="Sleep",
        )

        incomplete = transition(
            requested_action_id=3,
            executed_family="Restore",
            action_completed=False,
        )

        dataset = (
            build_world_model_dataset(
                [
                    complete,
                    incomplete,
                ]
            )
        )

        self.assertEqual(
            dataset.n_samples,
            1,
        )

    def test_incomplete_can_be_rejected(
        self,
    ):
        incomplete = transition(
            requested_action_id=3,
            executed_family="Restore",
            action_completed=False,
        )

        with self.assertRaises(
            ValueError
        ):
            build_world_model_dataset(
                [incomplete],
                drop_incomplete=False,
            )

    def test_normalizer_round_trip(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        normalizer = (
            StateNormalizer.fit(
                dataset.states,
                dataset.next_states,
            )
        )

        z = normalizer.normalize_np(
            dataset.states
        )

        recovered = (
            normalizer
            .denormalize_np(
                z
            )
        )

        np.testing.assert_allclose(
            recovered,
            dataset.states,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_bootstrap_members_are_independent(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        summary = model.fit(
            dataset
        )

        indices = (
            model.bootstrap_indices
        )

        self.assertEqual(
            len(indices),
            3,
        )

        self.assertFalse(
            np.array_equal(
                indices[0],
                indices[1],
            )
        )

        self.assertFalse(
            np.array_equal(
                indices[1],
                indices[2],
            )
        )

        self.assertEqual(
            summary.n_samples,
            dataset.n_samples,
        )

    def test_bootstrap_has_replacement(
        self,
    ):
        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            synthetic_dataset()
        )

        for indices in (
            model.bootstrap_indices
        ):
            self.assertLess(
                np.unique(
                    indices
                ).size,
                indices.size,
            )

    def test_training_losses_are_finite(
        self,
    ):
        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        summary = model.fit(
            synthetic_dataset()
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    np.asarray(
                        summary
                        .member_final_losses
                    )
                )
            )
        )

    def test_predict_ensemble_shapes(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        result = model.predict_ensemble(
            dataset.states[:7],
            dataset.actions[:7],
        )

        self.assertEqual(
            result[
                "member_means"
            ].shape,
            (
                3,
                7,
                FORMAL_STATE_DIM,
            ),
        )

        self.assertEqual(
            result["mean"].shape,
            (
                7,
                FORMAL_STATE_DIM,
            ),
        )

    def test_total_variance_decomposition(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        result = model.predict_ensemble(
            dataset.states[:4],
            dataset.actions[:4],
        )

        np.testing.assert_allclose(
            result["total_variance"],
            (
                result[
                    "epistemic_variance"
                ]
                +
                result[
                    "aleatoric_variance"
                ]
            ),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_epistemic_variance_nonnegative(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        result = model.predict_ensemble(
            dataset.states[:5],
            dataset.actions[:5],
        )

        self.assertTrue(
            np.all(
                result[
                    "epistemic_variance"
                ]
                >= 0.0
            )
        )

    def test_member_tensor_batch_prediction(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        states = torch.as_tensor(
            dataset.states[:8],
            dtype=torch.float32,
        )

        actions = torch.as_tensor(
            dataset.actions[:8],
            dtype=torch.long,
        )

        mean = (
            model
            .predict_member_mean_tensor(
                0,
                states,
                actions,
            )
        )

        self.assertEqual(
            tuple(mean.shape),
            (
                8,
                FORMAL_STATE_DIM,
            ),
        )

    def test_delta_mode_supported(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config(
                    target_mode="delta"
                )
            )
        )

        model.fit(
            dataset
        )

        result = model.predict_ensemble(
            dataset.states[:3],
            dataset.actions[:3],
        )

        self.assertEqual(
            result["mean"].shape,
            (
                3,
                FORMAL_STATE_DIM,
            ),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result["mean"]
                )
            )
        )

    def test_checkpoint_round_trip(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        before = model.predict_ensemble(
            dataset.states[:5],
            dataset.actions[:5],
        )["mean"]

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "wm.pt"
            )

            model.save_checkpoint(
                path
            )

            loaded = (
                BootstrapProbabilisticWorldModel
                .load_checkpoint(
                    path
                )
            )

            after = (
                loaded
                .predict_ensemble(
                    dataset.states[:5],
                    dataset.actions[:5],
                )["mean"]
            )

        np.testing.assert_allclose(
            before,
            after,
            rtol=0.0,
            atol=0.0,
        )

    def test_invalid_predict_action_rejected(
        self,
    ):
        dataset = (
            synthetic_dataset()
        )

        model = (
            BootstrapProbabilisticWorldModel(
                tiny_config()
            )
        )

        model.fit(
            dataset
        )

        bad_actions = (
            dataset.actions[:2]
            .copy()
        )

        bad_actions[0] = 99

        with self.assertRaises(
            ValueError
        ):
            model.predict_ensemble(
                dataset.states[:2],
                bad_actions,
            )

    def test_invalid_target_mode_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            BootstrapWorldModelConfig(
                target_mode="bad"
            )


if __name__ == "__main__":
    unittest.main()