# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrender", reason="pyrender optional dependency not installed.")
pytest.importorskip("trimesh", reason="trimesh optional dependency not installed.")

import unittest

import quaternion as qt
import trimesh

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.environments.mesh_environment import MeshEnvironment
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("agent_id_0")
SENSOR_ID_PATCH = SensorID("patch")
SENSOR_ID_VIEW_FINDER = SensorID("view_finder")
RESOLUTION = (32, 32)
INITIAL_POSITION = (0.0, 0.0, 1.0)


def _create_test_glb(directory: Path, name: str = "test_box") -> Path:
    """Save a simple box mesh as a .glb file in *directory*.

    Args:
        directory: Directory to save the file in.
        name: Stem name for the .glb file.

    Returns:
        Path to the created .glb file.
    """
    box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    path = directory / f"{name}.glb"
    box.export(str(path), file_type="glb")
    return path


class MeshEnvironmentTest(unittest.TestCase):
    """Tests for MeshEnvironment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._data_dir = Path(self._tmp.name)
        _create_test_glb(self._data_dir, "test_box")
        _create_test_glb(self._data_dir, "test_cube")
        self.env = MeshEnvironment(
            data_path=self._data_dir,
            resolution=RESOLUTION,
            agent_position=INITIAL_POSITION,
        )

    def tearDown(self):
        self.env.close()
        self._tmp.cleanup()

    # -- Helpers ----------------------------------------------------------

    def _get_agent_state(self) -> AgentState:
        """Step with no actions and return the agent's proprioceptive state.

        Returns:
            The agent's current state.
        """
        _, state = self.env.step([])
        return state[AGENT_ID]

    # ------------------------------------------------------------------
    # Lifecycle: reset / step / close
    # ------------------------------------------------------------------

    def test_reset_returns_observations_and_state(self):
        obs, state = self.env.reset()
        self.assertIn(AGENT_ID, obs)
        self.assertIn(SENSOR_ID_PATCH, obs[AGENT_ID])
        self.assertIn(SENSOR_ID_VIEW_FINDER, obs[AGENT_ID])
        self.assertIn(AGENT_ID, state)

    def test_reset_observation_shapes(self):
        obs, _ = self.env.reset()
        patch = obs[AGENT_ID][SENSOR_ID_PATCH]
        h, w = RESOLUTION
        self.assertEqual(patch["rgba"].shape, (h, w, 4))
        self.assertEqual(patch["depth"].shape, (h, w))
        self.assertEqual(patch["semantic"].shape, (h, w))

    def test_reset_observation_dtypes(self):
        obs, _ = self.env.reset()
        patch = obs[AGENT_ID][SENSOR_ID_PATCH]
        self.assertEqual(patch["rgba"].dtype, np.uint8)
        self.assertEqual(patch["depth"].dtype, np.float32)
        self.assertEqual(patch["semantic"].dtype, np.int32)

    def test_reset_restores_initial_state(self):
        # Move the agent away, then reset and verify it returns to initial state.
        self.env.actuate_move_forward(MoveForward(AGENT_ID, distance=0.5))
        _, state = self.env.reset()
        agent = state[AGENT_ID]
        np.testing.assert_array_almost_equal(
            np.array(agent.position), list(INITIAL_POSITION)
        )
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation), qt.as_float_array(qt.one)
        )

    def test_step_with_empty_actions(self):
        obs, state = self.env.step([])
        self.assertIn(AGENT_ID, obs)
        self.assertIn(AGENT_ID, state)

    def test_close_releases_renderer(self):
        self.env.close()
        self.assertIsNone(self.env._renderer)

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def test_add_object_returns_sequential_ids(self):
        id_0 = self.env.add_object("test_box", semantic_id=SemanticID(1))
        id_1 = self.env.add_object("test_cube", semantic_id=SemanticID(2))
        self.assertEqual(int(id_0), 0)
        self.assertEqual(int(id_1), 1)

    def test_add_unknown_object_raises_with_message(self):
        with self.assertRaisesRegex(ValueError, "Unknown mesh 'nonexistent'"):
            self.env.add_object("nonexistent")

    def test_remove_all_objects_clears_rendered_depth(self):
        self.env.add_object(
            "test_box", position=(0.0, 0.0, 0.0), semantic_id=SemanticID(1)
        )
        obs_before, _ = self.env.step([])
        depth_before = obs_before[AGENT_ID][SENSOR_ID_PATCH]["depth"]
        self.assertTrue(
            np.any(depth_before > 0), "Object should be visible before removal"
        )
        self.env.remove_all_objects()
        obs_after, _ = self.env.step([])
        depth_after = obs_after[AGENT_ID][SENSOR_ID_PATCH]["depth"]
        np.testing.assert_array_equal(depth_after, 0)

    def test_object_at_origin_produces_nonzero_depth(self):
        self.env.add_object(
            "test_box", position=(0.0, 0.0, 0.0), semantic_id=SemanticID(1)
        )
        obs, _ = self.env.step([])
        depth = obs[AGENT_ID][SENSOR_ID_PATCH]["depth"]
        self.assertTrue(np.any(depth > 0), "Expected some pixels with depth > 0")

    def test_semantic_id_covers_all_visible_pixels(self):
        sid = SemanticID(42)
        self.env.add_object("test_box", position=(0.0, 0.0, 0.0), semantic_id=sid)
        obs, _ = self.env.step([])
        semantic = obs[AGENT_ID][SENSOR_ID_PATCH]["semantic"]
        depth = obs[AGENT_ID][SENSOR_ID_PATCH]["depth"]
        on_object = depth > 0
        self.assertTrue(
            np.any(on_object), "Object must be visible to test semantic map"
        )
        np.testing.assert_array_equal(semantic[on_object], 42)

    def test_add_object_with_larger_scale_covers_more_pixels(self):
        self.env.add_object(
            "test_box",
            position=(0.0, 0.0, 0.0),
            scale=(0.5, 0.5, 0.5),
            semantic_id=SemanticID(1),
        )
        obs_small, _ = self.env.step([])
        small_pixels = np.sum(obs_small[AGENT_ID][SENSOR_ID_PATCH]["depth"] > 0)

        self.env.remove_all_objects()
        self.env.add_object(
            "test_box",
            position=(0.0, 0.0, 0.0),
            scale=(2.0, 2.0, 2.0),
            semantic_id=SemanticID(2),
        )
        obs_large, _ = self.env.step([])
        large_pixels = np.sum(obs_large[AGENT_ID][SENSOR_ID_PATCH]["depth"] > 0)

        self.assertGreater(large_pixels, small_pixels)

    def test_add_multiple_objects_both_visible(self):
        self.env.add_object(
            "test_box",
            position=(-0.3, 0.0, 0.0),
            semantic_id=SemanticID(1),
        )
        self.env.add_object(
            "test_cube",
            position=(0.3, 0.0, 0.0),
            semantic_id=SemanticID(2),
        )
        obs, _ = self.env.step([])
        depth = obs[AGENT_ID][SENSOR_ID_PATCH]["depth"]
        _, w = RESOLUTION
        left_half = depth[:, : w // 2]
        right_half = depth[:, w // 2 :]
        self.assertTrue(np.any(left_half > 0), "Left object should be visible")
        self.assertTrue(np.any(right_half > 0), "Right object should be visible")

    # ------------------------------------------------------------------
    # Action dispatch (action.act → actuate_*)
    # ------------------------------------------------------------------

    def test_step_dispatches_move_forward(self):
        _, state = self.env.step([MoveForward(AGENT_ID, distance=0.1)])
        agent = state[AGENT_ID]
        # Forward is -Z, so Z should decrease by 0.1.
        np.testing.assert_array_almost_equal(
            np.array(agent.position), [0.0, 0.0, 0.9], decimal=6
        )

    def test_step_dispatches_turn_left(self):
        _, state = self.env.step([TurnLeft(AGENT_ID, rotation_degrees=10.0)])
        agent = state[AGENT_ID]
        expected = qt.from_rotation_vector(np.array([0.0, np.radians(10.0), 0.0]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    # ------------------------------------------------------------------
    # Individual actuator methods
    # ------------------------------------------------------------------

    def test_move_forward(self):
        self.env.actuate_move_forward(MoveForward(AGENT_ID, distance=0.5))
        agent = self._get_agent_state()
        np.testing.assert_array_almost_equal(
            np.array(agent.position), [0.0, 0.0, 0.5], decimal=5
        )

    def test_turn_left_then_move_forward(self):
        self.env.actuate_turn_left(TurnLeft(AGENT_ID, rotation_degrees=90.0))
        self.env.actuate_move_forward(MoveForward(AGENT_ID, distance=1.0))
        agent = self._get_agent_state()
        # After 90-deg left turn around Y, forward (-Z) becomes -X.
        np.testing.assert_array_almost_equal(
            np.array(agent.position), [-1.0, 0.0, 1.0], decimal=5
        )

    def test_turn_right_then_move_forward(self):
        self.env.actuate_turn_right(TurnRight(AGENT_ID, rotation_degrees=90.0))
        self.env.actuate_move_forward(MoveForward(AGENT_ID, distance=1.0))
        agent = self._get_agent_state()
        # After 90-deg right turn around Y, forward (-Z) becomes +X.
        np.testing.assert_array_almost_equal(
            np.array(agent.position), [1.0, 0.0, 1.0], decimal=5
        )

    def test_look_up(self):
        self.env.actuate_look_up(LookUp(AGENT_ID, rotation_degrees=15.0))
        agent = self._get_agent_state()
        expected = qt.from_rotation_vector(np.array([np.radians(15.0), 0.0, 0.0]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    def test_look_down(self):
        self.env.actuate_look_down(LookDown(AGENT_ID, rotation_degrees=15.0))
        agent = self._get_agent_state()
        expected = qt.from_rotation_vector(np.array([-np.radians(15.0), 0.0, 0.0]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    def test_move_tangentially(self):
        self.env.actuate_move_tangentially(
            MoveTangentially(AGENT_ID, distance=0.5, direction=(1.0, 0.0, 0.0))
        )
        agent = self._get_agent_state()
        np.testing.assert_array_almost_equal(
            np.array(agent.position), [0.5, 0.0, 1.0], decimal=5
        )

    def test_orient_horizontal(self):
        left_dist, rot_deg, fwd_dist = 0.2, 10.0, 0.3
        self.env.actuate_orient_horizontal(
            OrientHorizontal(
                AGENT_ID,
                rotation_degrees=rot_deg,
                left_distance=left_dist,
                forward_distance=fwd_dist,
            )
        )
        agent = self._get_agent_state()

        # Recompute expected: move left → rotate Y → move forward.
        pos = np.array(INITIAL_POSITION, dtype=np.float64)
        rot = qt.one
        pos += qt.rotate_vectors(rot, np.array([-left_dist, 0.0, 0.0]))
        rot = rot * qt.from_rotation_vector(np.array([0.0, np.radians(-rot_deg), 0.0]))
        pos += qt.rotate_vectors(rot, np.array([0.0, 0.0, -fwd_dist]))

        np.testing.assert_array_almost_equal(np.array(agent.position), pos, decimal=5)
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(rot),
            decimal=6,
        )

    def test_orient_vertical(self):
        down_dist, rot_deg, fwd_dist = 0.2, 10.0, 0.3
        self.env.actuate_orient_vertical(
            OrientVertical(
                AGENT_ID,
                rotation_degrees=rot_deg,
                down_distance=down_dist,
                forward_distance=fwd_dist,
            )
        )
        agent = self._get_agent_state()

        # Recompute expected: move down → rotate X → move forward.
        pos = np.array(INITIAL_POSITION, dtype=np.float64)
        rot = qt.one
        pos += qt.rotate_vectors(rot, np.array([0.0, -down_dist, 0.0]))
        rot = rot * qt.from_rotation_vector(np.array([np.radians(rot_deg), 0.0, 0.0]))
        pos += qt.rotate_vectors(rot, np.array([0.0, 0.0, -fwd_dist]))

        np.testing.assert_array_almost_equal(np.array(agent.position), pos, decimal=5)
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(rot),
            decimal=6,
        )

    def test_set_agent_pose(self):
        target_pos = (1.0, 2.0, 3.0)
        target_rot = (0.707, 0.0, 0.707, 0.0)
        self.env.actuate_set_agent_pose(
            SetAgentPose(AGENT_ID, location=target_pos, rotation_quat=target_rot)
        )
        agent = self._get_agent_state()
        np.testing.assert_array_almost_equal(
            np.array(agent.position), list(target_pos), decimal=5
        )
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(qt.quaternion(*target_rot)),
            decimal=5,
        )

    def test_set_sensor_pose(self):
        target_pos = (0.1, 0.2, 0.3)
        target_rot = (1.0, 0.0, 0.0, 0.0)
        self.env.actuate_set_sensor_pose(
            SetSensorPose(AGENT_ID, location=target_pos, rotation_quat=target_rot)
        )
        agent = self._get_agent_state()
        sensor = agent.sensors[SENSOR_ID_PATCH]
        np.testing.assert_array_almost_equal(
            np.array(sensor.position), list(target_pos), decimal=5
        )
        np.testing.assert_array_almost_equal(
            qt.as_float_array(sensor.rotation),
            qt.as_float_array(qt.quaternion(*target_rot)),
            decimal=5,
        )

    def test_set_sensor_rotation(self):
        target_rot = (0.707, 0.707, 0.0, 0.0)
        self.env.actuate_set_sensor_rotation(
            SetSensorRotation(AGENT_ID, rotation_quat=target_rot)
        )
        agent = self._get_agent_state()
        sensor = agent.sensors[SENSOR_ID_PATCH]
        np.testing.assert_array_almost_equal(
            qt.as_float_array(sensor.rotation),
            qt.as_float_array(qt.quaternion(*target_rot)),
            decimal=3,
        )

    def test_set_sensor_pitch(self):
        self.env.actuate_set_sensor_pitch(SetSensorPitch(AGENT_ID, pitch_degrees=45.0))
        agent = self._get_agent_state()
        sensor = agent.sensors[SENSOR_ID_PATCH]
        expected = qt.from_rotation_vector(np.array([0.0, np.radians(45.0), 0.0]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(sensor.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    def test_set_agent_pitch(self):
        self.env.actuate_set_agent_pitch(SetAgentPitch(AGENT_ID, pitch_degrees=45.0))
        agent = self._get_agent_state()
        expected = qt.from_rotation_vector(np.array([0.0, np.radians(45.0), 0.0]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    def test_set_yaw(self):
        self.env.actuate_set_yaw(SetYaw(AGENT_ID, rotation_degrees=90.0))
        agent = self._get_agent_state()
        expected = qt.from_rotation_vector(np.array([0.0, 0.0, np.radians(90.0)]))
        np.testing.assert_array_almost_equal(
            qt.as_float_array(agent.rotation),
            qt.as_float_array(expected),
            decimal=6,
        )

    # ------------------------------------------------------------------
    # Proprioceptive state
    # ------------------------------------------------------------------

    def test_state_contains_agent_and_sensors(self):
        _, state = self.env.reset()
        agent = state[AGENT_ID]
        self.assertIn(SENSOR_ID_PATCH, agent.sensors)
        self.assertIn(SENSOR_ID_VIEW_FINDER, agent.sensors)

    def test_state_position_tracks_movement(self):
        target = (1.0, 2.0, 3.0)
        self.env.actuate_set_agent_pose(
            SetAgentPose(
                AGENT_ID,
                location=target,
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
            )
        )
        agent = self._get_agent_state()
        np.testing.assert_array_almost_equal(
            np.array(agent.position), list(target), decimal=5
        )

    # ------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
