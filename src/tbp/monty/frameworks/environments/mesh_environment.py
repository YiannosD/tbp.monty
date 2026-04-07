# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Environment for rendering 3D mesh models loaded from .glb files.

This module provides a ``MeshEnvironment`` that loads ``.glb`` meshes from disk
and renders RGBA, depth, and semantic observations using pyrender.  It implements
the ``SimulatedObjectEnvironment`` protocol and all 14 ``*Actuator`` protocols
so that it can be used with the standard Monty pipeline (including
``EnvironmentInterfacePerObject`` and the ``DepthTo3DLocations`` transform).

The coordinate system matches Habitat: +X right, +Y up, +Z toward viewer,
forward is -Z.

Note:
    Headless rendering requires an OpenGL backend.  Set the environment variable
    ``PYOPENGL_PLATFORM=egl`` (or ``osmesa``) before launching the experiment
    when no display is available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pyrender
import quaternion as qt
import trimesh

from tbp.monty.frameworks.actions.actions import (
    Action,
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
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    SemanticID,
)
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.path import monty_data_path

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["MeshEnvironment"]

logger = logging.getLogger(__name__)

# Local axis indices matching Habitat conventions.
_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2

# Default agent / sensor IDs used when constructing observations.
_AGENT_ID = AgentID("agent_id_0")
_SENSOR_ID_PATCH = SensorID("patch")
_SENSOR_ID_VIEW_FINDER = SensorID("view_finder")


# ---------------------------------------------------------------------------
# Thin wrappers around trimesh / pyrender
# ---------------------------------------------------------------------------
# The typing guide recommends isolating poorly-typed third-party libraries so
# that the rest of the codebase receives well-typed values.  The helpers below
# keep all trimesh and pyrender usage in one place.


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center a mesh at the origin and scale it to fit a unit bounding box.

    Args:
        mesh: The mesh to normalize.

    Returns:
        The same mesh, modified in place.
    """
    mesh.apply_translation(-mesh.bounding_box.centroid)
    max_extent = np.max(mesh.bounding_box.extents)
    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)
    return mesh


def _load_trimesh(path: Path) -> trimesh.Trimesh | None:
    """Load a ``.glb`` file and return it as a ``trimesh.Trimesh``.

    Args:
        path: Path to a ``.glb`` file.

    Returns:
        A ``trimesh.Trimesh``, or ``None`` if loading or conversion failed.
    """
    try:
        scene_or_mesh = trimesh.load(str(path))
    except (ValueError, OSError):
        logger.warning("Failed to load mesh from %s", path)
        return None

    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            logger.warning("Empty scene in %s", path)
            return None
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh

    if not isinstance(mesh, trimesh.Trimesh):
        logger.warning("Could not convert %s to Trimesh", path)
        return None

    return mesh


def _create_pyrender_components(
    width: int,
    height: int,
    yfov: float,
    aspect_ratio: float,
) -> tuple[pyrender.Scene, pyrender.Node, pyrender.OffscreenRenderer]:
    """Create a pyrender ``Scene``, camera, light, and ``OffscreenRenderer``.

    Args:
        width: Viewport width in pixels.
        height: Viewport height in pixels.
        yfov: Vertical field of view in radians.
        aspect_ratio: Width / height.

    Returns:
        Tuple of (scene, camera_node, renderer).
    """
    scene = pyrender.Scene(
        ambient_light=np.array([0.3, 0.3, 0.3, 1.0]),
    )
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
    camera_node = scene.add(camera)

    # Directional light pointing down and slightly forward.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, :3] = qt.as_rotation_matrix(
        qt.from_rotation_vector(np.array([np.pi / 4, 0, 0]))
    )
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
    )

    return scene, camera_node, renderer


def _trimesh_to_pyrender_mesh(
    mesh: trimesh.Trimesh,
    scale: VectorXYZ,
) -> pyrender.Mesh:
    """Convert a trimesh mesh to a pyrender ``Mesh``, applying scale.

    Args:
        mesh: A ``trimesh.Trimesh`` object.
        scale: (sx, sy, sz) scale factors.

    Returns:
        A ``pyrender.Mesh``.
    """
    scaled = trimesh.Trimesh(
        vertices=mesh.vertices * np.array(scale),
        faces=mesh.faces.copy(),
        vertex_normals=mesh.vertex_normals.copy(),
        process=False,
    )
    if mesh.visual is not None:
        scaled.visual = mesh.visual.copy()

    return pyrender.Mesh.from_trimesh(scaled)


def _render_scene(
    renderer: pyrender.OffscreenRenderer,
    scene: pyrender.Scene,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the current scene and return RGBA and depth images.

    Args:
        renderer: A ``pyrender.OffscreenRenderer``.
        scene: A ``pyrender.Scene``.

    Returns:
        ``(rgba, depth)`` where *rgba* is ``(H, W, 4) uint8`` and *depth* is
        ``(H, W) float32``.  Depth is 0.0 for background pixels.
    """
    color, depth = renderer.render(
        scene,
        flags=pyrender.RenderFlags.RGBA,
    )
    return color.astype(np.uint8), depth.astype(np.float32)


# ---------------------------------------------------------------------------
# MeshEnvironment
# ---------------------------------------------------------------------------


class MeshEnvironment:
    """3D mesh rendering environment implementing ``SimulatedObjectEnvironment``.

    Loads ``.glb`` meshes from ``data_path``, renders RGBA + depth + semantic
    observations via pyrender, and tracks agent / sensor pose using
    numpy-quaternion.  Compatible with ``EnvironmentInterfacePerObject`` and the
    ``DepthTo3DLocations`` transform.

    Args:
        data_path: Directory containing ``.glb`` mesh files.  If ``None``,
            defaults to ``$MONTY_DATA/Trees``.
        resolution: ``(height, width)`` of the rendered images.
        hfov: Horizontal field of view in degrees.
        zoom: Zoom factor applied to the camera (matches the
            ``DepthTo3DLocations`` intrinsic model ``fx = tan(hfov/2) / zoom``).
        agent_position: Initial world position of the agent as ``(x, y, z)``.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        resolution: tuple[int, int] = (64, 64),
        hfov: float = 90.0,
        zoom: float = 1.0,
        agent_position: VectorXYZ = (0.0, 1.5, 1.0),
        normalize_meshes: bool = False,
    ) -> None:
        super().__init__()

        self._data_path = monty_data_path(data_path, "Trees")
        self._resolution = resolution
        self._hfov = hfov
        self._zoom = zoom
        self._initial_agent_position = np.array(agent_position, dtype=np.float64)
        self._normalize_on_load = normalize_meshes

        # ---- Load meshes ----
        self._mesh_cache: dict[str, trimesh.Trimesh] = self._load_all_meshes()

        # ---- Pyrender scene / renderer ----
        height, width = resolution
        hfov_rad = np.radians(hfov)
        yfov = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) * height / (width * zoom))
        (
            self._scene,
            self._camera_node,
            self._renderer,
        ) = _create_pyrender_components(width, height, yfov, width / height)

        # ---- Agent / sensor state ----
        self._agent_position = self._initial_agent_position.copy()
        self._agent_rotation: np.quaternion = qt.one
        self._sensor_position = np.zeros(3, dtype=np.float64)
        self._sensor_rotation: np.quaternion = qt.one

        # ---- Object tracking ----
        self._object_nodes: dict[int, pyrender.Node] = {}
        self._object_semantic_ids: dict[int, SemanticID | None] = {}
        self._next_object_id: int = 0


    # ------------------------------------------------------------------
    # SimulatedObjectEnvironment protocol
    # ------------------------------------------------------------------

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Apply actions and return the resulting observations.

        Each action dispatches to the corresponding ``actuate_*`` method via
        ``action.act(self)``.

        Args:
            actions: The actions to apply to the environment.

        Returns:
            The current observations and proprioceptive state.
        """
        for action in actions:
            action.act(self)
        return self._build_observations()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset agent / sensor state to initial values and return observations.

        Returns:
            The environment's initial observations and proprioceptive state.
        """
        self._agent_position = self._initial_agent_position.copy()
        self._agent_rotation = qt.one
        self._sensor_position = np.zeros(3, dtype=np.float64)
        self._sensor_rotation = qt.one
        return self._build_observations()

    def close(self) -> None:
        """Release the pyrender offscreen renderer."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,  # noqa: ARG002
    ) -> ObjectID:
        """Add a mesh object to the scene.

        Args:
            name: Stem name of the ``.glb`` file (without extension).
            position: World position of the object.
            rotation: Rotation quaternion ``(w, x, y, z)`` of the object.
            scale: ``(sx, sy, sz)`` scale factors.
            semantic_id: Semantic class label for the object.
            primary_target_object: Accepted for protocol compatibility (unused).

        Returns:
            The ID of the added object.

        Raises:
            ValueError: If *name* does not correspond to a loaded mesh.
        """
        if name not in self._mesh_cache:
            raise ValueError(
                f"Unknown mesh '{name}'. Available: {sorted(self._mesh_cache.keys())}"
            )

        trimesh_mesh = self._mesh_cache[name]
        pyrender_mesh = _trimesh_to_pyrender_mesh(trimesh_mesh, scale)

        # Build 4x4 world pose for the object node.
        pos = np.array(position, dtype=np.float64)
        if isinstance(rotation, qt.quaternion):
            rot = rotation
        else:
            w, x, y, z = rotation
            rot = qt.quaternion(w, x, y, z)
        pose = np.eye(4)
        pose[:3, :3] = qt.as_rotation_matrix(rot)
        pose[:3, 3] = pos

        node = self._scene.add(pyrender_mesh, pose=pose)

        object_id = ObjectID(self._next_object_id)
        self._next_object_id += 1
        self._object_nodes[object_id] = node
        self._object_semantic_ids[object_id] = semantic_id
        return object_id

    def remove_all_objects(self) -> None:
        """Remove all mesh objects from the scene."""
        for node in self._object_nodes.values():
            self._scene.remove_node(node)
        self._object_nodes.clear()
        self._object_semantic_ids.clear()

    # ------------------------------------------------------------------
    # Actuator methods (14 protocols from actions.py)
    # ------------------------------------------------------------------

    def actuate_move_forward(self, action: MoveForward) -> None:
        """Translate the agent along its local -Z axis (forward)."""
        self._move_along_local(_Z_AXIS, -action.distance)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        """Rotate the agent left around the local Y axis."""
        self._rotate_local(_Y_AXIS, action.rotation_degrees)

    def actuate_turn_right(self, action: TurnRight) -> None:
        """Rotate the agent right around the local Y axis."""
        self._rotate_local(_Y_AXIS, -action.rotation_degrees)

    def actuate_look_up(self, action: LookUp) -> None:
        """Rotate the agent upward around the local X axis."""
        self._rotate_local(_X_AXIS, action.rotation_degrees)

    def actuate_look_down(self, action: LookDown) -> None:
        """Rotate the agent downward around the local X axis."""
        self._rotate_local(_X_AXIS, -action.rotation_degrees)

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        """Translate the agent along the given local direction vector."""
        direction = np.array(action.direction, dtype=np.float64)
        world_dir = qt.rotate_vectors(self._agent_rotation, direction)
        self._agent_position += world_dir * action.distance

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        """Compound action: move left, rotate Y, move forward.

        Keeps the fixation point approximately stable when orbiting an object
        horizontally.
        """
        self._move_along_local(_X_AXIS, -action.left_distance)
        self._rotate_local(_Y_AXIS, -action.rotation_degrees)
        self._move_along_local(_Z_AXIS, -action.forward_distance)

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        """Compound action: move down, rotate X, move forward.

        Keeps the fixation point approximately stable when orbiting an object
        vertically.
        """
        self._move_along_local(_Y_AXIS, -action.down_distance)
        self._rotate_local(_X_AXIS, action.rotation_degrees)
        self._move_along_local(_Z_AXIS, -action.forward_distance)

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        """Set the agent to an absolute world position and rotation."""
        self._agent_position = np.array(action.location, dtype=np.float64)
        w, x, y, z = action.rotation_quat
        self._agent_rotation = qt.quaternion(w, x, y, z)

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        """Set the sensor position and rotation relative to the agent."""
        self._sensor_position = np.array(action.location, dtype=np.float64)
        w, x, y, z = action.rotation_quat
        self._sensor_rotation = qt.quaternion(w, x, y, z)

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        """Set the sensor rotation relative to the agent."""
        w, x, y, z = action.rotation_quat
        self._sensor_rotation = qt.quaternion(w, x, y, z)

    def actuate_set_sensor_pitch(self, action: SetSensorPitch) -> None:
        """Set the sensor pitch to an absolute rotation about the Y axis."""
        angle_rad = np.radians(action.pitch_degrees)
        self._sensor_rotation = qt.from_rotation_vector(np.array([0.0, angle_rad, 0.0]))

    def actuate_set_agent_pitch(self, action: SetAgentPitch) -> None:
        """Set the agent pitch to an absolute rotation about the Y axis."""
        angle_rad = np.radians(action.pitch_degrees)
        self._agent_rotation = qt.from_rotation_vector(np.array([0.0, angle_rad, 0.0]))

    def actuate_set_yaw(self, action: SetYaw) -> None:
        """Set the agent yaw to an absolute rotation about the Z axis."""
        angle_rad = np.radians(action.rotation_degrees)
        self._agent_rotation = qt.from_rotation_vector(np.array([0.0, 0.0, angle_rad]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_along_local(self, axis: int, distance: float) -> None:
        """Translate the agent along a local axis by *distance*.

        Args:
            axis: 0 (X), 1 (Y), or 2 (Z).
            distance: Signed distance to move.
        """
        local_dir = np.zeros(3, dtype=np.float64)
        local_dir[axis] = 1.0
        world_dir = qt.rotate_vectors(self._agent_rotation, local_dir)
        self._agent_position += world_dir * distance

    def _rotate_local(self, axis: int, degrees: float) -> None:
        """Rotate the agent around a local axis by *degrees*.

        Args:
            axis: 0 (X), 1 (Y), or 2 (Z).
            degrees: Rotation in degrees (positive follows the right-hand rule
                around the local axis).
        """
        local_axis = np.zeros(3, dtype=np.float64)
        local_axis[axis] = 1.0
        angle_rad = np.radians(degrees)
        local_rotation = qt.from_rotation_vector(local_axis * angle_rad)
        self._agent_rotation = self._agent_rotation * local_rotation

    def _load_all_meshes(self) -> dict[str, trimesh.Trimesh]:
        """Load all ``.glb`` files from :attr:`_data_path`.

        Returns:
            Mapping from mesh stem name to trimesh object.
        """
        meshes: dict[str, trimesh.Trimesh] = {}
        if not self._data_path.is_dir():
            logger.warning("Data path does not exist: %s", self._data_path)
            return meshes

        for glb_file in sorted(self._data_path.glob("*.glb")):
            mesh = _load_trimesh(glb_file)
            if mesh is not None:
                if self._normalize_on_load:
                    mesh = _normalize_mesh(mesh)
                meshes[glb_file.stem] = mesh

        logger.info("Loaded %d meshes from %s", len(meshes), self._data_path)
        return meshes

    def _update_camera_pose(self) -> None:
        """Set the pyrender camera node pose to the current sensor world pose."""
        world_position = self._agent_position + qt.rotate_vectors(
            self._agent_rotation, self._sensor_position
        )
        world_rotation = self._agent_rotation * self._sensor_rotation

        pose = np.eye(4)
        pose[:3, :3] = qt.as_rotation_matrix(world_rotation)
        pose[:3, 3] = world_position
        self._scene.set_pose(self._camera_node, pose)

    def _build_semantic(self, depth: np.ndarray) -> np.ndarray:
        """Build a semantic segmentation map from the depth buffer.

        For single-object scenes, all non-background pixels receive the
        object's ``semantic_id``.  Background pixels (``depth == 0``) stay 0.

        Args:
            depth: ``(H, W)`` depth buffer from the renderer.

        Returns:
            ``(H, W)`` int32 semantic map.
        """
        semantic = np.zeros(depth.shape, dtype=np.int32)
        for sid in self._object_semantic_ids.values():
            if sid is not None:
                semantic[depth > 0] = int(sid)
        return semantic

    def _build_observations(self) -> tuple[Observations, ProprioceptiveState]:
        """Render the scene and package observations + proprioceptive state.

        Returns:
            Observations dict and proprioceptive state, matching the format
            expected by sensor modules and the ``DepthTo3DLocations`` transform.
        """
        self._update_camera_pose()
        rgba, depth = _render_scene(self._renderer, self._scene)
        semantic = self._build_semantic(depth)

        obs = Observations(
            {
                _AGENT_ID: AgentObservations(
                    {
                        _SENSOR_ID_PATCH: SensorObservation(
                            {
                                "rgba": rgba,
                                "depth": depth,
                                "semantic": semantic,
                            }
                        ),
                        _SENSOR_ID_VIEW_FINDER: SensorObservation(
                            {
                                "rgba": rgba,
                                "depth": depth,
                                "semantic": semantic,
                            }
                        ),
                    }
                )
            }
        )

        state = ProprioceptiveState(
            {
                _AGENT_ID: AgentState(
                    sensors={
                        _SENSOR_ID_PATCH: SensorState(
                            position=self._sensor_position.copy(),
                            rotation=self._sensor_rotation,
                        ),
                        _SENSOR_ID_VIEW_FINDER: SensorState(
                            position=self._sensor_position.copy(),
                            rotation=self._sensor_rotation,
                        ),
                    },
                    rotation=self._agent_rotation,
                    position=self._agent_position.copy(),
                )
            }
        )

        return obs, state
