# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Visualization utilities for unsupervised tree learning experiments.

Provides functions for:
- Dendrogram images showing species-to-model groupings
- Point cloud grid images of learned models
- Interactive 3D overlays of learned models on source meshes

These complement the general-purpose plotting in ``plot_utils_dev`` with
visualizations specific to the tree learning workflow.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def load_learned_models(model_path: Path) -> tuple[dict, dict]:
    """Load graph_id_to_target and graph_memory from a saved model.

    Args:
        model_path: Path to a ``model.pt`` file.

    Returns:
        Tuple of (graph_id_to_target, graph_memory) dicts.
    """
    sd = torch.load(model_path, weights_only=False)
    lm = sd["lm_dict"][0]
    return lm["graph_id_to_target"], lm["graph_memory"]


def render_point_cloud(
    positions: np.ndarray,
    width: int = 500,
    height: int = 500,
) -> Image.Image:
    """Render a 3D point cloud to a PIL Image using plotly.

    Points are colored by height (Y coordinate) using the Viridis colorscale.

    Args:
        positions: ``(N, 3)`` array of point positions.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        PIL Image of the rendered point cloud.
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 2],
                z=positions[:, 1],
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=positions[:, 1],
                    colorscale="Viridis",
                    opacity=0.9,
                ),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.0, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=width,
        height=height,
        paper_bgcolor="white",
    )
    img_bytes = fig.to_image(format="png", scale=2)
    return Image.open(BytesIO(img_bytes))


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------

_COLORS = [
    "#2196F3",
    "#FF9800",
    "#4CAF50",
    "#E91E63",
    "#9C27B0",
    "#00BCD4",
    "#FF5722",
    "#607D8B",
    "#8BC34A",
    "#795548",
    "#3F51B5",
    "#CDDC39",
    "#009688",
    "#F44336",
    "#673AB7",
    "#FFC107",
    "#03A9F4",
    "#FFEB3B",
]


def draw_dendrogram(
    groups_by_category: list[tuple[str, list]],
    out_path: Path,
) -> None:
    """Draw a text-only dendrogram of species→model groupings.

    Args:
        groups_by_category: List of ``(category_name, groups)`` where each
            group is ``(graph_id, species_list, n_points, positions)``.
        out_path: Where to save the PNG.
    """
    font_title = _get_font(28, bold=True)
    font_heading = _get_font(20, bold=True)
    font_model = _get_font(16, bold=True)
    font_species = _get_font(15)

    row_h = 30
    section_gap = 50
    top_pad = 70
    left_pad = 40

    total_rows = 0
    for _, groups in groups_by_category:
        if not groups:
            continue
        total_rows += 1
        for _, species, _, _ in groups:
            total_rows += max(len(species), 1)
        total_rows += 1

    width = 700
    height = top_pad + total_rows * row_h + section_gap * len(groups_by_category) + 40

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Unsupervised Tree Learning"
    tw = draw.textlength(title, font=font_title)
    draw.text(((width - tw) / 2, 15), title, fill="black", font=font_title)

    y = top_pad
    color_idx = 0

    for cat_name, groups in groups_by_category:
        if not groups:
            continue

        draw.text((left_pad, y), cat_name, fill="black", font=font_heading)
        y += row_h + 10

        for gid, species, n_pts, _ in groups:
            color = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1

            model_label = f"{gid} ({n_pts} pts)"
            draw.text((left_pad + 20, y), model_label, fill=color, font=font_model)

            x_bracket = left_pad + 280
            x_species = x_bracket + 30

            if len(species) == 1:
                draw.line(
                    [(x_bracket - 10, y + 10), (x_bracket + 15, y + 10)],
                    fill=color,
                    width=2,
                )
                draw.text(
                    (x_species, y - 2),
                    species[0],
                    fill="black",
                    font=font_species,
                )
                y += row_h
            else:
                y_start = y
                for sp in species:
                    draw.line(
                        [(x_bracket, y + 10), (x_bracket + 15, y + 10)],
                        fill=color,
                        width=2,
                    )
                    draw.text(
                        (x_species, y - 2),
                        sp,
                        fill="black",
                        font=font_species,
                    )
                    y += row_h
                draw.line(
                    [(x_bracket, y_start + 10), (x_bracket, y - row_h + 10)],
                    fill=color,
                    width=2,
                )
                y_mid = (y_start + y - row_h) // 2 + 10
                draw.line(
                    [(x_bracket - 10, y_mid), (x_bracket, y_mid)],
                    fill=color,
                    width=2,
                )

        y += section_gap

    canvas = canvas.crop((0, 0, width, y))
    canvas.save(out_path)


def draw_individual_grid(
    groups: list,
    out_path: Path,
    cols: int = 4,
    thumb_size: int = 400,
) -> None:
    """Draw a grid of point cloud thumbnails for single-species models.

    Args:
        groups: List of ``(graph_id, species_list, n_points, positions)``.
        out_path: Where to save the PNG.
        cols: Number of columns in the grid.
        thumb_size: Size of each thumbnail in pixels.
    """
    n = len(groups)
    rows = (n + cols - 1) // cols
    font_label = _get_font(20, bold=True)

    thumbs = []
    labels = []
    for _, species, n_pts, positions in groups:
        thumbs.append(
            render_point_cloud(positions, width=thumb_size, height=thumb_size)
        )
        labels.append(f"{species[0]} ({n_pts} pts)")

    tw = thumbs[0].width
    th = thumbs[0].height
    label_h = 35
    pad = 10
    title_h = 50

    canvas_w = cols * (tw + pad) + pad
    canvas_h = title_h + rows * (th + label_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = _get_font(24, bold=True)
    title = f"Individual models ({n} species)"
    ttw = draw.textlength(title, font=font_title)
    draw.text(((canvas_w - ttw) / 2, 12), title, fill="black", font=font_title)

    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = title_h + r * (th + label_h + pad)
        canvas.paste(thumb, (x, y))
        lw = draw.textlength(label, font=font_label)
        draw.text(
            (x + (tw - lw) / 2, y + th + 2),
            label,
            fill="black",
            font=font_label,
        )

    canvas.save(out_path)


def draw_group_image(
    gid: str,
    species: list[str],
    n_pts: int,
    positions: np.ndarray,
    out_path: Path,
    thumb_size: int = 600,
) -> None:
    """Draw a single point cloud image for a multi-species model.

    Args:
        gid: Graph ID of the model.
        species: List of species names in this model.
        n_pts: Number of points in the model.
        positions: ``(N, 3)`` array of point positions.
        out_path: Where to save the PNG.
        thumb_size: Size of the point cloud render.
    """
    img = render_point_cloud(positions, width=thumb_size, height=thumb_size)

    font_title = _get_font(22, bold=True)
    font_species = _get_font(18)

    label = " + ".join(species)
    title = f"{gid} ({n_pts} pts)"

    tw = img.width
    title_h = 40
    label_h = 35
    canvas_h = title_h + img.height + label_h + 10
    canvas = Image.new("RGB", (tw, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    ttw = draw.textlength(title, font=font_title)
    draw.text(((tw - ttw) / 2, 8), title, fill="black", font=font_title)

    canvas.paste(img, (0, title_h))

    lw = draw.textlength(label, font=font_species)
    draw.text(
        ((tw - lw) / 2, title_h + img.height + 5),
        label,
        fill="gray",
        font=font_species,
    )

    canvas.save(out_path)


def build_dendrogram(model_path: Path, out_dir: Path) -> None:
    """Generate all dendrogram and point cloud visualizations for a model.

    Produces:
    - ``dendrogram.png``: text dendrogram of species→model groupings
    - ``pointclouds_individual.png``: grid of single-species models
    - ``pointclouds_group_N.png``: one per combined model
    - ``pointclouds_catch_all.png``: the catch-all model (if any)

    Args:
        model_path: Path to a ``model.pt`` file.
        out_dir: Directory to write output images.
    """
    g2t, gmem = load_learned_models(model_path)
    n_total_species = len({s for targets in g2t.values() for s in targets})

    all_groups = []
    for gid in sorted(g2t.keys()):
        species = sorted(t.replace("_tree", "") for t in g2t[gid])
        pos = np.array(gmem[gid]["patch"].pos)
        all_groups.append((gid, species, pos.shape[0], pos))

    individual = [g for g in all_groups if len(g[1]) == 1]
    catch_all = [g for g in all_groups if len(g[1]) == n_total_species]
    combined = [g for g in all_groups if 1 < len(g[1]) < n_total_species]

    out_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        ("Individual models", individual),
        ("Combined models", combined),
        ("Catch-all model", catch_all),
    ]
    draw_dendrogram(categories, out_dir / "dendrogram.png")

    if individual:
        draw_individual_grid(individual, out_dir / "pointclouds_individual.png")

    for i, (gid, species, n_pts, pos) in enumerate(combined):
        draw_group_image(
            gid,
            species,
            n_pts,
            pos,
            out_dir / f"pointclouds_group_{i}.png",
        )

    for gid, species, n_pts, pos in catch_all:
        draw_group_image(
            gid,
            species,
            n_pts,
            pos,
            out_dir / "pointclouds_catch_all.png",
            thumb_size=700,
        )


# ---------------------------------------------------------------------------
# Interactive 3D overlay
# ---------------------------------------------------------------------------


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center a mesh at the origin and scale to fit a unit bounding box.

    Returns:
        The modified mesh (same object, mutated in place).
    """
    mesh.apply_translation(-mesh.bounding_box.centroid)
    max_extent = np.max(mesh.bounding_box.extents)
    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)
    return mesh


def load_source_mesh(
    data_path: Path,
    tree_name: str,
    max_verts: int = 20000,
) -> np.ndarray:
    """Load and normalize a source mesh, returning its vertices.

    Args:
        data_path: Directory containing ``.glb`` files.
        tree_name: Stem name of the ``.glb`` file.
        max_verts: Subsample to this many vertices if the mesh is larger.

    Returns:
        ``(N, 3)`` array of vertex positions.
    """
    glb = data_path / f"{tree_name}.glb"
    mesh = trimesh.load(str(glb))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    mesh = _normalize_mesh(mesh)
    verts = np.array(mesh.vertices)
    if len(verts) > max_verts:
        idx = np.random.default_rng(0).choice(len(verts), max_verts, replace=False)
        verts = verts[idx]
    return verts


def build_model_overlay(
    mesh_verts: np.ndarray,
    model_pts: np.ndarray,
    tree_name: str,
    graph_id: str,
) -> go.Figure:
    """Build an interactive 3D overlay of learned model on source mesh.

    Args:
        mesh_verts: ``(N, 3)`` source mesh vertices.
        model_pts: ``(M, 3)`` learned model points.
        tree_name: Name of the tree species.
        graph_id: Graph ID of the learned model.

    Returns:
        Plotly Figure that can be displayed or saved as HTML.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=mesh_verts[:, 0],
            y=mesh_verts[:, 2],
            z=mesh_verts[:, 1],
            mode="markers",
            marker=dict(size=2, color="rgba(0,0,0,0.5)"),
            name=f"Source mesh ({len(mesh_verts)} verts)",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=model_pts[:, 0],
            y=model_pts[:, 2],
            z=model_pts[:, 1],
            mode="markers",
            marker=dict(
                size=4,
                color=model_pts[:, 1],
                colorscale="Viridis",
                opacity=0.9,
            ),
            name=f"Learned model ({len(model_pts)} pts)",
        )
    )

    fig.update_layout(
        title=(f"{tree_name} — {graph_id} ({len(model_pts)} learned points)"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Z",
            zaxis_title="Y (height)",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
        width=1000,
        height=800,
    )
    return fig
