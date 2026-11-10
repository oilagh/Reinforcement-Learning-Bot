"""
RocketSim needs the arena's collision meshes to simulate a soccar field. The real ones can only be
dumped from a Windows install of Rocket League (RLArenaCollisionDumper). If they are present in
`collision_meshes/soccar/`, we use them. Otherwise we procedurally build an approximate soccar arena
from the published field dimensions so training works anywhere (including macOS).

The approximation has the right walls, corners, goals, ramps (floor/wall and wall/ceiling curves)
and dimensions, but not the exact curvature details of the real mesh. A policy trained on it transfers
to the real game; for best results, fine-tune briefly on the real meshes once you have them.
"""
import math
import os
import struct
from pathlib import Path

import RocketSim as rsim

from rlgym.rocket_league.common_values import (BACK_NET_Y, BACK_WALL_Y, CEILING_Z, CORNER_CATHETUS_LENGTH,
                                               GOAL_CENTER_TO_POST, GOAL_HEIGHT, RAMP_HEIGHT, SIDE_WALL_X)

ROOT = Path(__file__).resolve().parent.parent
REAL_MESH_DIR = ROOT / "collision_meshes"
GENERATED_MESH_DIR = ROOT / "generated_meshes"

UU_TO_MESH = 1 / 50  # .cmf vertices are in Bullet units (1 BU = 50 uu)
RAMP_SEGMENTS = 8

_initialized_with = None


def _has_meshes(path: Path) -> bool:
    return (path / "soccar").is_dir() and any((path / "soccar").glob("*.cmf"))


def init_rocketsim() -> str:
    """
    Initializes RocketSim with the real meshes if available, otherwise generated ones.
    Must be called before creating any RocketSimEngine (RLGym's own init attempt is then a no-op).
    Returns which mesh source is in use: "real" or "generated".
    """
    global _initialized_with
    if _initialized_with is not None:
        return _initialized_with

    if _has_meshes(REAL_MESH_DIR):
        mesh_dir, source = REAL_MESH_DIR, "real"
    else:
        if not _has_meshes(GENERATED_MESH_DIR):
            write_generated_meshes(GENERATED_MESH_DIR)
        mesh_dir, source = GENERATED_MESH_DIR, "generated"

    rsim.init(str(mesh_dir))
    _initialized_with = source
    return source


def _offset_outline(d: float):
    """
    Field outline (octagon with chamfered corners) shrunk inward by d, counter-clockwise from above.
    The back walls are split at the goal posts so the goal mouth can be left open.
    Returns a list of (x, y, is_goal_mouth_edge_start) where the flag marks the edge from this vertex to the next.
    """
    sx = SIDE_WALL_X - d
    by = BACK_WALL_Y - d
    # Corner chamfer line |x| + |y| = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH, offset along its normal
    c = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH - d * math.sqrt(2)
    post = GOAL_CENTER_TO_POST
    return [
        (sx, -(c - sx), False),
        (sx, c - sx, False),
        (c - by, by, False),
        (post, by, True),
        (-post, by, False),
        (-(c - by), by, False),
        (-sx, c - sx, False),
        (-sx, -(c - sx), False),
        (-(c - by), -by, False),
        (-post, -by, True),
        (post, -by, False),
        (c - by, -by, False),
    ]


def _wall_rings():
    """(inward offset, height) for each ring of the lofted wall surface, floor ramp -> wall -> ceiling ramp."""
    r = RAMP_HEIGHT
    rings = []
    for i in range(RAMP_SEGMENTS + 1):
        t = 0.5 * math.pi * i / RAMP_SEGMENTS
        rings.append((r * (1 - math.sin(t)), r * (1 - math.cos(t))))
    rings.append((0.0, GOAL_HEIGHT))  # Exact ring at crossbar height so the goal mouth edge is clean
    for i in range(RAMP_SEGMENTS + 1):
        t = 0.5 * math.pi * i / RAMP_SEGMENTS
        rings.append((r * (1 - math.cos(t)), CEILING_Z - r + r * math.sin(t)))
    return rings


def build_arena_mesh():
    verts = []
    tris = []

    def add_vert(x, y, z):
        verts.append((x * UU_TO_MESH, y * UU_TO_MESH, z * UU_TO_MESH))
        return len(verts) - 1

    def add_quad(a, b, c, d):
        tris.append((a, b, c))
        tris.append((a, c, d))

    # Walls with floor/ceiling ramps
    rings = _wall_rings()
    ring_idx = []
    for d, z in rings:
        ring_idx.append([add_vert(x, y, z) for x, y, _ in _offset_outline(d)])
    mouth = [flag for _, _, flag in _offset_outline(0)]
    n = len(mouth)
    for r in range(len(rings) - 1):
        top_z = rings[r + 1][1]
        for i in range(n):
            if mouth[i] and top_z <= GOAL_HEIGHT + 1e-3:
                continue  # Leave the goal mouth open
            j = (i + 1) % n
            add_quad(ring_idx[r][i], ring_idx[r][j], ring_idx[r + 1][j], ring_idx[r + 1][i])

    # Goal boxes
    for sign in (1, -1):
        y0, y1 = sign * BACK_WALL_Y, sign * BACK_NET_Y
        p, h = GOAL_CENTER_TO_POST, GOAL_HEIGHT
        for x in (p, -p):  # Side nets
            add_quad(add_vert(x, y0, 0), add_vert(x, y1, 0), add_vert(x, y1, h), add_vert(x, y0, h))
        add_quad(add_vert(-p, y1, 0), add_vert(p, y1, 0), add_vert(p, y1, h), add_vert(-p, y1, h))  # Back net
        add_quad(add_vert(-p, y0, h), add_vert(p, y0, h), add_vert(p, y1, h), add_vert(-p, y1, h))  # Roof

    return verts, tris


def write_generated_meshes(mesh_dir: Path):
    verts, tris = build_arena_mesh()
    os.makedirs(mesh_dir / "soccar", exist_ok=True)
    with open(mesh_dir / "soccar" / "generated_arena.cmf", "wb") as f:
        f.write(struct.pack("<ii", len(tris), len(verts)))
        for t in tris:
            f.write(struct.pack("<iii", *t))
        for v in verts:
            f.write(struct.pack("<fff", *v))
