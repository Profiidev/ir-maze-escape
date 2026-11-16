"""Seeded Minecraft-stronghold maze -> Webots world (1 MC block = 0.1 m)."""

import heapq
import json
import math
import random
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ir import pieces

W = H = 30  # cells per floor edge
Z = 1  # floors
P = 0.4  # cell pitch: 4 blocks = 3 interior + 1 wall
T = 0.1  # wall thickness
LV = 0.6  # floor pitch
WH = 0.4  # wall height
SLAB = 0.02
RAMP = 5  # cells per ramp (rise LV over RAMP * P -> ~17 deg)
MIN_PATH = 150  # min shortest-path steps start -> portal
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
JAR = Path.home() / ".local/share/io.profidev.proton/minecraft/versions/26.3/26.3.jar"
WALLS = ["stone_bricks"] * 12 + ["mossy_stone_bricks"] * 4 + ["cracked_stone_bricks"] * 3 + ["chiseled_stone_bricks"]
FLOOR = {
    "library": "oak_planks",
    "prison": "cobblestone",
    "crossing": "mossy_stone_bricks",
    "entrance": "chiseled_stone_bricks",
}
BLOCK = 0.1
TEXTURES = sorted(
    {*WALLS, *FLOOR.values(), "stone_bricks", "bookshelf", "end_portal_frame_top", "end_portal_frame_eye", "end_portal_frame_side", "oak_planks", "cobblestone"}
)
HEADER = Path(__file__).parents[2].joinpath("worlds/maze.wbt")


@dataclass
class Room:
    kind: str
    x: int
    y: int
    z: int
    w: int
    h: int
    boxes: list = None
    doors: set = None  # {(cell, dir)} allowed crossings of the room boundary
    start: tuple = None  # interior blocks
    goal: tuple = None

    def cells(self):
        return {(self.x + i, self.y + j, self.z) for i in range(self.w) for j in range(self.h)}

    def cell_of(self, blk):
        return (self.x + blk[0] // 4, self.y + blk[1] // 4, self.z)


@dataclass
class Maze:
    seed: int
    rooms: list
    seen: set  # open cells
    passages: set  # frozenset({cell, cell}) on one floor
    ramps: list  # (low cell, dir, high cell)
    start: tuple
    goal: tuple
    path: list
    loops: int


def ramp_cols(low, d):
    return [(low[0] + k * d[0], low[1] + k * d[1]) for k in range(1, RAMP + 1)]


def generate(seed, attempt):
    rng = random.Random(f"{seed}:{attempt}")
    room_at = {}
    rooms = []
    specs = [("entrance", Z - 1), ("portal", 0)] + [(k, None) for k, n in
                                                      [("library", 4), ("prison", 4), ("crossing", 6), ("chest", 6)]
                                                      for _ in range(n)]
    for kind, z in specs:
        boxes, doors, start, goal, w, h = pieces.make(kind, rng, rng.randrange(4))
        for _ in range(200):
            r = Room(kind, rng.randrange(W - w + 1), rng.randrange(H - h + 1), rng.randrange(Z) if z is None else z, w, h,
                     boxes, start=start, goal=goal)
            far = kind != "portal" or abs(r.x - rooms[0].x) + abs(r.y - rooms[0].y) >= W
            if far and not r.cells() & room_at.keys():
                break
        else:
            return None
        r.doors = {((r.x + ci, r.y + cj, r.z), d) for ci, cj, d in doors}
        rooms.append(r)
        room_at.update({c: r for c in r.cells()})

    def can_pass(a, b, d):
        """Edge a->b (direction d) respects the fixed doors of any room it crosses."""
        ra, rb = room_at.get(a), room_at.get(b)
        if ra is not None and ra is rb:
            return True
        return (ra is None or (a, d) in ra.doors) and (rb is None or (b, (-d[0], -d[1])) in rb.doors)

    seen, blocked, passages, ramps = set(), set(), set(), []

    def enter(c):
        cs = room_at[c].cells() if c in room_at else {c}
        seen.update(cs)
        for a in cs:
            for dx, dy in DIRS:
                if (b := (a[0] + dx, a[1] + dy, a[2])) in cs:
                    passages.add(frozenset((a, b)))
        return sorted(cs)

    def free(c):
        return 0 <= c[0] < W and 0 <= c[1] < H and c not in seen and c not in blocked

    def moves(c, last):
        out = []
        for d in DIRS:
            n = (c[0] + d[0], c[1] + d[1], c[2])
            if free(n) and can_pass(c, n, d):
                out.append((3 if d == last else 1, "h", n, d, None))
            for s in (1, -1):
                lo, hi = min(c[2], c[2] + s), min(c[2], c[2] + s) + 1
                n = (c[0] + (RAMP + 1) * d[0], c[1] + (RAMP + 1) * d[1], c[2] + s)
                cols = ramp_cols(c, d)
                if (
                    0 <= n[2] < Z
                    and free(n)
                    and can_pass(c, (*cols[0], c[2]), d)
                    and can_pass(n, (*cols[-1], n[2]), (-d[0], -d[1]))
                    and all(0 <= x < W and 0 <= y < H and not {(x, y, lo), (x, y, hi)} & (seen | blocked | room_at.keys())
                            for x, y in cols)
                ):
                    out.append((0.3, "r", n, d, (cols, lo, s)))
        return out

    e = next(r for r in rooms if r.kind == "entrance")
    start = e.cell_of(e.start)
    stack = enter(start)
    last = None
    while stack:
        c = stack[-1]
        opts = moves(c, last)
        if not opts:
            stack.pop()
            last = None
            continue
        _, kind, n, d, info = rng.choices(opts, [o[0] for o in opts])[0]
        if kind == "h":
            passages.add(frozenset((c, n)))
        else:
            cols, lo, s = info
            blocked.update((x, y, z) for x, y in cols for z in (lo, lo + 1))
            low, high = (c, n) if s > 0 else (n, c)
            ramps.append((low, d if s > 0 else (-d[0], -d[1]), high))
        cs = enter(n)
        rng.shuffle(cs)
        cs.remove(n)
        stack.extend(cs)
        stack.append(n)
        last = d
    portal = next(r for r in rooms if r.kind == "portal")
    if not portal.cells() <= seen:
        return None

    # braid: extra doors -> loops and crossings, plus fewer dead ends
    def deg(c):
        return sum(c in p for p in passages) + sum(c in (r[0], r[2]) for r in ramps)

    for c in sorted(seen):
        for dx, dy in DIRS[:2]:
            n = (c[0] + dx, c[1] + dy, c[2])
            if n in seen and frozenset((c, n)) not in passages and can_pass(c, n, (dx, dy)) and rng.random() < 0.03:
                passages.add(frozenset((c, n)))
    for c in sorted(seen):
        if deg(c) == 1 and c not in room_at and rng.random() < 0.3:
            ns = [(c[0] + dx, c[1] + dy, c[2]) for dx, dy in DIRS]
            ns = [n for n in ns if n in seen and frozenset((c, n)) not in passages and can_pass(c, n, (n[0] - c[0], n[1] - c[1]))]
            if ns:
                passages.add(frozenset((c, rng.choice(ns))))

    goal = portal.cell_of(portal.goal)
    path = solve(passages, ramps, start, goal)
    if path is None or steps(path) < MIN_PATH:
        return None
    loops = len(passages) + len(ramps) - len(seen) + 1
    return Maze(seed, rooms, seen, passages, ramps, start, goal, path, loops)


def adjacency(passages, ramps):
    adj = {}
    for p in passages:
        a, b = tuple(p)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for lo, _, hi in ramps:
        adj.setdefault(lo, []).append(hi)
        adj.setdefault(hi, []).append(lo)
    return adj


def steps(path):
    """Cell count walked; a ramp edge spans RAMP + 1 cells."""
    return sum(RAMP + 1 if a[2] != b[2] else 1 for a, b in zip(path, path[1:]))


def solve(passages, ramps, start, goal):
    adj = adjacency(passages, ramps)
    best, prev = {start: 0}, {start: None}
    q = [(0, start)]
    while q:
        d, c = heapq.heappop(q)
        if c == goal:
            out = []
            while c:
                out.append(c)
                c = prev[c]
            return out[::-1]
        for n in adj.get(c, []):
            nd = d + (RAMP + 1 if n[2] != c[2] else 1)
            if nd < best.get(n, 1e9):
                best[n], prev[n] = nd, c
                heapq.heappush(q, (nd, n))


def build(seed):
    for attempt in range(1000):
        if m := generate(seed, attempt):
            return m
    raise RuntimeError(f"no maze for seed {seed}")


# ---------- Webots output ----------


def cx(x):
    return (x - W / 2) * P


def cy(y):
    return (y - H / 2) * P


def axis_angle(phi, beta):
    c, s = math.cos(phi), math.sin(phi)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c, s = math.cos(beta), math.sin(beta)
    r = rz @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    ang = math.acos(max(-1, min(1, (np.trace(r) - 1) / 2)))
    ax = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * math.sin(ang))
    return *ax, ang


def rot_matrix(x, y, z, a):
    """Rodrigues: axis-angle (unit axis) -> 3x3 rotation matrix."""
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def num(v):
    return f"{v:.4f}".rstrip("0").rstrip(".")


# box faces as in SolidBox.proto: corner signs (x, y, z) and the size axis giving u, v (3 = 0); 4 corners per face
QUADS = [
    (-1, 1, 1, 0, 2), (1, 1, 1, 3, 2), (1, 1, -1, 3, 3), (-1, 1, -1, 0, 3),
    (-1, -1, -1, 3, 3), (1, -1, -1, 0, 3), (1, -1, 1, 0, 2), (-1, -1, 1, 3, 2),
    (-1, -1, 1, 3, 0), (1, -1, 1, 3, 3), (1, 1, 1, 1, 3), (-1, 1, 1, 1, 0),
    (1, -1, -1, 3, 0), (-1, -1, -1, 3, 3), (-1, 1, -1, 1, 3), (1, 1, -1, 1, 0),
    (-1, -1, -1, 1, 3), (-1, -1, 1, 1, 2), (-1, 1, 1, 3, 2), (-1, 1, -1, 3, 3),
    (1, -1, 1, 3, 2), (1, -1, -1, 3, 3), (1, 1, -1, 1, 3), (1, 1, 1, 1, 2),
]


class World:
    """All box visuals merged into one Solid with one Shape per texture (few draw calls); colliders see colliders()."""

    def __init__(self, rng):
        self.rng, self.app, self.mesh, self.bounds = rng, {}, {}, []

    def box(self, pos, size, tex, rot=None, solid=True, glow=False):
        if tex not in self.app:
            img = f'ImageTexture {{ url ["textures/{tex}.png"] filtering 0 }}'
            if tex == "bars":
                self.app[tex] = "PBRAppearance { baseColor 0.55 0.55 0.58 roughness 0.4 metalness 0.8 }"
            else:
                self.app[tex] = f"PBRAppearance {{ baseColorMap {img} " + (f"emissiveColorMap {img} " if glow else "") + "roughness 1 metalness 0 }"
        r = rot_matrix(*rot) if rot else np.eye(3)
        uv = (*size, 0)
        corners = [np.asarray(pos) + r @ (np.array(c) * size / 2) for *c, _, _ in QUADS]
        self.mesh.setdefault(tex, []).append((corners, [(uv[u] / BLOCK, uv[v] / BLOCK) for *_, u, v in QUADS]))  # one tile per MC block
        if solid:
            self.bounds.append((pos, rot, size))

    def wall_tex(self):
        return self.rng.choice(WALLS)

    def wbt(self):
        shapes = ""
        for tex, boxes in self.mesh.items():
            pts = "\n".join(" ".join(" ".join(map(num, p)) for p in c) for c, _ in boxes)
            uvs = "\n".join(" ".join(f"{num(u)} {num(v)}" for u, v in t) for _, t in boxes)
            idx = "\n".join(" ".join(f"{i} {i + 1} {i + 2} {i + 3} -1" for i in range(24 * b, 24 * b + 24, 4)) for b in range(len(boxes)))
            shapes += (
                f"    Shape {{\n      appearance {self.app[tex]}\n      geometry IndexedFaceSet {{\n"
                f"        coord Coordinate {{ point [\n{pts}\n        ] }}\n"
                f"        texCoord TextureCoordinate {{ point [\n{uvs}\n        ] }}\n"
                f"        coordIndex [\n{idx}\n        ]\n      }}\n    }}\n"
            )
        bounds = "".join(
            f'Solid {{ name "block {k}" translation {" ".join(map(num, p))} '
            + (f"rotation {r[0]:.5f} {r[1]:.5f} {r[2]:.5f} {r[3]:.5f} " if r else "")
            + f"boundingObject Box {{ size {' '.join(map(num, s))} }} }}\n"
            for k, (p, r, s) in enumerate(self.colliders())
        )
        return f'Solid {{\n  name "stronghold"\n  children [\n{shapes}  ]\n}}\n' + bounds

    def colliders(self):
        """Collision for the robot only: floor slabs and the first block layer, merged into rectangles; parts starting
        higher up never touch it and get none. Webots physics cost grows faster than the collider count (1408 single
        blocks: ~10x real time), and one Solid with a Group boundingObject was slower still, hence separate Solids."""
        grid, out = {}, []
        for p, r, sz in self.bounds:
            bot, top = p[2] - sz[2] / 2, p[2] + sz[2] / 2
            z = math.floor((bot + 0.03) / LV)
            if r:  # ramp parts keep their own box
                out.append((p, r, sz))
            elif top <= z * LV + 1e-3:
                grid.setdefault(("floor", z), set()).update(raster(p, sz, (cx(0), cy(0)), P))
            elif bot < z * LV + BLOCK / 2:
                grid.setdefault(("wall", z), set()).update(raster(p, sz, (cx(0) - T / 2, cy(0) - T / 2), BLOCK))
        for (kind, z), cells in grid.items():
            (ox, oy), u = ((cx(0), cy(0)), P) if kind == "floor" else ((cx(0) - T / 2, cy(0) - T / 2), BLOCK)
            z0, h = (z * LV - SLAB, SLAB) if kind == "floor" else (z * LV, WH)
            for i0, i1, j0, j1 in rects(cells):
                out.append(((ox + (i0 + i1) / 2 * u, oy + (j0 + j1) / 2 * u, z0 + h / 2), None, ((i1 - i0) * u, (j1 - j0) * u, h)))
        return out


def runs(vals, cap=None):
    """Consecutive int runs, optionally split to at most cap long."""
    out, vals = [], sorted(vals)
    for v in vals:
        if out and out[-1][-1] == v - 1 and (cap is None or len(out[-1]) < cap):
            out[-1].append(v)
        else:
            out.append([v])
    return out


def raster(p, size, origin, unit):
    """Grid cells (unit pitch from origin) under a box footprint; partly covered cells count."""
    lo = [math.floor((p[k] - size[k] / 2 - origin[k]) / unit + 1e-6) for k in (0, 1)]
    hi = [math.ceil((p[k] + size[k] / 2 - origin[k]) / unit - 1e-6) for k in (0, 1)]
    return {(i, j) for i in range(lo[0], hi[0]) for j in range(lo[1], hi[1])}


def rects(cells):
    """Greedy rectangle cover of grid cells: row runs, grown down while the next row has the same run.
    Returns (i0, i1, j0, j1), ends exclusive."""
    out, grow = [], {}
    for j in sorted({c[1] for c in cells}) + [None]:
        row = [] if j is None else [(r[0], r[-1] + 1) for r in runs(i for i, jj in cells if jj == j)]
        nxt = {r: (grow[r][0], j + 1) if r in grow and grow[r][1] == j else (j, j + 1) for r in row}
        out += [(*r, *v) for r, v in grow.items() if nxt.get(r, (None,))[0] != v[0]]
        grow = nxt
    return out


def block_xyz(r, what):
    i, j = getattr(r, what)
    return [round(cx(r.x) + T / 2 + (i + 0.5) * BLOCK, 4), round(cy(r.y) + T / 2 + (j + 0.5) * BLOCK, 4), r.z * LV]


# EndermiteModel BODY_TEXS / BODY_SIZES (w, h, d) from the 26.3 client jar, head first
ENDERMITE = (((0, 0), (4, 3, 2)), ((0, 5), (6, 4, 5)), ((0, 14), (3, 3, 1)), ((0, 18), (1, 2, 1)))


def endermite_skin(px=0.015, front=0.058):
    """Endermite look for the e-puck turret slot: one textured mesh, no bounding object, no mass, invisible to range rays.

    Rebuilds EndermiteModel.createBodyLayer with ModelPart.Cube's vertex order and UVs (64x32 texture). Model -z (head)
    becomes robot +x, model x robot y, model y (down, ground at 24) robot -z; 1 model pixel = px metres. The head front
    sits at robot x = front so the head box encloses the camera (0.03 0 0.028): faces seen from inside are culled, so
    the camera looks straight through it.
    """
    pts, uvs, idx = [], [], []
    z = -3.5  # segment centres chain along z like createBodyLayer; the head front ends at z = -4.5
    for i, ((u, v), (w, h, d)) in enumerate(ENDERMITE):
        x0, x1, y0, y1, z0, z1 = -w / 2, w / 2, 24 - h, 24, z - d / 2, z + d / 2
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = (  # (vertex order, u1, v1, u2, v2): DOWN, UP, WEST, NORTH, EAST, SOUTH
            ((5, 4, 0, 1), u + d, v, u + d + w, v + d),
            ((2, 3, 7, 6), u + d + w, v + d, u + d + 2 * w, v),
            ((0, 4, 7, 3), u, v + d, u + d, v + d + h),
            ((1, 0, 3, 2), u + d, v + d, u + d + w, v + d + h),
            ((5, 1, 2, 6), u + d + w, v + d, u + 2 * d + w, v + d + h),
            ((4, 5, 6, 7), u + 2 * d + w, v + d, u + 2 * d + 2 * w, v + d + h),
        )
        for order, u1, v1, u2, v2 in faces:
            idx.append(" ".join(str(len(pts) + k) for k in range(4)) + " -1")
            for k, (tu, tv) in zip(order, ((u2, v1), (u1, v1), (u1, v2), (u2, v2))):  # Polygon remap order
                mx, my, mz = c[k]
                pts.append(f"{front - (mz + 4.5) * px:.5f} {mx * px:.5f} {(24 - my) * px + 0.001:.5f}")
                uvs.append(f"{tu / 64:.5f} {1 - tv / 32:.5f}")
        if i < 3:
            z += (d + ENDERMITE[i + 1][1][2]) / 2
    return (
        "    Pose {\n      translation 0.0095 0 -0.0486\n      children [\n        Shape {\n"  # turret slot -> robot frame
        '          appearance PBRAppearance { baseColorMap ImageTexture { url ["textures/endermite.png"] filtering 0 } '
        "roughness 1 metalness 0 }\n"
        f"          geometry IndexedFaceSet {{\n            coord Coordinate {{ point [ {' '.join(pts)} ] }}\n"
        f"            texCoord TextureCoordinate {{ point [ {' '.join(uvs)} ] }}\n"
        f"            coordIndex [ {' '.join(idx)} ]\n          }}\n"
        "          castShadows FALSE\n          isPickable FALSE\n        }\n      ]\n    }\n"
    )


def write_world(m, path):
    w = World(random.Random(f"{m.seed}:tex"))
    room_at = {c: r for r in m.rooms for c in r.cells()}
    gaps = set()
    for lo, d, hi in m.ramps:
        r = ramp_cols(lo, d)
        gaps.add(frozenset((lo, (*r[0], lo[2]))))
        gaps.add(frozenset((hi, (*r[-1], hi[2]))))

    # floors
    for z in range(Z):
        rows = {}
        for c in m.seen:
            if c[2] == z:
                r = room_at.get(c)
                rows.setdefault((c[1], FLOOR.get(r.kind, "stone_bricks") if r else "stone_bricks"), []).append(c[0])
        for (y, tex), xs in rows.items():
            for run in runs(xs):
                w.box(((cx(run[0]) + cx(run[-1] + 1)) / 2, cy(y + 0.5), z * LV - SLAB / 2), (len(run) * P, P, SLAB), tex)

    # walls: boundary segments, then node ownership so no two solids overlap
    seg = {}  # (orient, line, idx, z) -> tall; H: row line / cell x, V: column line / cell y
    for c in m.seen:
        for dx, dy in DIRS:
            n = (c[0] + dx, c[1] + dy, c[2])
            if frozenset((c, n)) in m.passages or frozenset((c, n)) in gaps:
                continue
            tall = c in room_at and room_at[c].kind == "entrance"
            key = ("V", c[0] + (dx > 0), c[1], c[2]) if dx else ("H", c[1] + (dy > 0), c[0], c[2])
            seg[key] = seg.get(key, False) or tall

    def sides(x, y, z, o):  # wall segment of orientation o just before / after node (x, y)
        return ((o, y, x - 1, z) in seg, (o, y, x, z) in seg) if o == "H" else ((o, x, y - 1, z) in seg, (o, x, y, z) in seg)

    def owner(x, y, z):
        hb, ha = sides(x, y, z, "H")
        vb, va = sides(x, y, z, "V")
        if hb and ha:
            return "H"
        if vb and va:
            return "V"
        return "P" if hb or ha or vb or va else None

    def height(tall):
        return pieces.TALL * BLOCK if tall else WH

    for o, line, z in sorted({(k[0], k[1], k[3]) for k in seg}):
        cells = sorted(k[2] for k in seg if k[0] == o and k[1] == line and k[3] == z)
        for run in runs(cells):
            # V runs also split at nodes owned by H; chunks of <= 3 cells, tall change = chunk joint
            parts, cur = [], []
            for v in run:
                node = (v, line, z) if o == "H" else (line, v, z)
                if cur and (len(cur) == 3 or (o == "V" and owner(*node) == "H") or seg[(o, line, v, z)] != seg[(o, line, cur[-1], z)]):
                    parts.append((cur, o == "V" and owner(*node) == "H"))
                    cur = []
                cur.append(v)
            parts.append((cur, True))
            for k, (part, hard) in enumerate(parts):
                first = k == 0
                last = k == len(parts) - 1
                lo = (part[0] * P + T / 2) if first or parts[k - 1][1] else part[0] * P
                hi = ((part[-1] + 1) * P - T / 2) if last or hard else (part[-1] + 1) * P
                hh = height(any(seg[(o, line, v, z)] for v in part))
                zc = z * LV + hh / 2
                if o == "H":
                    w.box((cx(0) + (lo + hi) / 2, cy(line), zc), (hi - lo, T, hh), w.wall_tex())
                else:
                    w.box((cx(line), cy(0) + (lo + hi) / 2, zc), (T, hi - lo, hh), w.wall_tex())
    nodes = {(x, y, z) for (o, line, i, z) in seg for x, y in ([(i, line), (i + 1, line)] if o == "H" else [(line, i), (line, i + 1)])}
    for x, y, z in sorted(nodes):
        if owner(x, y, z) == "P":
            around = [seg.get(k, False) for k in [("H", y, x - 1, z), ("H", y, x, z), ("V", x, y - 1, z), ("V", x, y, z)] if k in seg]
            hh = height(any(around))
            w.box((cx(x), cy(y), z * LV + hh / 2), (T, T, hh), w.wall_tex())

    # ramps
    alpha = math.atan2(LV, RAMP * P)
    ls = math.hypot(RAMP * P, LV)
    for lo, d, _ in m.ramps:
        cols = ramp_cols(lo, d)
        mx, my = (cx(cols[0][0] + 0.5) + cx(cols[-1][0] + 0.5)) / 2, (cy(cols[0][1] + 0.5) + cy(cols[-1][1] + 0.5)) / 2
        rot = axis_angle(0 if d[0] else math.pi / 2, -alpha * (d[0] + d[1]))
        n3 = np.array([-math.sin(alpha) * d[0], -math.sin(alpha) * d[1], math.cos(alpha)])
        top = np.array([mx, my, lo[2] * LV + LV / 2])
        w.box(top - n3 * SLAB / 2, (ls + 0.02, P, SLAB), "cobblestone", rot)
        n = round(RAMP * P / T)  # vertical wall, one stepped column per block
        for s in (1, -1):
            for i in range(n):
                t = (i + 0.5) / n
                h = LV * t + WH
                u = (t - 0.5) * RAMP * P
                w.box((mx + d[0] * u - d[1] * s * P / 2, my + d[1] * u + d[0] * s * P / 2, lo[2] * LV + h / 2),
                      (T, T, h), w.wall_tex())

    # room pieces (block layouts from the game's stronghold pieces)
    for r in m.rooms:
        ox, oy, zl = cx(r.x) + T / 2, cy(r.y) + T / 2, r.z * LV
        for i, j, bw, bd, y0, h, tex, mode in r.boxes:
            pos = (ox + (i + bw / 2) * BLOCK, oy + (j + bd / 2) * BLOCK, zl + (y0 + h / 2) * BLOCK)
            size = (bw * BLOCK, bd * BLOCK, h * BLOCK)
            if mode == "portal":  # end portal surface sits 12/16 up inside the frame ring
                w.box((pos[0], pos[1], zl + (y0 + 0.75) * BLOCK), (size[0], size[1], 0.003), "end_portal", solid=False, glow=True)
            elif mode == "frame":  # 13/16 tall block: side body, top cap, filled eye
                body = 13 / 16 * BLOCK - 0.0015
                w.box((pos[0], pos[1], zl + y0 * BLOCK + body / 2), (BLOCK, BLOCK, body), "end_portal_frame_side")
                w.box((pos[0], pos[1], zl + y0 * BLOCK + body + 0.00075), (BLOCK, BLOCK, 0.0015), "end_portal_frame_top")
                w.box((pos[0], pos[1], zl + y0 * BLOCK + body + 0.0015 + 3 / 32 * BLOCK), (BLOCK / 2, BLOCK / 2, 3 / 16 * BLOCK), "end_portal_frame_eye")
            elif mode == "lava":
                w.box(pos, size, "lava", glow=True)
            else:
                w.box(pos, size, w.wall_tex() if tex == "stone_bricks" else tex)

    start = block_xyz(next(r for r in m.rooms if r.kind == "entrance"), "start")
    head = "".join(ln for ln in HEADER.read_text().split("WorldInfo")[0].splitlines(True) if "SolidBox" not in ln)
    head = head.replace("EXTERNPROTO", 'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/gctronic/e-puck/protos/E-puck.proto"\nEXTERNPROTO', 1)
    info = f"IR Maze Escape: Minecraft stronghold, seed {m.seed}, {steps(m.path)} steps start to portal, {m.loops} loops."
    head += (
        f'WorldInfo {{\n  info [\n    "{info}"\n  ]\n  title "Stronghold Escape"\n}}\n'
        f"Viewpoint {{\n  orientation -0.5773 0.5773 0.5773 2.0944\n  position 0 0 {W * P * 1.3:.1f}\n}}\n"
        # no shadows: stencil shadow volumes of every wall block made rendering crawl
        "TexturedBackground {\n}\nTexturedBackgroundLight {\n  castShadows FALSE\n}\n"
        f"Floor {{\n  translation 0 0 -0.3\n  size {W * P * 2:.0f} {W * P * 2:.0f}\n}}\n"
    )
    robot = (
        f"E-puck {{\n  translation {start[0]} {start[1]} 0\n  controller \"<none>\"\n  version \"2\"\n"
        f"  turretSlot [\n{endermite_skin()}  ]\n}}\n"
    )
    Path(path).write_text(head + w.wbt() + robot)
    Path(path).with_suffix(".json").write_text(json.dumps({
        "seed": m.seed,
        "start": start,
        "goal": block_xyz(next(r for r in m.rooms if r.kind == "portal"), "goal"),
        "steps": steps(m.path),
        "loops": m.loops,
    }, indent=2))


def extract_textures(out_dir, jar=JAR):
    from io import BytesIO

    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = "assets/minecraft/textures/"
    with zipfile.ZipFile(jar) as z:
        for t in TEXTURES:
            (out_dir / f"{t}.png").write_bytes(z.read(f"{tex}block/{t}.png"))
        (out_dir / "end_portal.png").write_bytes(z.read(f"{tex}entity/end_portal/end_portal.png"))
        Image.open(BytesIO(z.read(f"{tex}block/lava_still.png"))).crop((0, 0, 16, 16)).save(out_dir / "lava.png")
        # opaque palette PNG like the block textures: with alpha (or as 24-bit RGB) Webots draws the mesh without depth
        Image.open(BytesIO(z.read(f"{tex}entity/endermite/endermite.png"))).convert("RGB").convert("P", palette=Image.ADAPTIVE).save(out_dir / "endermite.png")


def preview(m, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, Z, figsize=(8 * Z, 8), squeeze=False)
    for z, ax in enumerate(axs[0]):
        for c in m.seen:
            if c[2] == z:
                ax.add_patch(plt.Rectangle((c[0], c[1]), 1, 1, color="#ccc"))
        for r in m.rooms:
            if r.z == z:
                ax.add_patch(plt.Rectangle((r.x, r.y), r.w, r.h, color={"portal": "purple", "entrance": "green"}.get(r.kind, "#9bd")))
        for p in m.passages:
            a, b = tuple(p)
            if a[2] == z:
                ax.plot([a[0] + .5, b[0] + .5], [a[1] + .5, b[1] + .5], "k", lw=2)
        for lo, d, hi in m.ramps:
            if lo[2] in (z, z - 1):
                ax.plot([lo[0] + .5, hi[0] + .5], [lo[1] + .5, hi[1] + .5], "r", lw=3)
        pts = [c for c in m.path if c[2] == z]
        ax.plot([c[0] + .5 for c in pts], [c[1] + .5 for c in pts], "y.", ms=4)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(f"floor {z}")
    fig.savefig(path, dpi=80)
