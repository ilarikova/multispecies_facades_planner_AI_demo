"""
Facade of bespoke tiles for additive manufacturing.

One tile per grid cell of the wall's own raster, as stored in the building
export. Tile depth follows the incident radiation at that position, so the
facade carries the solar analysis as relief; the corners of a tile are
interpolated from the surrounding grid points, which keeps the surface
continuous across tiles rather than stepped.

Tiles that cannot fit the full raster - at a wall edge, at a building corner,
or beside an opening - are cut to the area that remains. A tile holding a nest
box is deeper, takes the colour of its placement and carries the entrance,
dimensioned from the species sheet: at the top for birds, at the bottom for the
bats, which land from below.

The arrays returned are ready for a single plotly Mesh3d per group; nothing
here draws anything, and nothing here imports streamlit.
"""
from __future__ import annotations

import re

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# geometry of the system
TILE_GAP_M = 0.015           # joint between neighbouring tiles
DEPTH_MIN_M = 0.05           # shallowest tile that still carries relief
DEPTH_MAX_M = 0.15           # tile depth at the warmest point of the wall
# The coolest tiles of a wall stay flat: relief starts where the radiation
# does, and the flat field gives the eye a datum to read the gradient against.
FLAT_FRACTION = 0.10
WINDOW_BUFFER_M = 0.10       # tiles keep this clear of windows and doors
MIN_PIECE_AREA_M2 = 0.004    # a cut-off smaller than this is not worth a tile
# A leftover strip narrower than this is absorbed into the last full tile
# rather than left as a row of its own, so the bottom row grows to about one
# and a half tiles instead of a sliver appearing under it.
MERGE_BELOW_M = 0.20
# Every tile stands this far off the wall, flat ones included - sitting in the
# wall plane they merged with the surface behind and stopped reading as tiles.
BASE_DEPTH_M = 0.04
NEST_DEPTH_FALLBACK_M = 0.15
OPENING_PROUD_M = 0.004      # the entrance sits this far in front of its tile

# Shading: a light mostly from above, with a little from the side so that the
# left and right facets of a tile separate as well. SHADE_FLOOR is how dark a
# facet turned fully away is allowed to get.
# In each wall's own frame: mostly from in front and above, a little from the
# side. Expressed this way the same relief shades the same on every facade.
LIGHT_NORMAL, LIGHT_UP, LIGHT_SIDE = 0.55, 0.78, 0.30
# Depth is read as contrast, not as brightness. A flat tile faces the light
# evenly and stays near LIGHT_BASE; the deeper a tile, the further its ridge
# and its flanks swing away from that level. Flat areas therefore look light
# and calm, and the relief gets stronger the warmer the wall is.
LIGHT_BASE = 0.88        # tone of a flat tile
NEUTRAL_LAMBERT = 0.55   # the lambert value a flat tile returns
SWING_MIN = 0.30         # contrast of the shallowest relief
SWING_MAX = 1.45         # ... and of the deepest
SHADE_FLOOR = 0.18
SKIP_FLOOR_FUNCTIONS = {"neighbor_building", "neigbor_building"}


# ─────────────────────────────────────────────────────────────────
# SPECIES SHEET — entrance and cavity
# ─────────────────────────────────────────────────────────────────

def _mm(value: float) -> float:
    return float(value) / 1000.0


def parse_opening(raw_size, raw_shape) -> dict:
    """
    The entrance of one species, from `nest_openening_size`/`nest_opening_shape`.

    The column is free text and comes in four shapes:
        "35"                            a round hole, 35 mm across
        "35 x 70", "40 * 20"            width x height in mm
        "32-34", "45-50"                a range; the larger value is taken
        "half_front"                    no hole: the upper half of the front is open
        "20 - 40 * whole nest box side" a slit of that height, full tile width
    """
    text = str(raw_size or "").strip().lower()
    shape = str(raw_shape or "").strip().lower()
    if not text or text == "nan":
        return {"kind": "none", "w": 0.0, "h": 0.0}

    if "half_front" in text or "half_front" in shape:
        return {"kind": "half_front", "w": 0.0, "h": 0.0}

    full_width = "whole nest box side" in text
    numbers = [float(n) for n in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))]
    if not numbers:
        return {"kind": "none", "w": 0.0, "h": 0.0}

    if full_width:
        # "20 - 40 * whole nest box side": the numbers describe the slit height
        return {"kind": "slit", "w": 0.0, "h": _mm(max(numbers))}

    if re.search(r"\d\s*[x*]\s*\d", text) and len(numbers) >= 2:
        return {"kind": "rect", "w": _mm(numbers[0]), "h": _mm(numbers[1])}

    # a single value or a range: a round hole of that diameter
    d = _mm(max(numbers))
    return {"kind": "round", "w": d, "h": d}


def parse_cavity_depth(raw_size) -> float:
    """Depth of the nest box from `nest_size`, i.e. its smallest dimension in mm."""
    text = str(raw_size or "").replace("×", "x").replace("*", "x").lower()
    numbers = [float(n) for n in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))]
    numbers = [n for n in numbers if n >= 10]        # ignore stray range digits
    if not numbers:
        return NEST_DEPTH_FALLBACK_M
    return max(_mm(min(numbers)), DEPTH_MAX_M + 0.01)


def load_species_boxes(xlsx_path: str) -> dict:
    """
    {species: {opening, cavity_depth_m, bottom_entrance}} from the species sheet.

    The same workbook the planner reads; these columns are not part of the
    model's feature set, so they are taken from the sheet directly.
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    out = {}
    for _, row in df.iterrows():
        name = str(row.get("specie_name_EN") or "").strip()
        if not name.endswith("_core"):
            continue
        species = name[: -len("_core")]
        taxon = str(row.get("taxa") or "").strip().lower()
        out[species] = {
            "opening": parse_opening(row.get("nest_openening_size"),
                                     row.get("nest_opening_shape")),
            "cavity_depth_m": parse_cavity_depth(row.get("nest_size")),
            # bats hang in a crevice and land from below, so their slit stays at
            # the bottom; every bird box has its entrance at the top
            "bottom_entrance": taxon == "bat",
        }
    return out


# ─────────────────────────────────────────────────────────────────
# WALL GEOMETRY
# ─────────────────────────────────────────────────────────────────

def building_footprint(walls_data: dict):
    """The built area in plan, from the roof meshes - used to tell outside from in."""
    polys = []
    for w in walls_data.values():
        if not isinstance(w, dict) or w.get("type") != "roof":
            continue
        mesh = w.get("mesh") or {}
        V = np.asarray(mesh.get("vertices") or [], dtype=float)
        if len(V) < 3:
            continue
        for face in mesh.get("faces") or []:
            if len(face) >= 3:
                p = Polygon([(V[i][0], V[i][1]) for i in face])
                if p.is_valid and p.area > 1e-6:
                    polys.append(p)
    if not polys:
        return None
    merged = unary_union(polys).buffer(0)
    return merged if not merged.is_empty else None


def _plane_frame(wall: dict, footprint=None):
    """Origin and unit axes of the wall plane, with the outward normal."""
    plane = wall.get("plane") or {}
    if not plane:
        return None
    o = np.asarray(plane["origin"], dtype=float)
    ux = np.asarray(plane["xaxis"], dtype=float)
    uy = np.asarray(plane["yaxis"], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    n = np.cross(ux, uy)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Some walls store a plane whose normal points into the building, and the
    # exported face normals do not agree with each other either. Stepping off
    # the wall and asking whether that point is still inside the footprint
    # settles it, and gets courtyard walls right as well.
    mesh = wall.get("mesh") or {}
    V = np.asarray(mesh.get("vertices") or [], dtype=float)
    if footprint is not None and len(V):
        centre = V.mean(axis=0)
        ahead = Point(*(centre + 0.6 * n)[:2])
        behind = Point(*(centre - 0.6 * n)[:2])
        if footprint.contains(ahead) and not footprint.contains(behind):
            n = -n
    else:
        fn = mesh.get("face_normals")
        if fn:
            mean = np.mean(np.asarray(fn, dtype=float), axis=0)
            if np.dot(mean, n) < 0:
                n = -n
    return o, ux, uy, n


def _grid_arrays(wall: dict):
    """Unique u and v of the wall raster, and the radiation value per cell."""
    grid = (wall.get("grid") or {}).values()
    uv, climate = [], []
    for p in grid:
        if p.get("uv") is None or p.get("climate") is None:
            continue
        uv.append(p["uv"])
        climate.append(float(p["climate"]))
    if len(uv) < 4:
        return None
    uv = np.asarray(uv, dtype=float)
    us = np.unique(np.round(uv[:, 0], 3))
    vs = np.unique(np.round(uv[:, 1], 3))
    field = np.full((len(vs), len(us)), np.nan)
    iu = np.searchsorted(us, np.round(uv[:, 0], 3))
    iv = np.searchsorted(vs, np.round(uv[:, 1], 3))
    field[iv, iu] = climate
    return us, vs, field


def _fill_gaps(field: np.ndarray) -> np.ndarray:
    """Openings leave holes in the raster; fill them with the wall's median."""
    out = field.copy()
    if np.isnan(out).all():
        return np.zeros_like(out)
    out[np.isnan(out)] = float(np.nanmedian(out))
    return out


def _sampler(us, vs, field):
    """Bilinear sampling of the radiation field at any (u, v) of the wall."""
    filled = _fill_gaps(field)
    iu_axis = np.arange(len(us), dtype=float)
    iv_axis = np.arange(len(vs), dtype=float)

    def sample(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        fu = np.interp(u, us, iu_axis)
        fv = np.interp(v, vs, iv_axis)
        u0 = np.clip(np.floor(fu).astype(int), 0, len(us) - 1)
        v0 = np.clip(np.floor(fv).astype(int), 0, len(vs) - 1)
        u1 = np.clip(u0 + 1, 0, len(us) - 1)
        v1 = np.clip(v0 + 1, 0, len(vs) - 1)
        du, dv = fu - u0, fv - v0
        return ((1 - du) * (1 - dv) * filled[v0, u0] + du * (1 - dv) * filled[v0, u1]
                + (1 - du) * dv * filled[v1, u0] + du * dv * filled[v1, u1])

    return sample


def wall_outline(wall: dict, snap_m: float = 1e-3):
    """
    The wall's true silhouette, taken from the edges its mesh uses once.

    `boundary_uv` is the exporter's simplified outline and runs up to 7 % wide
    on some walls, which would place tiles past the end of the wall.
    """
    mesh = wall.get("mesh") or {}
    frame = _plane_frame(wall)
    V = mesh.get("vertices")
    if not V or frame is None:
        bd = wall.get("boundary_uv") or []
        poly = Polygon([(float(p[0]), float(p[1])) for p in bd]) if len(bd) >= 3 else None
        return poly if poly is not None and poly.is_valid and not poly.is_empty else None

    o, ux, uy, _ = frame
    rel = np.asarray(V, dtype=float) - o
    uv = np.round(np.stack([rel @ ux, rel @ uy], axis=1) / snap_m).astype(np.int64)

    keys, points, remap = {}, [], np.empty(len(uv), dtype=np.int64)
    for n, key in enumerate(map(tuple, uv)):
        at = keys.get(key)
        if at is None:
            at = keys[key] = len(points)
            points.append((key[0] * snap_m, key[1] * snap_m))
        remap[n] = at

    counts = {}
    for face in mesh.get("faces") or []:
        if len(face) < 3:
            continue
        idx = [int(remap[i]) for i in face]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            if a == b:
                continue
            counts[(min(a, b), max(a, b))] = counts.get((min(a, b), max(a, b)), 0) + 1

    neighbours = {}
    for (a, b), n in counts.items():
        if n != 1:
            continue
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if not neighbours or any(len(x) != 2 for x in neighbours.values()):
        bd = wall.get("boundary_uv") or []
        poly = Polygon([(float(p[0]), float(p[1])) for p in bd]) if len(bd) >= 3 else None
        return poly if poly is not None and poly.is_valid and not poly.is_empty else None

    start = next(iter(neighbours))
    ring, prev, cur = [start], None, start
    while True:
        a, b = neighbours[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        ring.append(nxt)
        prev, cur = cur, nxt
        if len(ring) > len(neighbours):
            break
    poly = Polygon([points[i] for i in ring]).simplify(snap_m)
    return poly if poly.is_valid and not poly.is_empty else None


def _bands(lo: float, hi: float, step: float, from_high: bool) -> list:
    """
    Divide a wall into bands of `step`, starting at the edge the facade is set
    out from: the top for rows, the left for columns. Whatever is left over at
    the far end joins the last band if it is too narrow to stand alone.
    """
    span = hi - lo
    if span <= step + 1e-9:
        return [(lo, hi)]
    count = int(span // step)
    rest = span - count * step

    bands = []
    if from_high:
        edge = hi
        for _ in range(count):
            bands.append((edge - step, edge))
            edge -= step
    else:
        edge = lo
        for _ in range(count):
            bands.append((edge, edge + step))
            edge += step

    if rest > 1e-6:
        if rest < MERGE_BELOW_M and bands:
            first, last = bands[-1]
            bands[-1] = (first - rest, last) if from_high else (first, last + rest)
        else:
            bands.append((edge - rest, edge) if from_high else (edge, edge + rest))
    return bands


def _courses(lo: float, hi: float, step: float, datum: float) -> list:
    """
    Horizontal courses counted down from `datum`, clipped to one wall.

    The datum belongs to the building, not to the wall, so two walls meeting
    at a corner get their tiles at the same heights. Returned top-down as
    (low, high, index); the index carries on across walls, which is what keeps
    the ridges alternating in step. A course left too short at either end of
    the wall joins its neighbour rather than becoming a sliver.
    """
    first = int(np.floor((datum - hi) / step + 1e-9))
    bands, k = [], max(first, 0)
    while True:
        top = datum - k * step
        if top <= lo + 1e-9:
            break
        band_lo, band_hi = max(top - step, lo), min(top, hi)
        if band_hi - band_lo > 1e-6:
            bands.append([band_lo, band_hi, k])
        k += 1
    if len(bands) > 1 and bands[0][1] - bands[0][0] < MERGE_BELOW_M:
        bands[1][1] = bands[0][1]
        bands.pop(0)
    if len(bands) > 1 and bands[-1][1] - bands[-1][0] < MERGE_BELOW_M:
        bands[-2][0] = bands[-1][0]
        bands.pop()
    return bands


def _blocked_area(wall: dict):
    """Windows and doors, widened by the buffer tiles have to keep."""
    polys = []
    for group in ("windows", "doors"):
        for opening in (wall.get(group) or {}).values():
            uv = opening.get("hull_uv")
            if uv and len(uv) >= 3:
                p = Polygon([(float(a), float(b)) for a, b in uv])
                if p.is_valid and not p.is_empty:
                    polys.append(p.buffer(WINDOW_BUFFER_M, join_style=2))
    return unary_union(polys) if polys else None


# ─────────────────────────────────────────────────────────────────
# TRIANGULATION
# ─────────────────────────────────────────────────────────────────

def _ear_clip(ring):
    """Triangles of a simple polygon ring; never bridges a concave corner."""
    pts = [(float(a), float(b)) for a, b in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    n = len(pts)
    if n < 3:
        return pts, []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    order = list(range(n))
    if sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
           for i in range(n)) < 0:
        order.reverse()

    tris, guard = [], 0
    while len(order) > 2 and guard <= n * n:
        guard += 1
        for k in range(len(order)):
            i0, i1, i2 = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(cross(a, b, pts[m]) >= -1e-12 and cross(b, c, pts[m]) >= -1e-12
                   and cross(c, a, pts[m]) >= -1e-12
                   for m in order if m not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            order.pop(k)
            break
        else:
            break
    return pts, tris


class _Mesh:
    """
    Vertices and triangles of one draw group.

    Tiles arrive in the wall's own uv frame and are only turned into world
    coordinates when the wall is finished, all at once: sampling the radiation
    field per vertex costs more than the geometry itself.
    """

    def __init__(self):
        self.xyz: list = []
        self.f: list = []
        self.shade: list = []         # one shading value per triangle
        self._uv: list = []
        self._d: list = []
        self._w: list = []            # share of the tile's depth per vertex
        self._tris: list = []

    def add(self, ring, tris, depth=None):
        """`depth` None means: take it from the wall's radiation field."""
        base = len(self._uv)
        self._uv.extend(ring)
        self._d.extend([np.nan if depth is None else float(depth)] * len(ring))
        self._w.extend([1.0] * len(ring))
        self._tris.extend((base + a, base + b, base + c) for a, b, c in tris)

    def add_wedge(self, u0, v0, u1, v1, apex_depth=None, ridge_up: bool = True,
                  flip: bool = False):
        """
        One tile as a wedge: its ridge on the upper or the lower edge.

        Four corners on the wall plane and a ridge point raised over the middle
        of one edge. The big face slopes from the opposite edge up to the
        ridge, and its angle follows the tile's depth, which is what makes a
        gradient across the wall readable.
        """
        base = len(self._uv)
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]   # bl, br, tr, tl
        self._uv.extend(corners)
        self._d.extend([BASE_DEPTH_M] * 4)
        ridge_v = v1 if ridge_up else v0
        self._uv.append(((u0 + u1) / 2.0, ridge_v))
        self._d.append(np.nan if apex_depth is None else float(apex_depth))
        self._w.extend([1.0] * 5)
        bl, br, tr, tl = base, base + 1, base + 2, base + 3
        a = base + 4
        if ridge_up:
            faces = [(bl, br, a), (br, tr, a), (tl, bl, a)]
        else:
            faces = [(tr, tl, a), (tl, bl, a), (br, tr, a)]
        if flip:                      # wall whose plane normal points the other way
            faces = [(c, b, a_) for a_, b, c in faces]
        self._tris += faces

    def add_fan(self, ring, apex_uv, apex_depth=None, flip: bool = False):
        """
        One tile: its outline on the wall plane and a raised apex over it.

        Four facets per tile, each catching the light at its own angle, which
        is what makes the relief legible - a single face at the tile's depth
        does not, however deep it sits.
        """
        n = len(ring)
        if n < 3:
            return
        base = len(self._uv)
        self._uv.extend(ring)
        self._d.extend([BASE_DEPTH_M] * n)
        self._uv.append((float(apex_uv[0]), float(apex_uv[1])))
        self._d.append(np.nan if apex_depth is None else float(apex_depth))
        self._w.extend([1.0] * (n + 1))
        apex = base + n
        au, av = self._uv[apex]
        for i in range(n):
            a, b = base + i, base + (i + 1) % n
            (u0, v0), (u1, v1) = ring[i], ring[(i + 1) % n]
            # a ridge sitting on the tile's own edge makes the two facets along
            # that edge degenerate; their normals are noise, and they rendered
            # as dark flecks around every opening
            area = abs((u1 - u0) * (av - v0) - (v1 - v0) * (au - u0))
            if area < 1e-7:
                continue
            self._tris.append((b, a, apex) if flip else (a, b, apex))

    def add_slab(self, ring, v_far, v_ridge, depth=None, flip: bool = False):
        """
        A cut tile as one sloping face, lying on the plane of its own row.

        A fan from a ridge point folds over itself where a window leaves an
        L-shaped piece, and the folded facets read as facing into the wall.
        Ear clipping copes with any shape, and taking each corner's height
        from the row's slope keeps the cut piece in line with the whole tiles
        on either side of it.
        """
        pts, tris = _ear_clip(ring)
        if not tris:
            return
        span = (v_ridge - v_far) or 1.0
        base = len(self._uv)
        self._uv.extend(pts)
        self._d.extend([np.nan if depth is None else float(depth)] * len(pts))
        self._w.extend(min(1.0, max(0.0, (v - v_far) / span)) for _, v in pts)
        faces = [(base + a, base + b, base + c) for a, b, c in tris]
        if flip:
            faces = [(c, b, a) for a, b, c in faces]
        self._tris += faces

    def flush(self, frame, depth_field):
        if not self._tris:
            self._uv, self._d, self._w, self._tris = [], [], [], []
            return
        o, ux, uy, normal = frame
        uv = np.asarray(self._uv, dtype=float)
        d = np.asarray(self._d, dtype=float)
        sampled = np.isnan(d)
        if sampled.any():
            d[sampled] = depth_field(uv[sampled, 0], uv[sampled, 1])
        # a slab's corners take only their share of the tile's depth
        d = BASE_DEPTH_M + np.asarray(self._w, dtype=float) * (d - BASE_DEPTH_M)
        xyz = o + uv[:, :1] * ux + uv[:, 1:2] * uy + d[:, None] * normal
        span = (DEPTH_MAX_M - BASE_DEPTH_M) or 1.0
        frac = np.clip((d - BASE_DEPTH_M) / span, 0.0, 1.0)
        base = len(self.xyz)
        self.xyz.extend(xyz.tolist())
        faces = [(base + a, base + b, base + c) for a, b, c in self._tris]
        self.f.extend(faces)

        # the light is built from this wall's own axes, so every facade reads
        # the same regardless of which way it faces
        up = uy if uy[2] >= 0 else -uy
        light = LIGHT_NORMAL * normal + LIGHT_UP * up + LIGHT_SIDE * ux
        light = light / (np.linalg.norm(light) + 1e-12)
        tri = np.asarray(self._tris, dtype=np.int64)
        a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
        normals = np.cross(b - a, c - a)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        # whichever way a facet was wound, it is shaded on the side that faces
        # out of the wall: a facet read as facing backwards goes black
        normals[normals @ normal < 0.0] *= -1.0
        lambert = np.clip(normals @ light, 0.0, 1.0)
        depth_term = frac[tri].max(axis=1)
        swing = SWING_MIN + (SWING_MAX - SWING_MIN) * depth_term
        shade = LIGHT_BASE + swing * (lambert - NEUTRAL_LAMBERT)
        self.shade.extend(np.clip(shade, SHADE_FLOOR, 1.0).tolist())

        self._uv, self._d, self._w, self._tris = [], [], [], []

    def arrays(self):
        """Vertices, triangles, and one shading value per triangle."""
        if not self.f:
            return None
        return (np.round(np.asarray(self.xyz, dtype=float), 3),
                np.asarray(self.f, dtype=np.int32),
                np.round(np.asarray(self.shade, dtype=float), 3))


# ─────────────────────────────────────────────────────────────────
# THE FACADE
# ─────────────────────────────────────────────────────────────────

def build_tiles(walls_data: dict, nests: list, species_boxes: dict) -> dict:
    """
    Tiles for every wall of the building that carries a raster.

    `nests` are the placements to integrate, each a dict with `wall_id`, `uv`,
    `species` and `color`. Walls facing a neighbouring building are skipped;
    so are roofs, which carry no raster.

    Returns {"tiles": (V, F), "nests": {colour: (V, F)}, "openings": (V, F),
             "counts": {...}} with every array ready for one Mesh3d.
    """
    climate = [float(p["climate"])
               for w in walls_data.values() if isinstance(w, dict)
               for p in (w.get("grid") or {}).values() if p.get("climate") is not None]
    if not climate:
        return {"tiles": None, "nests": {}, "openings": None, "counts": {}}
    lo, hi = float(min(climate)), float(max(climate))
    span = (hi - lo) or 1.0

    by_wall: dict = {}
    for nest in nests or []:
        by_wall.setdefault(nest["wall_id"], []).append(nest)

    footprint = building_footprint(walls_data)

    # Everything a tiled wall needs, gathered first: the courses are set out
    # from one datum for the whole building, so the highest wall top and a
    # single row spacing have to be known before any wall is tiled.
    prepared, spacings, tops = [], [], []
    for wall_id, wall in walls_data.items():
        if not isinstance(wall, dict) or wall.get("type") == "roof":
            continue
        if str(wall.get("floor_function") or "").strip().lower() in SKIP_FLOOR_FUNCTIONS:
            continue
        grid = _grid_arrays(wall)
        frame = _plane_frame(wall, footprint)
        outline = wall_outline(wall)
        if grid is None or frame is None or outline is None:
            continue
        prepared.append((wall_id, wall, grid, frame, outline))
        vs = grid[1]
        if len(vs) > 1:
            spacings.append(float(np.median(np.diff(vs))))
        o, _ux, uy, _n = frame
        v_lo, v_hi = outline.bounds[1], outline.bounds[3]
        tops.append(max(o[2] + v_lo * uy[2], o[2] + v_hi * uy[2]))
    if not prepared:
        return {"tiles": None, "nests": {}, "openings": None, "counts": counts}
    row_step = float(np.median(spacings)) if spacings else 0.3
    z_datum = max(tops)

    plain = _Mesh()
    openings = _Mesh()
    nest_meshes: dict = {}
    counts = {"tiles": 0, "partial": 0, "nests": 0, "walls": 0}

    for wall_id, wall, grid, frame, outline in prepared:
        us, vs, field = grid
        sample = _sampler(us, vs, field)
        o, ux, uy, normal = frame
        # a ring given anticlockwise in uv only winds outward when the normal
        # follows ux x uy; where it was flipped, the ring has to be reversed
        flipped = float(np.dot(np.cross(ux, uy), normal)) < 0
        if flipped:
            def _wind(ring):
                return list(reversed(ring))
        else:
            def _wind(ring):
                return ring
        du = float(np.median(np.diff(us))) if len(us) > 1 else 0.3

        blocked = _blocked_area(wall)
        usable = outline.difference(blocked) if blocked is not None else outline
        if usable.is_empty:
            continue

        wall_nests = by_wall.get(wall_id, [])
        counts["walls"] += 1

        # Courses come from the building datum, columns from this wall's own
        # left edge. Half the walls store v increasing downwards, so a row
        # taken from the larger v would start at the ground on those walls and
        # the courses would not meet at the corners.
        o_u0, o_v0, o_u1, o_v1 = outline.bounds
        rise = float(uy[2]) if abs(float(uy[2])) > 1e-6 else 1.0
        v_up = rise > 0.0
        z_a, z_b = o[2] + o_v0 * rise, o[2] + o_v1 * rise
        v_bands = []
        for band_lo, band_hi, k in _courses(min(z_a, z_b), max(z_a, z_b),
                                            row_step, z_datum):
            v_x, v_y = (band_lo - o[2]) / rise, (band_hi - o[2]) / rise
            # the ridge alternates by course index, counted from the datum, so
            # the zigzag carries across a corner instead of restarting
            v_bands.append((min(v_x, v_y), max(v_x, v_y), (k % 2 == 0) == v_up))
        if not v_bands:
            continue
        u_bands = _bands(o_u0, o_u1, du, from_high=False)
        gap = TILE_GAP_M / 2.0
        cols = np.array(u_bands, dtype=float)
        rows = np.array([(a, b) for a, b, _ in v_bands], dtype=float)
        ridge_up = np.repeat(np.array([r for _, _, r in v_bands], dtype=bool),
                             len(cols))
        n_cols = len(cols)
        tile_u0 = np.tile(cols[:, 0], len(rows)) + gap
        tile_u1 = np.tile(cols[:, 1], len(rows)) - gap
        tile_v0 = np.repeat(rows[:, 0], n_cols) + gap
        tile_v1 = np.repeat(rows[:, 1], n_cols) - gap
        grid_u = (tile_u0 + tile_u1) / 2.0
        grid_v = (tile_v0 + tile_v1) / 2.0
        rects = shapely.box(tile_u0, tile_v0, tile_u1, tile_v1)
        whole = shapely.contains(usable, rects)
        cut = ~whole & shapely.intersects(usable, rects)
        pieces = np.empty(len(rects), dtype=object)
        if cut.any():
            pieces[cut] = shapely.intersection(rects[cut], usable)

        # Each wall is remapped over its own range, as the Grasshopper
        # definition did. Normalising across the whole building leaves almost
        # no relief on a shaded wall: on building 4868, wall 07 spans 347 to
        # 365, which is 3 mm of depth over the entire facade.
        wall_lo = float(np.nanmin(field))
        wall_span = (float(np.nanmax(field)) - wall_lo) or 1.0

        def depth_field(u, v, _lo=wall_lo, _span=wall_span):
            t = (sample(u, v) - _lo) / _span
            above = np.clip((t - FLAT_FRACTION) / (1.0 - FLAT_FRACTION), 0.0, 1.0)
            depth = DEPTH_MIN_M + (DEPTH_MAX_M - DEPTH_MIN_M) * above
            return np.where(t < FLAT_FRACTION, BASE_DEPTH_M, depth)

        for n in range(len(rects)):
            if whole[n]:
                parts = [None]                    # the full rectangle
            elif cut[n]:
                piece = pieces[n]
                parts = (list(piece.geoms) if piece.geom_type.startswith("Multi")
                         else [piece])
            else:
                continue

            u_centre, v_centre = float(grid_u[n]), float(grid_v[n])
            lo_u, hi_u = float(tile_u0[n]), float(tile_u1[n])
            lo_v, hi_v = float(tile_v0[n]), float(tile_v1[n])
            up = bool(ridge_up[n])
            nest = next((x for x in wall_nests
                         if lo_u <= x["uv"][0] <= hi_u and lo_v <= x["uv"][1] <= hi_v), None)

            for part in parts:
                if part is None:                  # a whole tile
                    ring = [(lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v)]
                    apex = (u_centre, v_centre)
                    bounds = (lo_u, lo_v, hi_u, hi_v)
                    whole_tile = True
                else:
                    if part.geom_type != "Polygon" or part.area < MIN_PIECE_AREA_M2:
                        continue
                    ring = list(part.exterior.coords)[:-1]
                    if len(ring) < 3:
                        continue
                    # a cut tile keeps its apex over the piece that is left
                    bounds = part.bounds
                    # the ridge sits on the same edge as the uncut neighbours,
                    # pulled onto the piece if the cut took that edge away
                    apex = _ridge_point(part, (bounds[0] + bounds[2]) / 2.0,
                                        hi_v if up else lo_v)
                    whole_tile = False
                    counts["partial"] += 1
                counts["tiles"] += 1

                if nest is None:
                    if whole_tile:
                        # courses alternate, so the ridges zigzag over the wall
                        plain.add_wedge(lo_u, lo_v, hi_u, hi_v, ridge_up=up,
                                        flip=flipped)
                    else:
                        # a cut piece stays on the plane of its own course, so
                        # the slope runs on past a window or a corner
                        plain.add_slab(ring, lo_v if up else hi_v,
                                       hi_v if up else lo_v, flip=flipped)
                    continue

                spec = species_boxes.get(nest["species"], {})
                depth = float(spec.get("cavity_depth_m", NEST_DEPTH_FALLBACK_M))
                nest_meshes.setdefault(nest.get("color", "#F4A623"), _Mesh()) \
                    .add_fan(ring, apex, depth, flip=flipped)
                counts["nests"] += 1

                _add_opening(openings, bounds, spec, depth, v_up)

        for mesh in (plain, openings, *nest_meshes.values()):
            mesh.flush(frame, depth_field)

    return {
        "tiles": plain.arrays(),
        "nests": {c: m.arrays() for c, m in nest_meshes.items() if m.arrays() is not None},
        "openings": openings.arrays(),
        "counts": counts,
    }


def _ridge_point(piece, u: float, v: float):
    """The ridge of a cut tile: that point, pulled onto the piece if it fell outside."""
    from shapely.geometry import Point as _P
    from shapely.ops import nearest_points

    p = _P(u, v)
    if piece.covers(p):
        return (u, v)
    on = nearest_points(piece, p)[0]
    return (on.x, on.y)


def _add_opening(mesh: _Mesh, bounds, spec: dict, depth: float, v_up: bool = True):
    """The entrance on the outer face of a nest tile."""
    opening = spec.get("opening") or {}
    kind = opening.get("kind", "none")
    if kind == "none":
        return

    u0, v0, u1, v1 = bounds
    # on a wall whose v runs downwards the larger v is the foot of the tile,
    # so the side the species enters from has to be swapped back
    bottom = bool(spec.get("bottom_entrance")) == v_up
    width, height = u1 - u0, v1 - v0
    cu = (u0 + u1) / 2.0
    margin = min(0.02, height * 0.15)

    if kind == "half_front":
        w, h = width * 0.9, height * 0.45
    elif kind == "slit":
        w, h = width * 0.9, max(opening.get("h", 0.02), 0.015)
    else:
        w = max(opening.get("w", 0.035), 0.02)
        h = max(opening.get("h", w), 0.02)
        w, h = min(w, width * 0.8), min(h, height * 0.6)

    if bottom:
        vb = v0 + margin
        vt = min(vb + h, v1 - margin)
    else:
        vt = v1 - margin
        vb = max(vt - h, v0 + margin)
    ub, ut = cu - w / 2.0, cu + w / 2.0
    if ut <= ub or vt <= vb:
        return

    # The entrance is always a triangle; only its size comes from the species.
    # Its base sits on the edge the species enters from - the top of the tile for
    # the birds, the bottom for the bats - and the apex points into the tile.
    cu_mid = (ub + ut) / 2.0
    if bottom:
        ring = [(ub, vb), (ut, vb), (cu_mid, vt)]
    else:
        ring = [(ub, vt), (ut, vt), (cu_mid, vb)]
    mesh.add(ring, [(0, 1, 2)], depth + OPENING_PROUD_M)
