import math

from ir import stronghold as s


def test_deterministic_and_solvable():
    for seed in range(8):
        m = s.build(seed)
        assert m.path[0] == m.start and m.path[-1] == m.goal
        assert s.steps(m.path) >= s.MIN_PATH and m.loops >= 1
        assert {c[2] for c in m.path} == {0}
        assert s.build(seed).path == m.path


def test_ramp_slope():
    assert math.degrees(math.atan2(s.LV, s.RAMP * s.P)) < 20


def test_world_stable(tmp_path):
    out = [tmp_path / f"{i}.wbt" for i in range(2)]
    for o in out:
        s.write_world(s.build(3), o)
    assert out[0].read_text() == out[1].read_text()


def test_pieces_fit_and_doors_align():
    import random

    from ir import pieces

    for kind in pieces.PIECES:
        for rot in range(4):
            boxes, doors, start, goal, nx, ny = pieces.make(kind, random.Random(rot), rot)
            for i, j, w, d, *_ in boxes:
                assert 0 <= i and i + w <= 4 * nx - 1 and 0 <= j and j + d <= 4 * ny - 1, (kind, rot)
            for ci, cj, _ in doors:
                assert 0 <= ci < nx and 0 <= cj < ny, (kind, rot)
            for pt in (start, goal):
                assert pt is None or (0 <= pt[0] < 4 * nx - 1 and 0 <= pt[1] < 4 * ny - 1)
