"""Stronghold pieces, block layouts read from Minecraft 26.3 StrongholdPieces bytecode.

1 block = 0.1 m. Coordinates in the builders are the game's piece-local (x, y, z); the shell is x/z = 0 and
max, so the interior starts at 1. Boxes are stored in room-interior blocks: (i, j, w, d, y0, h, tex, mode)
with y0/h in blocks above the floor. Features above 3 blocks are clipped (floors stack 0.3 m apart).
A room spans nx x ny maze cells; its interior is 4n - 1 blocks (3 per cell + wall blocks between).
"""

TALL = 9  # entrance shaft height in blocks (top floor only, nothing above)
DIRS = {"S": (0, -1), "N": (0, 1), "W": (-1, 0), "E": (1, 0)}


class Piece:
    def __init__(self, nx, ny, px, pz, tall=False):
        self.nx, self.ny, self.px, self.pz, self.tall = nx, ny, px, pz, tall
        self.boxes, self.doors, self.bars, self.stone = [], [], [], set()
        self.start = self.goal = None  # (i, j) interior blocks
        self.I, self.J = 4 * nx - 1, 4 * ny - 1

    def box(self, x0, z0, x1, z1, y0, y1, tex, slab=False, mode=""):
        if not self.tall and mode != "beacon":
            y1 = min(y1, 3)
        if y0 > y1:
            return
        cells = {(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)}
        if tex == "iron_bars":  # expanded to thin rods in rotated()
            self.bars += [(x, z, y0, y1) for x, z in cells]
            return
        if tex == "stone_bricks":
            self.stone |= cells
        h = 0.5 if slab else y1 - y0 + 1
        self.boxes.append((x0 - 1 + self.px, z0 - 1 + self.pz, x1 - x0 + 1, z1 - z0 + 1, y0 - 1, h, tex, mode))

    def door(self, side, off):
        """3-wide opening at interior block offset along the S/N (x) or W/E (z) wall."""
        i, j, w, d = {"S": (off, -1, 3, 1), "N": (off, self.J, 3, 1), "W": (-1, off, 1, 3), "E": (self.I, off, 1, 3)}[side]
        self.doors.append((i, j, w, d))

    def rods(self):
        """Iron bars as thin rails and rods along the neighbouring bars/brick direction (see-through)."""
        solid = self.stone | {(x, z) for x, z, *_ in self.bars}
        out = []
        for x, z, y0, y1 in self.bars:
            h, i, j = y1 - y0 + 1, x - 1 + self.px, z - 1 + self.pz
            ax = (x - 1, z) in solid or (x + 1, z) in solid
            az = (x, z - 1) in solid or (x, z + 1) in solid
            for along_x in ([True] if ax and not az else [False] if az and not ax else [True, False]):
                t = 0.125
                for k in range(4):  # rods
                    o = 0.25 * k + 0.0625
                    out.append((i + o, j + 0.4375, t, t, y0 - 1, h, "bars", "") if along_x else (i + 0.4375, j + o, t, t, y0 - 1, h, "bars", ""))
                for yy, rh in ((y0 - 1, 0.4), (y1 - t, t)):  # rails; the low one is robot-sensor height
                    out.append((i, j + 0.4375, 1, t, yy, rh, "bars", "") if along_x else (i + 0.4375, j, t, 1, yy, rh, "bars", ""))
        return out

    def rotated(self, r):
        """Return (boxes, doors, start, goal, nx, ny) turned r quarter turns."""
        boxes, doors, I, J = self.boxes + self.rods(), self.doors, self.I, self.J
        pts = [p for p in (self.start, self.goal) if p]
        for _ in range(r % 4):
            rot = lambda i, j, w, d: (J - (j + d), i, d, w)
            boxes = [(*rot(*b[:4]), *b[4:]) for b in boxes]
            doors = [rot(*b) for b in doors]
            pts = [rot(p[0], p[1], 1, 1)[:2] for p in pts]
            I, J = J, I
        it = iter(pts)
        start = next(it) if self.start else None
        goal = next(it) if self.goal else None
        cells = []
        for i, j, w, d in doors:
            if j < 0:
                cells.append(((i + 1) // 4, 0, "S"))
            elif j >= J:
                cells.append(((i + 1) // 4, (J + 1) // 4 - 1, "N"))
            elif i < 0:
                cells.append((0, (j + 1) // 4, "W"))
            else:
                cells.append(((I + 1) // 4 - 1, (j + 1) // 4, "E"))
        return boxes, [(a, b, DIRS[s]) for a, b, s in cells], start, goal, (I + 1) // 4, (J + 1) // 4


def entrance(rng):
    """StairsDown: 5x11x5 shaft, spiral staircase of stone bricks and slabs (start piece)."""
    p = Piece(1, 1, 0, 0, tall=True)
    p.door("N", 0)
    for x, y, z0, z1 in [(2, 6, 1, 1), (1, 5, 1, 2), (1, 4, 3, 3), (2, 4, 3, 3), (3, 3, 2, 3), (2, 2, 1, 1), (3, 2, 1, 1), (1, 1, 1, 2)]:
        p.box(x, z0, x, z1, y, y, "stone_bricks")
    for x, y, z in [(1, 6, 1), (1, 5, 3), (3, 4, 3), (3, 3, 1), (1, 2, 1), (1, 1, 3)]:
        p.box(x, z, x, z, y, y, "stone_bricks", slab=True)
    p.start = (1, 1)
    return p


def crossing(rng):
    """RoomCrossing 11x7x11, three variants, exits on all four sides."""
    p = Piece(3, 3, 1, 1)
    for s in "SNWE":
        p.door(s, 4)
    v = rng.randrange(3)
    if v == 0:  # pillar with slab ring
        p.box(5, 5, 5, 5, 1, 3, "stone_bricks")
        for x in (4, 5, 6):
            for z in (4, 5, 6):
                if (x, z) != (5, 5):
                    p.box(x, z, x, z, 1, 1, "stone_bricks", slab=True)
    elif v == 1:  # fountain ring
        p.box(3, 3, 7, 3, 1, 1, "stone_bricks")
        p.box(3, 7, 7, 7, 1, 1, "stone_bricks")
        p.box(3, 4, 3, 6, 1, 1, "stone_bricks")
        p.box(7, 4, 7, 6, 1, 1, "stone_bricks")
        p.box(5, 5, 5, 5, 1, 3, "stone_bricks")
    else:  # cobblestone frame at y=3, cross, planks balcony
        for a in [(1, 1, 9, 1), (1, 9, 9, 9), (1, 2, 1, 8), (9, 2, 9, 8)]:
            p.box(*a, 3, 3, "cobblestone")
        for x, z in [(4, 4), (6, 4), (4, 6), (6, 6)]:
            p.box(x, z, x, z, 1, 3, "cobblestone")
        p.box(5, 4, 5, 4, 1, 1, "cobblestone")
        p.box(5, 6, 5, 6, 1, 1, "cobblestone")
        p.box(4, 5, 4, 5, 1, 1, "cobblestone")
        p.box(6, 5, 6, 5, 1, 1, "cobblestone")
        p.box(2, 2, 3, 8, 3, 3, "oak_planks")
        for z0, z1 in ((2, 3), (7, 8)):
            p.box(4, z0, 8, z1, 3, 3, "oak_planks")
    return p


def prison(rng):
    """PrisonHall 9x5x11: hall x1-3, cell block behind iron bars."""
    p = Piece(2, 3, 0, 1)
    p.door("S", 0)
    p.door("N", 0)
    for z in (1, 3, 7, 9):
        p.box(4, z, 4, z, 1, 3, "stone_bricks")
    for x, z in [(4, 4), (4, 5), (4, 6), (5, 5), (6, 5), (7, 5), (4, 2), (4, 8)]:
        p.box(x, z, x, z, 1, 3, "iron_bars")
    return p


def library(rng):
    """Library 14x6x15, dead end: shelves along both walls and five free-standing rows."""
    p = Piece(4, 4, 1, 1)
    p.door("S", 4)
    for z in range(1, 14):
        t = "oak_planks" if (z - 1) % 4 == 0 else "bookshelf"
        p.box(1, z, 1, z, 1, 4, t)
        p.box(12, z, 12, z, 1, 4, t)
    for z in (3, 5, 7, 9, 11):
        for x0, x1 in ((3, 4), (6, 7), (9, 10)):
            p.box(x0, z, x1, z, 1, 3, "bookshelf")
    return p


def portal(rng):
    """PortalRoom 11x8x16, dead end: stair pyramid up to the frame ring, lava pool, side ledges."""
    p = Piece(3, 4, 1, 0)
    p.door("S", 4)
    for y, z0 in ((1, 4), (2, 5), (3, 6)):
        p.box(4, z0, 6, 7, y, y, "stone_bricks")
    for z in (8, 12):
        p.box(3, z, 7, z, 1, 2, "stone_bricks")
    for x in (3, 7):
        p.box(x, 9, x, 11, 1, 2, "stone_bricks")
    p.box(4, 9, 6, 11, 1, 2, "lava", mode="lava")
    for x, z in [(4, 8), (5, 8), (6, 8), (4, 12), (5, 12), (6, 12), (3, 9), (3, 10), (3, 11), (7, 9), (7, 10), (7, 11)]:
        p.box(x, z, x, z, 3, 3, "frame", mode="frame")
    p.box(5, 10, 5, 10, 4, 9, "beacon", mode="beacon")  # glow column above the portal (the robot's camera target)
    p.box(4, 9, 6, 11, 3, 3, "portal", slab=True, mode="portal")  # all 12 eyes set: portal active
    for stone, lava, side in [((2, 1, 2, 4), 1, 1), ((8, 1, 8, 4), 9, 9)]:
        p.box(*stone, 1, 1, "stone_bricks")
        p.box(1 if side == 1 else 9, 4, 1 if side == 1 else 9, 4, 1, 1, "stone_bricks")
        p.box(lava, 1, lava, 3, 1, 1, "lava", mode="lava")
    p.goal = (5, 2)  # foot of the staircase
    return p


def chest(rng):
    """ChestCorridor 5x5x7: raised stone-brick run on one side with a chest."""
    p = Piece(1, 2, 0, 1)
    p.door("S", 0)
    p.door("N", 0)
    p.box(3, 2, 3, 4, 1, 1, "stone_bricks")
    p.box(3, 3, 3, 3, 2, 2, "oak_planks")
    return p


PIECES = {
    "entrance": entrance,
    "crossing": crossing,
    "prison": prison,
    "library": library,
    "portal": portal,
    "chest": chest,
}


def make(kind, rng, rot):
    p = PIECES[kind](rng)
    return p.rotated(rot)
