import argparse
from pathlib import Path

from ir import stronghold


def main():
    ap = argparse.ArgumentParser(description="Generate a seeded Minecraft stronghold maze world for Webots")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("worlds/stronghold.wbt"))
    ap.add_argument("--jar", type=Path, default=stronghold.JAR, help="Minecraft client jar (textures)")
    ap.add_argument("--preview", action="store_true", help="also write <out>.png floor plans")
    a = ap.parse_args()
    m = stronghold.build(a.seed)
    stronghold.extract_textures(a.out.parent / "textures", a.jar)
    stronghold.write_world(m, a.out)
    if a.preview:
        stronghold.preview(m, a.out.with_suffix(".png"))
    print(f"{a.out}: seed {a.seed}, {stronghold.steps(m.path)} steps, {m.loops} loops, {len(m.ramps)} ramps")
