import csv
import math
from pathlib import Path

FS = 30.0
DURATION = 10.0
N = int(FS * DURATION)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
PENTAGON_CSV = CONFIG_DIR / "trajectory_pentagon_pose.csv"
HARMONIC_CSV = CONFIG_DIR / "trajectory_harmonic_pose.csv"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "theta"])
        writer.writerows(rows)


def regular_pentagon_vertices(radius=2.0, cx=0.0, cy=0.0):
    start = -math.pi / 2
    return [
        (
            cx + radius * math.cos(start + k * 2.0 * math.pi / 5.0),
            cy + radius * math.sin(start + k * 2.0 * math.pi / 5.0),
        )
        for k in range(5)
    ]


def generate_pentagon():
    v = regular_pentagon_vertices(radius=2.0)
    rows = []

    for i in range(N):
        u = (5.0 * i) / N
        e = int(u) % 5
        frac = u - int(u)

        x0, y0 = v[e]
        x1, y1 = v[(e + 1) % 5]

        x = (1.0 - frac) * x0 + frac * x1
        y = (1.0 - frac) * y0 + frac * y1
        theta = math.atan2(y1 - y0, x1 - x0)

        rows.append([x, y, theta])

    return rows


def generate_harmonic():
    # Trajectoire plane lisse basée sur sin(sin(.))
    # x suit une progression régulière, y oscille selon sin(sin(.))
    A = 1.0
    x_speed = 0.3  # unités/s
    w = 2.0 * math.pi / DURATION

    rows = []
    for i in range(N):
        t = i / FS
        x = x_speed * t
        y = A * math.sin(math.sin(w * t))

        dy_dt = A * math.cos(math.sin(w * t)) * math.cos(w * t) * w
        dx_dt = x_speed
        theta = math.atan2(dy_dt, dx_dt)

        rows.append([x, y, theta])

    return rows


if __name__ == "__main__":
    write_csv(PENTAGON_CSV, generate_pentagon())
    write_csv(HARMONIC_CSV, generate_harmonic())
    print(f"Created: {PENTAGON_CSV}")
    print(f"Created: {HARMONIC_CSV}")