#!/usr/bin/env python3
"""
Generate a circular trajectory CSV for the cRAMPC path_generator.

Columns: delta_theta [rad], speed_x [m/s]
  - delta_theta: per-timestep heading change (omega * dt)
  - speed_x:     forward body-frame velocity

Usage (standalone):
  python3 generate_circle_traj.py [--radius R] [--vmax V] [--laps N] [--freq F]
  [--ramp_steps K] [--output PATH]

Usage (ROS2):
  ros2 run cRAMPC generate_circle_traj
"""
import argparse
import csv
import os

import numpy as np


DEFAULT_RADIUS = 1.0          # circle radius [m]
DEFAULT_V_MAX = 0.46          # max forward velocity [m/s]
DEFAULT_LAPS = 2              # number of full laps
DEFAULT_FREQ = 30             # sampling frequency [Hz]
DEFAULT_RAMP_STEPS = 30       # cosine ramp-up duration [steps]
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'config', 'trajectory_circle.csv'
)


def cosine_ramp(i: int, n: int, v_max: float) -> float:
    """Return smooth cosine-ramp velocity at step i over n total ramp steps."""
    return v_max * 0.5 * (1.0 - np.cos(np.pi * i / n))


def generate(
    radius: float = DEFAULT_RADIUS,
    v_max: float = DEFAULT_V_MAX,
    laps: int = DEFAULT_LAPS,
    freq: float = DEFAULT_FREQ,
    ramp_steps: int = DEFAULT_RAMP_STEPS,
    output: str = DEFAULT_OUTPUT,
) -> list:
    """Generate the trajectory and write it to *output*. Returns list of rows."""
    dt = 1.0 / freq
    omega_max = v_max / radius       # [rad/s] for full-speed circle
    dtheta_max = omega_max * dt      # [rad]   per step at full speed

    rows = []

    # --- Ramp-up phase (cosine profile: v goes from 0 → v_max) ---
    angle_ramp = 0.0
    for i in range(ramp_steps):
        v = cosine_ramp(i, ramp_steps, v_max)
        dtheta = (v / radius) * dt
        angle_ramp += dtheta
        rows.append((dtheta, v))

    # --- Constant-velocity phase (fill remaining angle for requested laps) ---
    total_angle = laps * 2.0 * np.pi
    angle_remaining = total_angle - angle_ramp
    n_const = int(np.ceil(angle_remaining / dtheta_max))
    for _ in range(n_const):
        rows.append((dtheta_max, v_max))

    # Write CSV
    output = os.path.normpath(output)
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delta_theta', 'speed_x'])
        for dtheta, v in rows:
            writer.writerow([dtheta, v])

    print(
        f'Circle trajectory written to {output}\n'
        f'  radius={radius} m, v_max={v_max} m/s, laps={laps}, '
        f'freq={freq} Hz, ramp={ramp_steps} steps\n'
        f'  total steps={len(rows)}  '
        f'(ramp={ramp_steps}, constant={n_const})\n'
        f'  delta_theta_const={dtheta_max:.6f} rad, speed_x_const={v_max} m/s'
    )
    return rows


def main(args=None):
    """Entry point for standalone or ROS2 execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS,
                        help='Circle radius [m]')
    parser.add_argument('--vmax', type=float, default=DEFAULT_V_MAX,
                        help='Maximum forward velocity [m/s]')
    parser.add_argument('--laps', type=int, default=DEFAULT_LAPS,
                        help='Number of full laps')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ,
                        help='Sampling frequency [Hz]')
    parser.add_argument('--ramp_steps', type=int, default=DEFAULT_RAMP_STEPS,
                        help='Number of cosine ramp-up steps')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output CSV file path')
    parsed = parser.parse_args(args)

    generate(
        radius=parsed.radius,
        v_max=parsed.vmax,
        laps=parsed.laps,
        freq=parsed.freq,
        ramp_steps=parsed.ramp_steps,
        output=parsed.output,
    )


if __name__ == '__main__':
    main()
