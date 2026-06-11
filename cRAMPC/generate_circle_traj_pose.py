#!/usr/bin/env python3
"""
Generate a circular trajectory CSV with absolute poses for the cRAMPC path_generator.

Columns: x [m], y [m], theta [rad]
  - x, y:   position in the world frame
  - theta:  heading angle (tangent to circle)

The robot starts at (radius, 0) heading in the +y direction (theta = pi/2)
and travels counter-clockwise. Velocity is ramped up with a cosine profile.

Usage (standalone):
  python3 generate_circle_traj_pose.py [--radius R] [--vmax V] [--laps N]
  [--freq F] [--ramp_steps K] [--output PATH]

Usage (ROS2):
  ros2 run cRAMPC generate_circle_traj_pose
"""
import argparse
import csv
import os

import numpy as np


DEFAULT_RADIUS     = 1.0    # circle radius [m]
DEFAULT_V_MAX      = 0.46   # max forward velocity [m/s]
DEFAULT_LAPS       = 1      # number of full laps
DEFAULT_FREQ       = 30     # sampling frequency [Hz]
DEFAULT_RAMP_STEPS = 600     # cosine ramp-up duration [steps]
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'config', 'trajectory_circle_pose.csv'
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
    """Generate the pose trajectory and write it to *output*. Returns list of rows."""
    dt = 1.0 / freq

    # Initial pose: start at (0, 0), heading in +x direction (theta = 0)
    # Circle center is at (0, radius); robot travels counter-clockwise.
    x     = 0.0
    y     = 0.0
    theta = 0.0

    rows = []

    def step(v: float):
        nonlocal x, y, theta
        omega  = v / radius
        x     += np.cos(theta) * dt
        y     += np.sin(theta) * dt
        theta += 2*np.pi/(10*freq)
        rows.append((x, y, theta))

    # --- Ramp-up phase ---
    angle_ramp = 0.0
    for i in range(ramp_steps):
        v = cosine_ramp(i, ramp_steps, v_max)
        angle_ramp += (v / radius) * dt
        step(v)

    # --- Constant-velocity phase (fill remaining angle for requested laps) ---
    omega_max      = v_max / radius
    dtheta_max     = omega_max * dt
    total_angle    = laps * 2.0 * np.pi
    angle_remaining = total_angle - angle_ramp
    n_const = int(np.ceil(angle_remaining / dtheta_max))
    for _ in range(n_const):
        step(v_max)

    # Write CSV
    output = os.path.normpath(output)
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'theta'])
        for row in rows:
            writer.writerow(row)

    print(
        f'Circle pose trajectory written to {output}\n'
        f'  radius={radius} m, v_max={v_max} m/s, laps={laps}, '
        f'freq={freq} Hz, ramp={ramp_steps} steps\n'
        f'  total steps={len(rows)}  '
        f'(ramp={ramp_steps}, constant={n_const})'
    )
    return rows


def main(args=None):
    """Entry point for standalone or ROS2 execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radius',     type=float, default=DEFAULT_RADIUS,
                        help='Circle radius [m]')
    parser.add_argument('--vmax',       type=float, default=DEFAULT_V_MAX,
                        help='Maximum forward velocity [m/s]')
    parser.add_argument('--laps',       type=int,   default=DEFAULT_LAPS,
                        help='Number of full laps')
    parser.add_argument('--freq',       type=float, default=DEFAULT_FREQ,
                        help='Sampling frequency [Hz]')
    parser.add_argument('--ramp_steps', type=int,   default=DEFAULT_RAMP_STEPS,
                        help='Number of cosine ramp-up steps')
    parser.add_argument('--output',     type=str,   default=DEFAULT_OUTPUT,
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
    circle_angle = np.arange(0, 2*np.pi, 2*np.pi/(20*30))
    circle_pose = np.array([[np.cos(a)-1, np.sin(a), a-np.pi/2] for a in circle_angle])
    output = os.path.normpath(DEFAULT_OUTPUT)
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'theta'])
        for pose in circle_pose:
            writer.writerow(pose)


if __name__ == '__main__':
    main()
