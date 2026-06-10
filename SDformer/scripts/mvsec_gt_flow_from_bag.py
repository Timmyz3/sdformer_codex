#!/usr/bin/env python3
"""Generate MVSEC gt.hdf5 flow_dist from local gt.bag (depth + odometry)."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import h5py
import numpy as np
import yaml
from importRosbag.importRosbag import importRosbag
from scipy.linalg import logm
from scipy.spatial.transform import Rotation


class FlowModel:
    def __init__(self, intrinsics: list[float], projection_matrix: list[list[float]], resolution: tuple[int, int]):
        self.Pfx = projection_matrix[0][0]
        self.Ppx = projection_matrix[0][2]
        self.Pfy = projection_matrix[1][1]
        self.Ppy = projection_matrix[1][2]
        self.P = np.array(
            [[self.Pfx, 0.0, self.Ppx], [0.0, self.Pfy, self.Ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        x_inds, y_inds = np.meshgrid(np.arange(resolution[0]), np.arange(resolution[1]))
        x_inds = x_inds.astype(np.float32)
        y_inds = y_inds.astype(np.float32)
        x_inds = (x_inds - self.P[0, 2]) / self.P[0, 0]
        y_inds = (y_inds - self.P[1, 2]) / self.P[1, 1]
        self.flat_x_map = x_inds.reshape(-1)
        self.flat_y_map = y_inds.reshape(-1)

        n = self.flat_x_map.shape[0]
        self.omega_mat = np.zeros((n, 2, 3), dtype=np.float64)
        self.omega_mat[:, 0, 0] = self.flat_x_map * self.flat_y_map
        self.omega_mat[:, 1, 0] = 1.0 + np.square(self.flat_y_map)
        self.omega_mat[:, 0, 1] = -(1.0 + np.square(self.flat_x_map))
        self.omega_mat[:, 1, 1] = -(self.flat_x_map * self.flat_y_map)
        self.omega_mat[:, 0, 2] = self.flat_y_map
        self.omega_mat[:, 1, 2] = -self.flat_x_map

    def compute_flow_single_frame(self, velocity: np.ndarray, omega: np.ndarray, depth_image: np.ndarray, dt: float):
        flat_depth = depth_image.reshape(-1).astype(np.float64)
        mask = np.isfinite(flat_depth) & (flat_depth > 0)

        fdm = 1.0 / flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask, :, :]

        x_flow = np.zeros(depth_image.shape, dtype=np.float64)
        y_flow = np.zeros(depth_image.shape, dtype=np.float64)
        flat_x = x_flow.reshape(-1)
        flat_y = y_flow.reshape(-1)

        flat_x[mask] = fdm * (fxm * velocity[2] - velocity[0])
        flat_x[mask] += np.squeeze(omm[:, 0, :] @ omega)
        flat_y[mask] = fdm * (fym * velocity[2] - velocity[1])
        flat_y[mask] += np.squeeze(omm[:, 1, :] @ omega)

        flat_x[mask] *= dt * self.P[0, 0]
        flat_y[mask] *= dt * self.P[1, 1]
        return x_flow.astype(np.float32), y_flow.astype(np.float32)

    @staticmethod
    def velocity_from_poses(p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray, dt: float):
        rot0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
        rot1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
        h0 = np.eye(4, dtype=np.float64)
        h1 = np.eye(4, dtype=np.float64)
        h0[:3, :3] = rot0.as_matrix()
        h1[:3, :3] = rot1.as_matrix()
        h0[:3, 3] = p0
        h1[:3, 3] = p1
        h01 = np.linalg.inv(h0) @ h1
        velocity = h01[:3, 3] / dt
        w_hat = logm(h01[:3, :3]) / dt
        omega = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64)
        return velocity, omega, dt


def load_calibration(calib_zip: Path) -> FlowModel:
    with zipfile.ZipFile(calib_zip) as handle:
        with handle.open("camchain-imucam-indoor_flying.yaml") as stream:
            calib = yaml.safe_load(stream)
    cam0 = calib["cam0"]
    return FlowModel(cam0["intrinsics"], cam0["projection_matrix"], tuple(cam0["resolution"]))


def generate_gt_hdf5(sequence: str, mvsec_root: Path, calib_zip: Path, output: Path, filter_size: int = 10) -> None:
    gt_bag = mvsec_root / sequence / f"{sequence}_gt.bag"
    topics = importRosbag(str(gt_bag), log="ERROR")
    depth = topics["/davis/left/depth_image_rect"]
    odom = topics["/davis/left/odometry"]

    depth_frames = np.stack(depth["frames"]).astype(np.float32)
    depth_ts = np.asarray(depth["ts"], dtype=np.float64)
    odom_ts = np.asarray(odom["ts"], dtype=np.float64)
    odom_point = np.asarray(odom["point"], dtype=np.float64)
    odom_rot = np.asarray(odom["rotation"], dtype=np.float64)

    nframes = min(len(depth_frames), len(odom_ts))
    flow_model = load_calibration(calib_zip)

    velocities = np.zeros((nframes, 3), dtype=np.float64)
    omegas = np.zeros((nframes, 3), dtype=np.float64)
    dts = np.zeros((nframes,), dtype=np.float64)
    timestamps = odom_ts[:nframes]

    for idx in range(1, nframes):
        dt = odom_ts[idx] - odom_ts[idx - 1]
        if dt <= 0:
            continue
        vel, omega, _ = flow_model.velocity_from_poses(
            odom_point[idx - 1], odom_rot[idx - 1], odom_point[idx], odom_rot[idx], dt
        )
        velocities[idx] = vel
        omegas[idx] = omega
        dts[idx] = dt

    x_flow = np.zeros((nframes, depth_frames.shape[1], depth_frames.shape[2]), dtype=np.float32)
    y_flow = np.zeros_like(x_flow)

    for idx in range(nframes):
        lo = max(0, idx - filter_size)
        hi = min(nframes, idx + filter_size + 1)
        vel = velocities[lo:hi].mean(axis=0)
        omega = omegas[lo:hi].mean(axis=0)
        dt = dts[idx] if dts[idx] > 0 else np.median(dts[dts > 0])
        xf, yf = flow_model.compute_flow_single_frame(vel, omega, depth_frames[idx], dt)
        x_flow[idx] = xf
        y_flow[idx] = yf

    flow_dist = np.stack((x_flow, y_flow), axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        grp = handle.create_group("davis").create_group("left")
        grp.create_dataset("flow_dist", data=flow_dist, compression="gzip")
        grp.create_dataset("flow_dist_ts", data=timestamps)
    print(f"[gt] wrote {output} flow_dist={flow_dist.shape} ts={timestamps.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="indoor_flying3")
    parser.add_argument("--mvsec-root", type=Path, required=True)
    parser.add_argument(
        "--calib-zip",
        type=Path,
        default=Path("third_party/SDformerFlow/data/Datasets/MVSEC/indoor_flying_calib.zip"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.mvsec_root / args.sequence / f"{args.sequence}_gt.hdf5"
    generate_gt_hdf5(args.sequence, args.mvsec_root, args.calib_zip, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())