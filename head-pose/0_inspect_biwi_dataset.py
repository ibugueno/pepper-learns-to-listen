#!/usr/bin/env python3
"""
inspect_biwi_dataset.py

Utility script to inspect the file formats of the Biwi Kinect Head Pose Database.
It helps you understand exactly how the dataset is structured before building
a PyTorch Dataset or training a ViT for head-pose estimation.

It inspects:
- pose.txt: rotation matrix + head center
- depth.bin: compressed depth map (RLE format)
- pose.bin: binary ground truth pose (6 floats)
- subject.obj: 3D head mesh template (optional)
"""

import argparse
import struct
from pathlib import Path

import numpy as np


# -------------------------------------------------------------------------
# Read pose.txt (rotation matrix + 3D center)
# -------------------------------------------------------------------------

def read_pose_txt(path: Path):
    """
    Read Biwi pose.txt files.

    Format:
        Line 1-3 : 3×3 rotation matrix
        Line 4   : head center (x, y, z)

    Returns:
        R      — np.ndarray of shape (3,3)
        center — np.ndarray of shape (3,)
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 4:
        raise ValueError(f"pose.txt file {path} does not contain 4 numeric lines.")

    R = np.array([[float(v) for v in lines[i].split()] for i in range(3)])
    center = np.array([float(v) for v in lines[3].split()])
    return R, center


# -------------------------------------------------------------------------
# Read compressed depth.bin (run-length encoding)
# -------------------------------------------------------------------------

def read_depth_bin(path: Path):
    """
    Parse Biwi compressed depth.bin files.

    Format:
        int32 width
        int32 height
        Then repeated blocks:
            int32 numempty   (# pixels = 0)
            int32 numfull    (# pixels with depth values)
            int16[numfull]   depth values in mm

    Returns:
        depth_map — np.ndarray of shape (H, W), dtype int16
    """
    with open(path, "rb") as f:
        # Read width and height (int32, little endian)
        width = struct.unpack("i", f.read(4))[0]
        height = struct.unpack("i", f.read(4))[0]

        depth = np.zeros((height, width), dtype=np.int16)
        total_pixels = width * height
        p = 0

        # Decode RLE-like compressed structure
        while p < total_pixels:
            # How many zeros
            numempty_bytes = f.read(4)
            if not numempty_bytes:
                break
            numempty = struct.unpack("i", numempty_bytes)[0]

            # How many valid depth values
            numfull_bytes = f.read(4)
            if not numfull_bytes:
                break
            numfull = struct.unpack("i", numfull_bytes)[0]

            # Read numfull int16 depth values
            values = np.fromfile(f, dtype=np.int16, count=numfull)

            # Fill the depth map
            depth.flat[p:p+numempty] = 0
            depth.flat[p+numempty:p+numempty+numfull] = values

            p += numempty + numfull

    return depth


# -------------------------------------------------------------------------
# Read pose.bin (db_annotations) — 6 float32 values
# -------------------------------------------------------------------------

def read_pose_bin(path: Path):
    """
    Read the binary pose from db_annotations/<subject>/frame_xxxxx_pose.bin.

    Format (commonly used in Biwi code):
        float32[6] = (tx, ty, tz, rx, ry, rz)

    Returns:
        np.ndarray of shape (6,)
    """
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != 6:
        raise ValueError(f"Expected 6 floats in pose.bin, got {arr.size} in {path}")
    return arr


# -------------------------------------------------------------------------
# Summarize OBJ file (3D face template)
# -------------------------------------------------------------------------

def summarize_obj(path: Path, max_lines=10):
    """
    Print a brief summary of a Wavefront OBJ file:
    number of vertices, normals, faces, etc.
    """
    v = vt = vn = f = 0

    with open(path, "r") as file:
        for line in file:
            if line.startswith("v "):
                v += 1
            elif line.startswith("vt "):
                vt += 1
            elif line.startswith("vn "):
                vn += 1
            elif line.startswith("f "):
                f += 1

    print(f"\n[OBJ SUMMARY] {path.name}")
    print(f"  vertices:   {v}")
    print(f"  texcoords:  {vt}")
    print(f"  normals:    {vn}")
    print(f"  faces:      {f}")

    # Show the first lines for reference
    print("\n  First lines:")
    with open(path, "r") as file:
        for i, line in enumerate(file):
            if i >= max_lines:
                break
            print("   ", line.rstrip())


# -------------------------------------------------------------------------
# Main inspection routine
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect Biwi Head Pose dataset formats.")
    parser.add_argument("--faces_dir", required=True,
                        help="Path to archive/faces_0 directory")
    parser.add_argument("--db_annotations", required=True,
                        help="Path to archive/db_annotations directory")
    parser.add_argument("--subject", default="01",
                        help="Subject ID (e.g., 01–24)")
    parser.add_argument("--frame", default="00003",
                        help="Frame number (5 digits, e.g., 00003)")
    args = parser.parse_args()

    faces_dir = Path(args.faces_dir) / args.subject
    ann_dir = Path(args.db_annotations) / args.subject

    # File paths
    rgb_path   = faces_dir / f"frame_{args.frame}_rgb.png"
    depth_path = faces_dir / f"frame_{args.frame}_depth.bin"
    pose_txt   = faces_dir / f"frame_{args.frame}_pose.txt"
    pose_bin   = ann_dir   / f"frame_{args.frame}_pose.bin"
    obj_path   = faces_dir.with_suffix(".obj")

    print("\n========== Checking file existence ==========")
    for p in [rgb_path, depth_path, pose_txt, pose_bin, obj_path]:
        print(f"{p}:  {'FOUND' if p.exists() else 'MISSING'}")

    # ---------------------------------------------------------------
    # Inspect pose.txt
    # ---------------------------------------------------------------

    if pose_txt.exists():
        print("\n========== pose.txt ==========")
        R, center = read_pose_txt(pose_txt)
        print("Rotation matrix R:\n", R)
        print("Head center (x,y,z):", center)

    # ---------------------------------------------------------------
    # Inspect depth.bin
    # ---------------------------------------------------------------

    if depth_path.exists():
        print("\n========== depth.bin ==========")
        depth = read_depth_bin(depth_path)
        print("Depth shape:", depth.shape)
        print("Min/Max depth:", depth.min(), "/", depth.max())
        print("Top-left 5×5 sample:\n", depth[:5, :5])

    # ---------------------------------------------------------------
    # Inspect pose.bin
    # ---------------------------------------------------------------

    if pose_bin.exists():
        print("\n========== pose.bin ==========")
        arr = read_pose_bin(pose_bin)
        print("Binary pose (6 floats):", arr)

    # ---------------------------------------------------------------
    # Inspect OBJ model
    # ---------------------------------------------------------------

    if obj_path.exists():
        summarize_obj(obj_path)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
