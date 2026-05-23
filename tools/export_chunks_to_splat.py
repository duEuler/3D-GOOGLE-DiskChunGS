#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import numpy as np

TORCH_DTYPE_TO_NUMPY = {
    0: np.uint8,
    1: np.int8,
    2: np.int16,
    3: np.int32,
    4: np.int64,
    5: np.float16,
    6: np.float32,
    7: np.float64,
    11: np.bool_,
}

TENSOR_NAMES = [
    "xyz",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
    "exist_since",
    "position_lrs",
    "gaussian_ids",
]

SH_C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def read_tensor(f):
    header = f.read(48)
    dims = struct.unpack("<I", header[0:4])[0]
    sizes = struct.unpack("<8I", header[4:36])
    dtype_id = struct.unpack("<I", header[36:40])[0]
    data_size = struct.unpack("<Q", header[40:48])[0]

    dtype = TORCH_DTYPE_TO_NUMPY[dtype_id]
    shape = tuple(sizes[:dims])
    raw = f.read(data_size)
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)

def read_chunk(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != 0x43484E4B:
            raise RuntimeError(f"Magic inválido: {path}")

        chunk_id = struct.unpack("<q", f.read(8))[0]
        num_points = struct.unpack("<I", f.read(4))[0]

        data = {}
        for name in TENSOR_NAMES:
            data[name] = read_tensor(f)

        return chunk_id, num_points, data

def flatten_dc(x):
    if x.ndim == 3:
        if x.shape[1] == 1 and x.shape[2] == 3:
            return x[:, 0, :]
        if x.shape[1] == 3 and x.shape[2] == 1:
            return x[:, :, 0]
    return x.reshape((x.shape[0], -1))[:, :3]

def normalize_quat(q):
    q = q.astype(np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return q / norm

def export_splat(out_path, xyz, fdc, scaling, rotation, opacity):
    n = xyz.shape[0]

    # GraphDECO armazena scale/opacidade em espaço pré-ativação.
    scale = np.exp(scaling.astype(np.float32)) * 0.35
    alpha = sigmoid(opacity.astype(np.float32).reshape(n, -1)[:, 0])

    # Cor DC para RGB aproximado: RGB = SH*C0 + 0.5
    rgb = np.clip(fdc.astype(np.float32) * SH_C0 + 0.5, 0.0, 1.0)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    rgba[:, 0:3] = (rgb * 255).astype(np.uint8)
    rgba[:, 3] = (alpha * 255).astype(np.uint8)

    q = normalize_quat(rotation)
    rot = np.clip(q * 128.0 + 128.0, 0, 255).astype(np.uint8)

    with open(out_path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<3f", float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))
            f.write(struct.pack("<3f", float(scale[i,0]), float(scale[i,1]), float(scale[i,2])))
            f.write(rgba[i].tobytes())
            f.write(rot[i].tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "*.bin")))
    if not files:
        raise SystemExit("Nenhum chunk encontrado")

    all_xyz, all_fdc, all_scaling, all_rotation, all_opacity = [], [], [], [], []

    print(f"Chunks: {len(files)}")

    for path in files:
        chunk_id, num_points, data = read_chunk(path)
        xyz = data["xyz"].astype(np.float32)
        fdc = flatten_dc(data["features_dc"]).astype(np.float32)
        scaling = data["scaling"].astype(np.float32)
        rotation = data["rotation"].astype(np.float32)
        opacity = data["opacity"].astype(np.float32)

        print(f"{os.path.basename(path)}: {xyz.shape[0]} gaussians")

        all_xyz.append(xyz)
        all_fdc.append(fdc)
        all_scaling.append(scaling)
        all_rotation.append(rotation)
        all_opacity.append(opacity)

    xyz = np.concatenate(all_xyz, axis=0)
    fdc = np.concatenate(all_fdc, axis=0)
    scaling = np.concatenate(all_scaling, axis=0)
    rotation = np.concatenate(all_rotation, axis=0)
    opacity = np.concatenate(all_opacity, axis=0)

    print(f"Total: {xyz.shape[0]} gaussians")
    export_splat(args.out, xyz, fdc, scaling, rotation, opacity)
    print(f"OK: {args.out}")

if __name__ == "__main__":
    main()
