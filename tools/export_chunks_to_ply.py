#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import numpy as np

TORCH_DTYPE_TO_NUMPY = {
    0: np.uint8,     # torch.uint8
    1: np.int8,      # torch.int8
    2: np.int16,     # torch.int16
    3: np.int32,     # torch.int32
    4: np.int64,     # torch.int64
    5: np.float16,   # torch.float16
    6: np.float32,   # torch.float32
    7: np.float64,   # torch.float64
    11: np.bool_,    # torch.bool
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

def read_tensor(f):
    header = f.read(48)
    if len(header) != 48:
        raise RuntimeError("TensorHeader incompleto")

    dims = struct.unpack("<I", header[0:4])[0]
    sizes = struct.unpack("<8I", header[4:36])
    dtype_id = struct.unpack("<I", header[36:40])[0]
    data_size = struct.unpack("<Q", header[40:48])[0]

    if dtype_id not in TORCH_DTYPE_TO_NUMPY:
        raise RuntimeError(f"dtype torch não mapeado: {dtype_id}")

    shape = tuple(sizes[:dims])
    dtype = TORCH_DTYPE_TO_NUMPY[dtype_id]
    raw = f.read(data_size)

    arr = np.frombuffer(raw, dtype=dtype).copy()
    return arr.reshape(shape)

def read_chunk(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != 0x43484E4B:
            raise RuntimeError(f"Magic inválido em {path}: {hex(magic)}")

        chunk_id = struct.unpack("<q", f.read(8))[0]
        num_points = struct.unpack("<I", f.read(4))[0]

        data = {}
        for name in TENSOR_NAMES:
            data[name] = read_tensor(f)

        return chunk_id, num_points, data

def flatten_features_dc(x):
    # esperado: [N,1,3] ou [N,3,1]
    if x.ndim == 3:
        if x.shape[1] == 1 and x.shape[2] == 3:
            return x[:, 0, :]
        if x.shape[1] == 3 and x.shape[2] == 1:
            return x[:, :, 0]
    return x.reshape((x.shape[0], -1))[:, :3]

def flatten_features_rest(x):
    # esperado: [N,15,3] ou [N,3,15]
    if x.ndim == 3:
        if x.shape[2] == 3:
            return x.reshape((x.shape[0], -1))
        if x.shape[1] == 3:
            return np.transpose(x, (0, 2, 1)).reshape((x.shape[0], -1))
    return x.reshape((x.shape[0], -1))

def write_ply(out_path, xyz, fdc, frest, opacity, scaling, rotation):
    n = xyz.shape[0]
    frest_count = frest.shape[1]

    props = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    props += [f"f_rest_{i}" for i in range(frest_count)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for p in props:
            f.write(f"property float {p}\n")
        f.write("end_header\n")

        zeros = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            row = []
            row.extend(xyz[i].astype(float).tolist())
            row.extend(zeros[i].astype(float).tolist())
            row.extend(fdc[i].astype(float).tolist())
            row.extend(frest[i].astype(float).tolist())
            row.append(float(opacity[i].reshape(-1)[0]))
            row.extend(scaling[i].astype(float).tolist())
            row.extend(rotation[i].astype(float).tolist())
            f.write(" ".join(map(str, row)) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks, "*.bin")))
    if not files:
        raise SystemExit(f"Nenhum .bin encontrado em: {args.chunks}")

    all_xyz = []
    all_fdc = []
    all_frest = []
    all_scaling = []
    all_rotation = []
    all_opacity = []

    print(f"Chunks encontrados: {len(files)}")

    for path in files:
        chunk_id, num_points, data = read_chunk(path)

        xyz = data["xyz"].astype(np.float32)
        fdc = flatten_features_dc(data["features_dc"]).astype(np.float32)
        frest = flatten_features_rest(data["features_rest"]).astype(np.float32)
        scaling = data["scaling"].astype(np.float32)
        rotation = data["rotation"].astype(np.float32)
        opacity = data["opacity"].astype(np.float32)

        print(f"\n{os.path.basename(path)}")
        print(f"  chunk_id: {chunk_id}")
        print(f"  num_points header: {num_points}")
        print(f"  xyz: {xyz.shape}")
        print(f"  features_dc: {data['features_dc'].shape} -> {fdc.shape}")
        print(f"  features_rest: {data['features_rest'].shape} -> {frest.shape}")
        print(f"  scaling: {scaling.shape}")
        print(f"  rotation: {rotation.shape}")
        print(f"  opacity: {opacity.shape}")

        all_xyz.append(xyz)
        all_fdc.append(fdc)
        all_frest.append(frest)
        all_scaling.append(scaling)
        all_rotation.append(rotation)
        all_opacity.append(opacity)

    xyz = np.concatenate(all_xyz, axis=0)
    fdc = np.concatenate(all_fdc, axis=0)
    frest = np.concatenate(all_frest, axis=0)
    scaling = np.concatenate(all_scaling, axis=0)
    rotation = np.concatenate(all_rotation, axis=0)
    opacity = np.concatenate(all_opacity, axis=0)

    print(f"\nTOTAL GAUSSIANS: {xyz.shape[0]}")
    print(f"Exportando PLY: {args.out}")

    write_ply(args.out, xyz, fdc, frest, opacity, scaling, rotation)

    print("OK: PLY gerado.")

if __name__ == "__main__":
    main()
