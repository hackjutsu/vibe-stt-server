# How We Resolved the GPU Dependencies for Whisper (faster-whisper)

This document summarizes how we fixed the missing GPU dependencies (`cuDNN` and `cuBLAS`) required by `faster-whisper` on the remote Ubuntu machine.

---

## 1. Original Problem

Running faster-whisper on Ubuntu with Nvidia GPU(4090) produced two major errors:

### cuDNN missing/broken
```
Unable to load any of {libcudnn_ops.so.9...}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

### cuBLAS missing
```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

Whisper requires:
- cuDNN 9
- cuBLAS 12 (including libcublasLt.so.12)

---

## 2. Fixing cuDNN (libcudnn_ops.so.9)

1. Download cuDNN 9 for CUDA 13 (redistributable tarball).
2. Extract to: `~/cudnn-linux-x86_64-9.13.0.50_cuda13-archive/lib`
3. Create symlink: `ln -sf libcudnn_ops.so.9.13.0 libcudnn_ops.so.9`
4. Add to library path:
```bash
export LD_LIBRARY_PATH="$HOME/cudnn-linux-x86_64-9.13.0.50_cuda13-archive/lib:$LD_LIBRARY_PATH"
```

6. Verify via Python:
```python
ctypes.CDLL("libcudnn_ops.so.9")
```
Result: cuDNN loads correctly.

## 3. Fixing cuBLAS (libcublas.so.12)
Installing CUDA 12 via `apt` caused NVIDIA driver conflicts, so we used redistributable cuBLAS packages.

1. Go to: https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/
2. Download: `libcublas-linux-x86_64-12.4.2.65-archive.tar.xz`
3. Extract and copy:
```bash
mkdir -p ~/cuda12-lib
cp libcublas*.so* ~/cuda12-lib/
```

4. Add to library path:
```bash
export LD_LIBRARY_PATH="$HOME/cuda12-lib:$LD_LIBRARY_PATH"
```

5. Verify via Python
```python
ctypes.CDLL("libcublas.so.12")
ctypes.CDLL("libcublasLt.so.12")
ctypes.CDLL("libcudnn_ops.so.9")
```
Result: cuBLAS loads correctly.

## 4. Final Runtime Environment
Use this environment to run faster-whisper:

```bash
export LD_LIBRARY_PATH="\
$HOME/cuda12-lib:\
$HOME/cudnn-linux-x86_64-9.13.0.50_cuda13-archive/lib:\
/usr/local/cuda/lib64:\
${LD_LIBRARY_PATH}"
```

Then:
```bash
cd ~/projects/vibe-stt-server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 5. Summary
- Avoided apt to prevent NVIDIA driver conflicts.
- Installed only:
  - cuDNN 9
  - cuBLAS 12 (incl. Lt)
- Stored them in:
```bash
~/cuda12-lib
~/cudnn-linux-x86_64-9.13.0.50_cuda13-archive/lib
```
- Loaded them via `LD_LIBRARY_PATH`.
- Verified with Python ctypes.
- Ran uvicorn inside that environment.

## 6. Final Folder Layout
```bash
$HOME/
 ├─ cuda12-lib/
 │   ├─ libcublas.so.12
 │   ├─ libcublasLt.so.12
 │   └─ (other cuBLAS libs)
 │
 ├─ cudnn-linux-x86_64-9.13.0.50_cuda13-archive/
 │   └─ lib/
 │       ├─ libcudnn_ops.so.9
 │       └─ (other cuDNN libs)
 │
 └─ projects/vibe-stt-server/
     ├─ .venv/
     └─ app/main.py
```
