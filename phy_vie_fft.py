"""
Matrix-free implementation of the VIE coefficient matrix W.

Background
----------
In `SpectrumNet/gen_incident_field.py:build_W_matrix`,
    W[i,j] = 1j*pi*ka/2 * J1(ka) * H2_0(k*R_ij)      (i != j)
    W[i,i] = 1j/2 * (pi*ka*H2_1(ka) - 2j)
depends only on the pairwise distance R_ij = |p_i - p_j|. And `build_grid_no_center`
produces a regular grid
    p_i = (col_i*dx, row_i*dy),  i = row_i*Nx + col_i
so
    R_ij = sqrt(((col_i-col_j)*dx)^2 + ((row_i-row_j)*dy)^2)
depends only on the **index differences** (drow, dcol). In other words W is a BTTB matrix
(block-Toeplitz with Toeplitz blocks), equivalent to a 2D convolution kernel w(drow, dcol):

    (W v)(m,n) = sum_{p,q} w(m-p, n-q) v(p,q)

W @ v can therefore be computed exactly with circulant embedding + FFT:
    - only the (2Ny, 2Nx) kernel spectrum has to be stored; on a 128x128 grid that is
      256x256 complex values ~= 0.5 MB (instead of 16384x16384 complex128 = 4.3 GB
      per frequency)
    - each matvec costs O(N log N) instead of O(N^2)
    - the result is **identical** to the dense W @ v up to numerical precision (not an
      approximation)
This is the usual CG-FFT / BCGS-FFT trick found in MoM/VIE solvers.

The kernel depends only on (Ny, Nx, dy, dx, f_hz, cell_area) and not on the sample, and
the whole dataset uses just 5 frequencies -> 5 kernels, built once at startup and cached
on the GPU.

Usage
-----
    from phy_vie_fft import WKernelCache, incident_from_total_fft

    kcache = WKernelCache(Ny=128, Nx=128, dy=10.0, dx=10.0,
                          cell_area=100.0, device=device)
    ...
    E2 = incident_from_total_fft(E_total_complex, chi, f_hz, kcache)

Note
----
dx/dy/cell_area must match the values used when generating E_inc / chi (the __main__ of
`gen_incident_field.py` uses dx=dy=10.0, cell_area=100.0), otherwise the
physics-consistency loss and the input E_inc live on different discretisations.
"""

import numpy as np
import torch
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0, pi
from scipy.special import hankel2, j1

__all__ = [
    "wavenumber",
    "build_W_kernel_lags",
    "build_W_kernel_spectrum",
    "WKernelCache",
    "apply_W_fft",
    "incident_from_total_fft",
]


def wavenumber(f_hz, eps0=EPS0, mu0=MU0):
    """Kept consistent with gen_incident_field.wavenumber."""
    omega = 2 * pi * f_hz
    return omega * np.sqrt(eps0 * mu0), omega


def _lag_index(n):
    """
    Lag indices of a circulant embedding of length 2n:
        [0, 1, ..., n-1, (unused), -(n-1), ..., -1]
    Returns (lags, valid_mask). The row/column at index == n corresponds to no valid lag
    and is zeroed out — the standard way of keeping circulant embedding free of
    wrap-around contamination.
    """
    idx = np.arange(2 * n)
    lags = np.where(idx > n, idx - 2 * n, idx)
    valid = idx != n
    return lags, valid


def build_W_kernel_lags(Ny, Nx, dy, dx, f_hz, cell_area):
    """
    Build the 2D convolution kernel of W, already laid out as a (2Ny, 2Nx) circulant
    embedding.

    Returns: (2Ny, 2Nx) complex128 ndarray
    """
    k, _ = wavenumber(f_hz)
    a = np.sqrt(cell_area / pi)
    ka = k * a

    dr, valid_r = _lag_index(Ny)
    dc, valid_c = _lag_index(Nx)

    # Pairwise distances depend only on the lag
    R = np.sqrt((dr[:, None] * dy) ** 2 + (dc[None, :] * dx) ** 2)  # (2Ny, 2Nx)

    # Off-diagonal terms: 1j*pi*ka/2 * J1(ka) * H2_0(k*R)
    R_safe = np.where(R == 0.0, 1.0, R)
    K = 1j * pi * ka / 2.0 * j1(ka) * hankel2(0, k * R_safe)

    # Diagonal term (lag = (0,0))
    K[0, 0] = 1j / 2.0 * (pi * ka * hankel2(1, ka) - 2j)

    # Zero out the invalid lag row/column
    mask = valid_r[:, None] & valid_c[None, :]
    K = np.where(mask, K, 0.0)

    return K.astype(np.complex128)


def build_W_kernel_spectrum(Ny, Nx, dy, dx, f_hz, cell_area,
                            device=None, dtype=torch.complex64):
    """
    Return the FFT spectrum of the kernel, Ghat: (2Ny, 2Nx) complex tensor.

    The kernel itself is built and FFT'd on the CPU in complex128 (at 22 GHz k*R is large
    and the phase oscillates violently, so building in double precision and downcasting to
    complex64 is more stable than staying in single precision throughout), then cast to
    `dtype`.
    """
    K = build_W_kernel_lags(Ny, Nx, dy, dx, f_hz, cell_area)
    Ghat = np.fft.fft2(K)  # complex128
    t = torch.from_numpy(Ghat).to(dtype)
    if device is not None:
        t = t.to(device)
    return t


class WKernelCache:
    """
    Cache of kernel spectra keyed by frequency. SpectrumNet only has 5 frequencies, so at
    most 5 * (2Ny x 2Nx) complex values stay resident on the GPU (about 2.5 MB for a
    128x128 grid).
    """

    def __init__(self, Ny, Nx, dy, dx, cell_area,
                 device=None, dtype=torch.complex64):
        self.Ny, self.Nx = int(Ny), int(Nx)
        self.dy, self.dx = float(dy), float(dx)
        self.cell_area = float(cell_area)
        self.device = device
        self.dtype = dtype
        self._cache = {}

    @staticmethod
    def _key(f_hz):
        # Frequencies come from the png metadata (150/1500/1700/3500/22000 MHz), so
        # rounding to whole Hz is a stable enough key.
        return int(round(float(f_hz)))

    def get(self, f_hz):
        key = self._key(f_hz)
        Ghat = self._cache.get(key)
        if Ghat is None:
            Ghat = build_W_kernel_spectrum(
                self.Ny, self.Nx, self.dy, self.dx, float(key),
                self.cell_area, device=self.device, dtype=self.dtype)
            self._cache[key] = Ghat
        return Ghat

    def prebuild(self, f_hz_list):
        for f in f_hz_list:
            self.get(f)
        return self

    def __len__(self):
        return len(self._cache)


def apply_W_fft(v_bhw, Ghat):
    """
    Compute y = W @ v, where v flattened in row-major order corresponds to an (H, W) image.

    v_bhw: (B, H, W) complex
    Ghat:  (2H, 2W) complex  -- kernel spectrum
    Returns: (B, H, W) complex

    Differentiable: torch.fft supports autograd on complex tensors.
    """
    B, H, W = v_bhw.shape
    P, Q = Ghat.shape
    assert (P, Q) == (2 * H, 2 * W), \
        f"kernel spectrum {Ghat.shape} does not match field size {(H, W)}"

    # Zero-pad to (2H, 2W) to get a linear (non-circular) convolution
    v_pad = torch.zeros(B, P, Q, dtype=Ghat.dtype, device=v_bhw.device)
    v_pad[:, :H, :W] = v_bhw.to(Ghat.dtype)

    y = torch.fft.ifft2(torch.fft.fft2(v_pad) * Ghat.unsqueeze(0))
    return y[:, :H, :W]


def incident_from_total_fft(E_tot_bchw, chi_bn, f_hz, kernel_cache):
    """
    E_inc = E_tot + W @ (chi * E_tot), implemented with FFTs. Equivalent to
    train_phy_predict_lambda.batch_incident_from_total_bmm but without needing W.

    E_tot_bchw: (B,1,H,W) complex
    chi_bn:     (B,N) or (B,H,W) complex
    f_hz:       scalar / tensor or sequence of length B (a batch may mix frequencies)
    kernel_cache: WKernelCache

    Returns: (B,2,H,W) float -- real/imaginary channels, same interface as before
    """
    B, _, H, W = E_tot_bchw.shape
    E_tot = E_tot_bchw.view(B, H, W)
    v = chi_bn.reshape(B, H, W).to(E_tot.dtype) * E_tot

    # Normalise the frequencies into a python list of length B
    if torch.is_tensor(f_hz):
        f_list = [float(x) for x in f_hz.reshape(-1)]
    elif np.isscalar(f_hz):
        f_list = [float(f_hz)] * B
    else:
        f_list = [float(x) for x in f_hz]
    if len(f_list) == 1 and B > 1:
        f_list = f_list * B
    assert len(f_list) == B, f"f_hz length {len(f_list)} does not match batch {B}"

    uniq = sorted(set(WKernelCache._key(f) for f in f_list))
    if len(uniq) == 1:
        y = apply_W_fft(v, kernel_cache.get(uniq[0]))
    else:
        # Mixed frequencies within a batch: group by frequency, one FFT per group
        y = torch.zeros_like(v, dtype=kernel_cache.dtype)
        keys = torch.tensor([WKernelCache._key(f) for f in f_list])
        for u in uniq:
            sel = (keys == u).nonzero(as_tuple=True)[0].to(v.device)
            y = y.index_copy(0, sel,
                             apply_W_fft(v.index_select(0, sel),
                                         kernel_cache.get(u)))

    E_inc = E_tot + y
    return torch.cat([E_inc.real.unsqueeze(1),
                      E_inc.imag.unsqueeze(1)], dim=1).float()


# ---------------------------------------------------------------- self-check ----
if __name__ == "__main__":
    import time
    from scipy.spatial.distance import cdist

    def build_W_dense(positions, f_hz, cell_area):
        """Copy of gen_incident_field.build_W_matrix, used as ground truth."""
        k, _ = wavenumber(f_hz)
        a = np.sqrt(cell_area / pi)
        ka = k * a
        R = cdist(positions, positions, metric="euclidean")
        Wm = 1j * pi * ka / 2.0 * j1(ka) * hankel2(0, k * np.where(R == 0.0, 1.0, R))
        np.fill_diagonal(Wm, 0.0)
        np.fill_diagonal(Wm, 1j / 2.0 * (pi * ka * hankel2(1, ka) - 2j))
        return Wm.astype(np.complex128)

    def build_grid_no_center(Nx, Ny, dx, dy, origin=(0.0, 0.0)):
        x0, y0 = origin
        cols, rows = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="xy")
        X = x0 + cols * dx
        Y = y0 + rows * dy
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    dx = dy = 10.0
    cell_area = dx * dy
    f_mhz_lst = [150, 1500, 1700, 3500, 22000]

    print("=== accuracy check: FFT matvec vs dense W @ v  (Ny=Nx=24) ===")
    Ny = Nx = 24
    pos = build_grid_no_center(Nx, Ny, dx, dy)
    rng = np.random.default_rng(0)
    v = (rng.standard_normal(Ny * Nx) + 1j * rng.standard_normal(Ny * Nx)).astype(np.complex128)

    for f_mhz in f_mhz_lst:
        f_hz = f_mhz * 1e6
        Wd = build_W_dense(pos, f_hz, cell_area)
        ref = Wd @ v

        for dt in (torch.complex128, torch.complex64):
            Ghat = build_W_kernel_spectrum(Ny, Nx, dy, dx, f_hz, cell_area, dtype=dt)
            got = apply_W_fft(
                torch.from_numpy(v.reshape(1, Ny, Nx)).to(dt), Ghat
            ).numpy().reshape(-1)
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            print(f"  f={f_mhz:>6} MHz  {str(dt).split('.')[-1]:>10}  rel.err = {rel:.3e}")

    print("\n=== 128x128 measurements (kernel build / matvec) ===")
    Ny = Nx = 128
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kc = WKernelCache(Ny, Nx, dy, dx, cell_area, device=dev, dtype=torch.complex64)
    t0 = time.time()
    kc.prebuild([f * 1e6 for f in f_mhz_lst])
    print(f"  total build time for 5 frequency kernels: {time.time() - t0:.2f} s, {len(kc)} cached")
    one = kc.get(150e6)
    print(f"  one kernel spectrum: {tuple(one.shape)} {one.dtype} = "
          f"{one.element_size() * one.numel() / 1024**2:.2f} MB "
          f"(dense W complex128 = {(Ny*Nx)**2 * 16 / 1024**3:.2f} GB)")

    B = 4
    E = torch.randn(B, 1, Ny, Nx, device=dev) + 1j * torch.randn(B, 1, Ny, Nx, device=dev)
    E = E.to(torch.complex64).requires_grad_(True)
    chi = (torch.randn(B, Ny * Nx, device=dev) + 1j * torch.randn(B, Ny * Nx, device=dev)).to(torch.complex64)
    f_batch = torch.tensor([150e6, 1500e6, 150e6, 22000e6])

    out = incident_from_total_fft(E, chi, f_batch, kc)
    out.sum().backward()
    print(f"  mixed-frequency batch output {tuple(out.shape)}, backward OK, grad finite = "
          f"{bool(torch.isfinite(E.grad.real).all())}")

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    reps = 50
    for _ in range(reps):
        incident_from_total_fft(E.detach(), chi, 150e6, kc)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"  B={B} forward matvec: {(time.time() - t0) / reps * 1e3:.2f} ms/call on {dev}")
