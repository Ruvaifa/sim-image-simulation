import numpy as np
import matplotlib.pyplot as plt

# ---------- Ground truth (synthetic) ----------
def generate_ground_truth(N=512):
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    img = np.zeros((N, N), float)
    # grid (fine detail)
    img += (np.sin(24*np.pi*X) > 0).astype(float) * 0.4
    img += (np.sin(24*np.pi*Y) > 0).astype(float) * 0.4
    # rings (varied frequencies, inside a circle)
    img += (np.sin(30*np.pi*R) > 0).astype(float) * 0.3 * (R < 0.6)

    # normalize to [0,1]
    img -= img.min()
    img /= (img.max() + 1e-9)
    return img

# ---------- Low-pass filter ----------
def low_pass_filter(img, cutoff=0.2):
    """
    Apply a circular low-pass filter in Fourier domain.
    cutoff: relative frequency (0..1), where 1 = Nyquist frequency.
    """
    N, M = img.shape
    f = np.fft.fftshift(np.fft.fft2(img))

    # frequency grid
    u = np.linspace(-0.5, 0.5, N, endpoint=False)
    U, V = np.meshgrid(u, u)
    R = np.sqrt(U**2 + V**2)

    # mask: keep only frequencies within cutoff
    mask = R < cutoff

    f_filtered = f * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(f_filtered))
    return np.abs(img_filtered)

# ---------- Scale bar helper ----------
def add_scale_bar(ax, img_shape, microns_per_pixel=0.10, bar_um=5,
                  thickness_px=4, margin_px=12):
    h, w = img_shape
    bar_px = int(round(bar_um / microns_per_pixel))

    x1 = margin_px
    x2 = x1 + bar_px
    y = h - margin_px

    ax.hlines(y, x1, x2, colors='white', linewidth=thickness_px)
    ax.text(x1, y - 8, f"{bar_um} µm", color='white', fontsize=10, va='bottom')

def illumination_pattern(N, freq_cycles=8, angle_deg=0.0, phase=0.0):
    """
    Create a 2D sinusoidal grating pattern:
      pattern = 0.5 * (1 + cos(2*pi*(fx*x + fy*y) + phase))
    freq_cycles: spatial frequency in cycles across the whole image width
    angle_deg: orientation of the grating in degrees (0 => horizontal stripes along x)
    phase: phase in radians
    """
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    # spatial frequency vector in normalized coords (cycles per unit)
    theta = np.deg2rad(angle_deg)
    fx = freq_cycles * np.cos(theta) / 2.0   # /2 because x in [-1,1] is width 2 units
    fy = freq_cycles * np.sin(theta) / 2.0
    arg = 2 * np.pi * (fx * X + fy * Y) + phase
    pattern = 0.5 * (1.0 + np.cos(arg))   # range [0,1]
    return pattern

def show_fft(img, ax, title="FFT"):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(f))  # log for visibility
    ax.imshow(mag, cmap='inferno')
    ax.set_title(title)
    ax.axis('off')

def separate_components(img, cutoff=0.08, band_radius=0.015, band_shift=0.12):
    N, M = img.shape
    f = np.fft.fftshift(np.fft.fft2(img))
    
    # frequency grid
    u = np.linspace(-0.5, 0.5, N, endpoint=False)
    U, V = np.meshgrid(u, u)
    R = np.sqrt(U**2 + V**2)

    # --- Low-frequency mask (center) ---
    low_mask = R < cutoff

    # --- High-frequency mask (sidebands) ---
    high_mask = ((np.sqrt((U- band_shift)**2 + V**2) < band_radius) |
                 (np.sqrt((U+ band_shift)**2 + V**2) < band_radius))

    # Apply masks
    f_low = f * low_mask
    f_high = f * high_mask

    low_comp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_low)))
    high_comp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_high)))
    return low_comp, high_comp

def demodulate_three_phase_F(F_imgs, phases):
    """
    Vectorized demodulation of three-phase SIM images in Fourier domain.
    Inputs:
      F_imgs : array shape (3, N, N) - FFTshifted Fourier transforms of the 3 phase images
      phases : list/array length 3 - phases in radians (phi0, phi1, phi2)
    Returns:
      F0, Fp, Fm : arrays (N, N) - the 0th, +1 and -1 Fourier components (still centered / fftshifted)
    """
    # Build mixing matrix M where:
    # I_p(u) = F0(u) + Fp(u) * e^{i phi_p} + Fm(u) * e^{-i phi_p}
    # rows = p (phases), cols = [1, e^{i phi}, e^{-i phi}]
    ph = np.array(phases)
    M = np.vstack([np.ones_like(ph), np.exp(1j*ph), np.exp(-1j*ph)]).T  # shape (3,3)
    Minv = np.linalg.pinv(M)  # shape (3,3) - pseudo-inverse (constant across frequencies)

    # reshape F_imgs to (3, N*N) to do linear algebra in one matrix multiplication
    s0, s1, s2 = F_imgs.shape
    assert s0 == 3
    N = s1
    Fstack = F_imgs.reshape(3, -1)  # (3, N*N)
    comps = Minv @ Fstack            # (3, N*N)
    F0 = comps[0].reshape(N, N)
    Fp = comps[1].reshape(N, N)
    Fm = comps[2].reshape(N, N)
    return F0, Fp, Fm


def shift_spectrum(F, shift_pix):
    """
    Shift a 2D fftshifted spectrum by integer pixels using np.roll.
    shift_pix: tuple (shift_y, shift_x) in pixels (positive means roll downward / right).
    We assume input is fftshifted; roll acts on array indices.
    """
    sy, sx = int(round(shift_pix[0])), int(round(shift_pix[1]))
    return np.roll(np.roll(F, sy, axis=0), sx, axis=1)

def shift_spectrum_subpixel(F, shift_pix):
    """
    Subpixel shift of a 2D fftshifted spectrum using phase ramps.
    shift_pix: (shift_y, shift_x) in pixels
    """
    N, M = F.shape
    u = np.fft.fftfreq(N) * N   # frequency indices [-N/2..N/2-1] after fftshift
    U, V = np.meshgrid(u, u)
    dy, dx = shift_pix
    # Phase ramp for shifting
    ramp = np.exp(-2j * np.pi * (U*dx/N + V*dy/N))
    return F * ramp


def reconstruct_from_orientation(sim_imgs_phases, phases, angle_deg, band_shift_norm, N, noise_var=1e-6):
    """
    Reconstruct image from three-phase SIM images of one grating orientation.
    Includes Wiener-like regularization.
    """
    # compute FFTs (fftshifted)
    F_imgs = np.stack([np.fft.fftshift(np.fft.fft2(im)) for im in sim_imgs_phases], axis=0)  # (3,N,N)

    # demodulate -> obtain F0(u), F+1(u), F-1(u) (still centered)
    F0, Fp, Fm = demodulate_three_phase_F(F_imgs, phases)  # complex arrays

    # compute pixel shifts to move sidebands to center
    theta = np.deg2rad(angle_deg)
    kx = band_shift_norm * np.cos(theta) * N
    ky = band_shift_norm * np.sin(theta) * N

    Fp_shifted = shift_spectrum_subpixel(Fp, (-ky, -kx))
    Fm_shifted = shift_spectrum_subpixel(Fm, ( ky,  kx))
    # --- Wiener-like weighting ---
    P0 = np.abs(F0)**2
    Pp = np.abs(Fp_shifted)**2
    Pm = np.abs(Fm_shifted)**2

    w0 = P0 / (P0 + noise_var)
    wp = Pp / (Pp + noise_var)
    wm = Pm / (Pm + noise_var)

    # Combine spectra using weights
    F_combined = w0 * F0 + wp * Fp_shifted + wm * Fm_shifted

    # Inverse transform to get reconstructed image
    recon = np.fft.ifft2(np.fft.ifftshift(F_combined))
    recon_real = np.real(recon)
    recon_real -= recon_real.min()
    recon_real /= (recon_real.max() + 1e-12)
    return recon_real, F_combined


# ---------- Noise models ----------
def add_gaussian_noise(img, sigma=0.05):
    noisy = img + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 1)

def add_poisson_noise(img, scale=50):
    vals = np.random.poisson(img * scale) / float(scale)
    return np.clip(vals, 0, 1)

def show_with_scalebar(ax, img, title="", cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    add_scale_bar(ax, img.shape)
    ax.set_title(title)
    ax.axis("off")

if __name__ == "__main__":
    N = 512
    gt = generate_ground_truth(N)
    lp = low_pass_filter(gt, cutoff=0.15)
    
    freq = 12
    phases = [0, 2*np.pi/3, 4*np.pi/3]
    angles = [0, 60, 120]   # multiple orientations

    for angle in angles:
        sim_images = []
        patterns = []
        for p in phases:
            pat = illumination_pattern(N, freq_cycles=freq, angle_deg=angle, phase=p)
            patterns.append(pat)
            sim_raw = lp * pat
            sim_images.append(sim_raw)

        # ---------- Display patterns and SIM raw images ----------
        fig, axes = plt.subplots(2, len(phases)+1, figsize=(12, 5))
        fig.suptitle(f"Orientation {angle}°", fontsize=14)

        show_with_scalebar(axes[0,0], gt, f"Ground Truth")
        show_with_scalebar(axes[1,0], lp, f"Low-pass (widefield)")
        for i in range(len(phases)):

            show_with_scalebar(axes[0,i+1], patterns[i], f"Pattern phase {i}")
            show_with_scalebar(axes[1,i+1], sim_images[i], f"Pattern phase {i}")
        plt.tight_layout()

        # ---------- Fourier Transforms ----------
        fig, axes = plt.subplots(2, len(phases)+1, figsize=(12, 6))
        fig.suptitle(f"Fourier domain (orientation {angle}°)", fontsize=14)

        show_fft(gt, axes[0,0], "GT Fourier")
        show_fft(lp, axes[1,0], "Low-pass Fourier")

        for i in range(len(phases)):
            show_fft(sim_images[i], axes[0,i+1], f"SIM raw {i} FFT")
            show_with_scalebar(axes[1,i+1], sim_images[i], f"SIM raw {i} (image)")
        plt.tight_layout()

        # ---------- Low & High frequency split for one phase ----------
        low, high = separate_components(sim_images[0])
        plt.figure(figsize=(10,4))
        plt.suptitle(f"Frequency components (orientation {angle}°)", fontsize=14)
        plt.subplot(1,2,1); plt.imshow(low, cmap='gray'); plt.title("Low-frequency part")
        
        plt.subplot(1,2,2); plt.imshow(high, cmap='gray'); plt.title("High-frequency part")
        
    # ---------- Reconstruction across orientations ----------
    band_shift_norm = 0.12   # matches your separation setting
    recon_sum = np.zeros_like(lp)
    per_angle_recons = []

    for angle in angles:
        sim_images = []
        for p in phases:
            pat = illumination_pattern(N, freq_cycles=freq, angle_deg=angle, phase=p)
            sim_images.append(lp * pat)

        # Reconstruct for this orientation
        recon_img, _ = reconstruct_from_orientation(sim_images, phases, angle,
                                                    band_shift_norm, N, noise_var=1e-3)
        per_angle_recons.append(recon_img)
        recon_sum += recon_img

    # Average across orientations for isotropy
    recon_iso = recon_sum / len(angles)

    # ---------- Display Reconstructions ----------
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    show_with_scalebar(axs[0,0], gt, "Ground Truth")
    show_with_scalebar(axs[0,1], lp, "Low-pass (Widefield)")
    show_with_scalebar(axs[1,0], recon_iso,"Isotropic SIM Reconstruction")
    show_with_scalebar(axs[1,1], np.concatenate(per_angle_recons, axis=1), "Per-Orientation Reconstructions")
    plt.tight_layout()

    # ---------- NOISY Reconstruction ----------
    band_shift_norm = 0.12   # same as before
    recon_sum_noisy = np.zeros_like(lp)
    per_angle_recons_noisy = []

    for angle in angles:
        sim_images_noisy = []
        for p in phases:
            pat = illumination_pattern(N, freq_cycles=freq, angle_deg=angle, phase=p)
            sim_raw = lp * pat
            # --- Add noise here ---
            sim_raw = add_gaussian_noise(sim_raw, sigma=0.05)
            #sim_raw = add_poisson_noise(sim_raw, scale=255)
            sim_images_noisy.append(sim_raw)

        # Reconstruct for this orientation (using noisy data)
        recon_img_noisy, _ = reconstruct_from_orientation(sim_images_noisy, phases, angle,
                                                          band_shift_norm, N, noise_var=1e-3)
        per_angle_recons_noisy.append(recon_img_noisy)
        recon_sum_noisy += recon_img_noisy

    # Average across orientations for isotropy
    recon_iso_noisy = recon_sum_noisy / len(angles)

    # ---------- Display NOISY Reconstructions ----------
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("SIM Reconstruction with Noise", fontsize=14)

    show_with_scalebar(axs[0,0], gt, "Ground Truth")
    show_with_scalebar(axs[0,1], lp, "Low-pass (Widefield)")
    show_with_scalebar(axs[1,0], recon_iso_noisy, "Isotropic SIM Reconstruction (Noisy)")
    show_with_scalebar(axs[1,1], np.concatenate(per_angle_recons_noisy, axis=1), "Per-Orientation Reconstructions")
    plt.tight_layout()

    # ---------- NOISY Fourier Transforms ----------
    for angle in angles:
        sim_images_noisy = []
        patterns = []
        for p in phases:
            pat = illumination_pattern(N, freq_cycles=freq, angle_deg=angle, phase=p)
            patterns.append(pat)
            sim_raw = lp * pat
            # --- Add noise here ---
            sim_raw = add_gaussian_noise(sim_raw, sigma=0.05)
            # sim_raw = add_poisson_noise(sim_raw, scale=255)
            sim_images_noisy.append(sim_raw)

        # Display noisy Fourier transforms
        fig, axes = plt.subplots(2, len(phases)+1, figsize=(12, 6))
        fig.suptitle(f"Fourier domain with Noise (orientation {angle}°)", fontsize=14)

        show_fft(gt, axes[0,0], "GT Fourier")
        show_fft(lp, axes[1,0], "Low-pass Fourier")
        show_with_scalebar(axes[1,i+1], sim_images_noisy[i], f"Noisy SIM raw {i} (image)")

        for i in range(len(phases)):
            show_fft(sim_images_noisy[i], axes[0,i+1], f"Noisy SIM raw {i} FFT")
            show_with_scalebar(axes[1,i+1], sim_images_noisy[i], f"Noisy SIM raw {i} (image)")

        plt.tight_layout()
    plt.show()
