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


if __name__ == "__main__":
    N = 512
    gt = generate_ground_truth(N)
    # low-pass the GT once (the "widefield" baseline)
    lp = low_pass_filter(gt, cutoff=0.15)

    # choose grating parameters
    freq = 12            # cycles across the whole image (adjust by eye)
    angle = 0.0          # degrees (0 = stripes along x)
    phases = [0, 2*np.pi/3, 4*np.pi/3]   # three-phase SIM

    sim_images = []
    patterns = []
    for p in phases:
        pat = illumination_pattern(N, freq_cycles=freq, angle_deg=angle, phase=p)
        patterns.append(pat)
        sim_raw = lp * pat           # multiply low-passed GT with grating
        sim_images.append(sim_raw)

    # display
    fig, axes = plt.subplots(2, len(phases)+1, figsize=(12, 5))

    axes[0,0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title("Ground Truth")
    axes[0,0].axis('off')

    axes[1,0].imshow(lp, cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title("Low-pass (widefield)")
    axes[1,0].axis('off')

    for i in range(len(phases)):
        axes[0,i+1].imshow(patterns[i], cmap='gray', vmin=0, vmax=1)
        axes[0,i+1].set_title(f"Pattern (phase {i})")
        axes[0,i+1].axis('off')

        axes[1,i+1].imshow(sim_images[i], cmap='gray', vmin=0, vmax=1)
        axes[1,i+1].set_title(f"SIM raw (phase {i})")
        axes[1,i+1].axis('off')

    plt.tight_layout()
    plt.show()