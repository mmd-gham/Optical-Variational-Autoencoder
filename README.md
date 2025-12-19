---

## 📘 Optical Variational Autoencoder (MNIST)

This project implements a **generative optical neural network** using Fourier-optics-based layers that mimic a *4f optical system*. In such systems, convolution operations can be performed by modulating the Fourier spectrum of an image using phase masks. Simulations use FFT to approximate this behavior in software.

### Convolution Theorem

A core property of the Fourier transform:

```
Fourier{ f(x,y) * h(x,y) } = Fourier{f(x,y)} · Fourier{h(x,y)}
```

This means **convolution in the spatial domain** is equivalent to **point-wise multiplication in the frequency domain**. Free-space optical systems exploit this by performing Fourier transforms with lenses, multiplying by a phase mask, and then inverse transforming back to spatial intensity.([MDPI][1])

### Model Highlights

* Optical layers simulate a 4f convolution (FFT → phase mask → IFFT → intensity)
* Variational Autoencoder (VAE) architecture for generative modeling
* Phase masks are **trainable** and regularized for smoothness
* Optical power is normalized per channel to mimic physical energy conservation

### Loss Function

```
Total Loss = Reconstruction Loss + KL Divergence + Phase Smoothness
```

Where:

```
Reconstruction Loss   = MSE(original, reconstructed)
KL Divergence         = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
Phase Smoothness      = mean(|∇x θ|) + mean(|∇y θ|)
```

### Usage

Train the model:

```bash
python train.py
```

Evaluate and visualize:

```bash
python evaluate.py
```

---
