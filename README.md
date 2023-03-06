# MetalGPU
Test [Metal.jl](https://github.com/JuliaGPU/Metal.jl) on ARM M-serie GPUs.

## About
Mostly dev sandbox to try out new Metal.jl features on Apple's M2 processors.

## Performance
Running [`diffusion_2D_kp.jl`](scripts/diffusion_2D_kp.jl) on a MacBookAir with M2 chip results in

```
 Perf. memcopy: time (s) = 0.00929, T_eff (GB/s) = 86.7
 Perf. Laplace: time (s) = 0.00974, T_eff (GB/s) = 82.72 (0.95% of memcopy)
```

## Docs
Recent blog posts with features highlights:
- https://juliagpu.org/post/2022-06-24-metal/
- https://juliagpu.org/post/2023-03-03-metal_0.2/

### Notes
Soon, `grid` will become `groups` in kernel launch params to avoid confusion
