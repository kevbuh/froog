# Environment Variables

| Variable | Purpose |
|----------|---------|
| WARNING=1 | Display warnings when tensor data isn't float32 (needed for numerical jacobian) |
| DEBUG=1 | Allow repeated warnings (don't suppress duplicates) |
| GPU=1 | Enable GPU acceleration via OpenCL |
| VIZ=1 | Enable visualization in EfficientNet model |
| CI=1 | Disable progress bars in tests for CI environments |

Multiple variables can be used together: `WARNING=1 DEBUG=1 GPU=1 python your_script.py` 