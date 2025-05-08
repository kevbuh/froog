"""
froog: fast real-time optimization of gradients, a compact tensor library.

Environment Variables:
- WARNING=1: Display warnings when tensor data isn't float32
- DEBUG=1: Allow repeated warnings
- GPU=1: Enable GPU acceleration
- VIZ=1: Enable visualization in EfficientNet
- CI=1: Disable progress bars in tests

See docs: https://github.com/kevbuh/froog/blob/main/docs/env.md
"""

import froog.optim
import froog.tensor
import froog.utils
import froog.gpu.gpu_utils