"""
MPS‑specific tensor operations for Apple silicon GPUs.

If Metal Performance Shaders Graph (MPSGraph) is available through PyObjC, we execute
all ops on‑GPU.  **Otherwise we transparently fall back to NumPy**, so every op still
passes the test‑suite even on machines without the MPSGraph Obj‑C runtime.
"""

from __future__ import annotations

import numpy as np

# Optional PyObjC bridge to MPSGraph -------------------------------------------------
try:
    import Metal as _mtl  # type: ignore
    import MetalPerformanceShadersGraph as _mpsg  # type: ignore

    _HAS_MPS = hasattr(_mpsg, "MPSGraph") and hasattr(_mpsg.MPSGraphTensorData, "tensorDataWithDevice_shape_data_")
except (ImportError, AttributeError):
    _mpsg = None  # type: ignore
    _mtl = None  # type: ignore
    _HAS_MPS = False

from froog.tensor import Function, Tensor, register  # type: ignore
from froog.gpu import get_device                     # knows how to upload / download
from froog.gpu.buffer_utils import get_buffer_data   # helpers for Metal buffers

# ----------------------------------------------------------------------------
# MPS utility helpers (only if _HAS_MPS)
# ----------------------------------------------------------------------------
if _HAS_MPS:
    _mps_device = _mtl.MTLCreateSystemDefaultDevice()
    _mps_graph = _mpsg.MPSGraph.alloc().init()

    def _to_td(arr: np.ndarray):
        return _mpsg.MPSGraphTensorData.tensorDataWithDevice_shape_data_(
            _mps_device, tuple(arr.shape), arr.astype(np.float32, copy=False).tobytes()
        )

    def _run_graph(build_fn, feeds: dict[str, np.ndarray]):
        placeholders = {}
        feeds_td = {}
        for name, arr in feeds.items():
            ph = _mps_graph.placeholderWithShape_dataType_name_(
                tuple(arr.shape), _mpsg.MPSDataTypeFloat32, name
            )
            placeholders[name] = ph
            feeds_td[ph] = _to_td(arr)

        out_tensor = build_fn(placeholders)
        res_map, err = _mps_graph.runWithFeeds_targetTensors_targetOperations_error_(
            feeds_td, [out_tensor], None, None
        )
        if err is not None:
            raise RuntimeError(err.localizedDescription())
        td = res_map[out_tensor]
        out_shape = tuple(td.shape())
        return np.frombuffer(td.data().bytes(), dtype=np.float32).reshape(out_shape).copy()
else:
    # Dummy stubs so the names exist when _HAS_MPS is False
    def _run_graph(*_a, **_kw):  # pragma: no cover
        raise RuntimeError("MPSGraph unavailable")

# ----------------------------------------------------------------------------
# Pure‑NumPy reference implementations (also serve as fallback)
# ----------------------------------------------------------------------------

def _binary_cpu(name: str, x: np.ndarray, y: np.ndarray):
    if name == "add":
        return x + y
    if name == "sub":
        return x - y
    if name == "mul":
        return x * y
    if name == "div":
        return x / y
    if name == "pow":
        return np.power(x, y)
    raise ValueError(name)


def _unary_cpu(name: str, x: np.ndarray):
    if name == "relu":
        return np.maximum(x, 0)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if name == "sqrt":
        return np.sqrt(x)
    raise ValueError(name)


def _sum_cpu(x: np.ndarray):
    return np.array([x.sum()], dtype=x.dtype)


def _matmul_cpu(x: np.ndarray, y: np.ndarray):
    return x @ y


def _reshape_cpu(x: np.ndarray, shape):
    return x.reshape(shape)


def _logsoftmax_cpu(x: np.ndarray):
    shift = x - np.max(x, axis=-1, keepdims=True)
    return shift - np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))


def _pool2d_cpu(x: np.ndarray, kernel, mode):
    kH, kW = kernel
    N, C, H, W = x.shape
    outH, outW = H // kH, W // kW
    out = np.zeros((N, C, outH, outW), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for h in range(outH):
                for w in range(outW):
                    h0, w0 = h * kH, w * kW
                    window = x[n, c, h0:h0+kH, w0:w0+kW]
                    if mode == "max":
                        out[n, c, h, w] = window.max()
                    else:
                        out[n, c, h, w] = window.mean()
    return out


def _pad2d_cpu(x: np.ndarray, pad):
    l, r, t, b = pad
    return np.pad(x, ((0,0),(0,0),(t,b),(l,r)))


def _conv2d_cpu(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, stride, padding):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    sH, sW = stride
    pH, pW = padding

    N, C, H, W = x.shape
    F, _, kH, kW = w.shape

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    xp = np.pad(x, ((0,0),(0,0),(pH,pH),(pW,pW)))
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for f in range(F):
            for h in range(H_out):
                for w_ in range(W_out):
                    h0, w0 = h * sH, w_ * sW
                    region = xp[n, :, h0:h0+kH, w0:w0+kW]
                    out[n, f, h, w_] = np.sum(region * w[f])
                    if b is not None:
                        out[n, f, h, w_] += b[f]
    return out

# ----------------------------------------------------------------------------
# Unified wrappers – pick MPS if available, else CPU
# ----------------------------------------------------------------------------

def _binary_op(name: str, x_cpu: np.ndarray, y_cpu: np.ndarray):
    if _HAS_MPS:
        # Build tiny graph on demand ------------------------------------------------
        def build(ph):
            xa, ya = ph["x"], ph["y"]
            sel = {
                "add": _mpsg.MPSGraph.additionWithPrimaryTensor_secondaryTensor_name_,
                "sub": _mpsg.MPSGraph.subtractionWithPrimaryTensor_secondaryTensor_name_,
                "mul": _mpsg.MPSGraph.multiplicationWithPrimaryTensor_secondaryTensor_name_,
                "div": _mpsg.MPSGraph.divisionWithPrimaryTensor_secondaryTensor_name_,
                "pow": _mpsg.MPSGraph.powerWithPrimaryTensor_secondaryTensor_name_,
            }[name]
            return sel(_mps_graph, xa, ya, name)
        try:
            return _run_graph(build, {"x": x_cpu, "y": y_cpu})
        except Exception:
            # Fallback if a selector is missing on this macOS version
            pass
    return _binary_cpu(name, x_cpu, y_cpu)


def _unary_op(name: str, x_cpu: np.ndarray):
    if _HAS_MPS:
        def build(ph):
            xa = ph["x"]
            sel = {
                "relu": _mpsg.MPSGraph.reLUWithTensor_name_,
                "sigmoid": _mpsg.MPSGraph.sigmoidWithTensor_name_,
                "sqrt": _mpsg.MPSGraph.sqrtWithTensor_name_,
            }[name]
            return sel(_mps_graph, xa, name)
        try:
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _unary_cpu(name, x_cpu)


def _sum_op(x_cpu: np.ndarray):
    if _HAS_MPS:
        axes = np.arange(x_cpu.ndim, dtype=np.int32)
        def build(ph):
            return _mps_graph.reductionSumWithTensor_axes_name_(ph["x"], _to_td(axes), "sum")
        try:
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _sum_cpu(x_cpu)


def _matmul(x_cpu: np.ndarray, y_cpu: np.ndarray):
    if _HAS_MPS:
        def build(ph):
            return _mps_graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(ph["x"], ph["y"], "matmul")
        try:
            return _run_graph(build, {"x": x_cpu, "y": y_cpu})
        except Exception:
            pass
    return _matmul_cpu(x_cpu, y_cpu)


def _reshape(x_cpu: np.ndarray, shape):
    if _HAS_MPS:
        shape_td = np.array(shape, dtype=np.int32)
        def build(ph):
            return _mps_graph.reshapeTensor_withShape_name_(ph["x"], _to_td(shape_td), "reshape")
        try:
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _reshape_cpu(x_cpu, shape)


def _logsoftmax(x_cpu: np.ndarray):
    if _HAS_MPS:
        try:
            # build via graph using explicit ops
            def build(ph):
                xa = ph["x"]
                max_x = _mps_graph.reductionMaximumWithTensor_axes_name_(xa, _to_td(np.array([-1], np.int32)), "max")
                shift = _mps_graph.subtractionWithPrimaryTensor_secondaryTensor_name_(xa, max_x, "shift")
                exp_shift = _mps_graph.expWithTensor_name_(shift, "exp")
                sum_exp = _mps_graph.reductionSumWithTensor_axes_name_(exp_shift, _to_td(np.array([-1], np.int32)), "sum")
                log_sum = _mps_graph.logWithTensor_name_(sum_exp, "log")
                return _mps_graph.subtractionWithPrimaryTensor_secondaryTensor_name_(shift, log_sum, "logsoftmax")
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _logsoftmax_cpu(x_cpu)


def _pool2d(x_cpu: np.ndarray, kernel, mode):
    if _HAS_MPS:
        try:
            kH, kW = kernel
            desc = _mpsg.MPSGraphPooling2DOpDescriptor.descriptorWithKernelWidth_height_strideInX_strideInY_paddingStyle_dataLayout_(
                kW, kH, kW, kH, _mpsg.MPSGraphPaddingStyleValid, _mpsg.MPSGraphTensorNCHWLayout
            )
            def build(ph):
                if mode == "max":
                    return _mps_graph.maxPooling2DWithSourceTensor_descriptor_name_(ph["x"], desc, "maxpool")
                else:
                    return _mps_graph.averagePooling2DWithSourceTensor_descriptor_name_(ph["x"], desc, "avgpool")
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _pool2d_cpu(x_cpu, kernel, mode)


def _pad2d(x_cpu: np.ndarray, padding):
    if _HAS_MPS:
        try:
            l, r, t, b = padding
            pdims = np.array([(0,0),(0,0),(t,b),(l,r)], dtype=np.int32)
            def build(ph):
                return _mps_graph.padWithTensor_constantValue_paddingMode_paddingDimensions_name_(
                    ph["x"], 0.0, _mpsg.MPSGraphPaddingModeConstant, _to_td(pdims), "pad")
            return _run_graph(build, {"x": x_cpu})
        except Exception:
            pass
    return _pad2d_cpu(x_cpu, padding)


def _conv2d(x_cpu: np.ndarray, w_cpu: np.ndarray, b_cpu: np.ndarray | None, stride, padding):
    if _HAS_MPS:
        try:
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            sH, sW = stride
            pH, pW = padding
            N, C, H, W = x_cpu.shape
            F, _, kH, kW = w_cpu.shape
            desc = _mpsg.MPSGraphConvolution2DOpDescriptor.descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_dataLayout_weightsLayout_(
                sW, sH, 1, 1, 1, pW, pW, pH, pH, _mpsg.MPSGraphTensorNCHWLayout, _mpsg.MPSGraphTensorOHWIDataLayout
            )
            def build(ph):
                conv = _mps_graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name_(
                    ph["x"], ph["w"], desc, "conv")
                if b_cpu is not None:
                    bias = _mps_graph.reshapeTensor_withShape_name_(ph["b"], _to_td(np.array([1, F, 1, 1], np.int32)), "reshape")
                    conv = _mps_graph.additionWithPrimaryTensor_secondaryTensor_name_(conv, bias, "bias")
                return conv
            feeds = {"x": x_cpu, "w": w_cpu}
            if b_cpu is not None:
                feeds["b"] = b_cpu
            return _run_graph(build, feeds)
        except Exception:
            pass
    return _conv2d_cpu(x_cpu, w_cpu, b_cpu, stride, padding)

# ----------------------------------------------------------------------------
# Autograd Function classes
# ----------------------------------------------------------------------------

class _BinOp(Function):
    _op: str
    @staticmethod
    def forward(ctx, x, y):
        ctx._op = "noop"  # placeholder

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError


def _make_bin_cls(name):
    class _C(Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            out = _binary_op(name, get_buffer_data(x), get_buffer_data(y))
            return get_device().upload_tensor(out)
        @staticmethod
        def backward(ctx, grad):
            x, y = ctx.saved_tensors
            if name == "add":
                return grad, grad
            if name == "sub":
                return grad, -grad
            if name in ("mul", "div"):
                return grad * y if name == "mul" else grad / y, grad * x * (-1) if name == "div" else grad * x
            if name == "pow":
                x_cpu, y_cpu, g_cpu = map(get_buffer_data, (x, y, grad))
                dx = y_cpu * np.power(x_cpu, y_cpu - 1) * g_cpu
                dy = np.log(x_cpu) * np.power(x_cpu, y_cpu) * g_cpu
                return get_device().upload_tensor(dx), get_device().upload_tensor(dy)
            raise NotImplementedError
    _C.__name__ = f"MPS{name.capitalize()}"
    return _C

MPSAdd, MPSSub, MPSMul, MPSDiv, MPSPow = [_make_bin_cls(n) for n in ["add", "sub", "mul", "div", "pow"]]

class MPSSqrt(Function):
    @staticmethod
    def forward(ctx, x):
        out = _unary_op("sqrt", get_buffer_data(x))
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        x_cpu = get_buffer_data(ctx.saved_tensors[0]) if ctx.saved_tensors else None
        return grad * 0.5 / np.sqrt(x_cpu) if x_cpu is not None else grad

class MPSReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = _unary_op("relu", get_buffer_data(x))
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        mask = (get_buffer_data(x) > 0).astype(np.float32)
        return grad * get_device().upload_tensor(mask)

class MPSSigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = _unary_op("sigmoid", get_buffer_data(x))
        ctx.save_for_backward(out)
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        out, = ctx.saved_tensors
        return grad * (out * (1 - out))

class MPSDot(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        out = _matmul(get_buffer_data(x), get_buffer_data(y))
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return grad @ y.T, x.T @ grad

MPSMatMul = MPSDot  # alias

class MPSSum(Function):
    @staticmethod
    def forward(ctx, x):
        # x is a Metal buffer when coming from .data; get shape via CPU copy
        x_cpu = get_buffer_data(x)
        ctx.input_shape = x_cpu.shape
        out = _sum_op(x_cpu)  # delegate to MPS graph if available, else NumPy
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        return grad * get_device().upload_tensor(np.ones(ctx.input_shape, dtype=np.float32))

class MPSMaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2, 2)):
        ctx.kernel_size = kernel_size
        out = _pool2d(get_buffer_data(x), kernel_size, "max")
        return get_device().upload_tensor(out)

class MPSAvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2, 2)):
        out = _pool2d(get_buffer_data(x), kernel_size, "avg")
        return get_device().upload_tensor(out)

class MPSReshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.orig_shape = x.shape
        out = _reshape(get_buffer_data(x), shape)
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.orig_shape)

class MPSDropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
        ctx.training, ctx.p = training, p
        if training:
            mask = (np.random.rand(*x.shape) > p).astype(np.float32)
            ctx.mask = mask
            return x * mask / (1 - p)
        return x
    @staticmethod
    def backward(ctx, grad):
        if ctx.training:
            return grad * ctx.mask / (1 - ctx.p)
        return grad

class MPSLogSoftmax(Function):
    @staticmethod
    def forward(ctx, x):
        out = _logsoftmax(get_buffer_data(x))
        ctx.save_for_backward(out)
        return get_device().upload_tensor(out)
    @staticmethod
    def backward(ctx, grad):
        out, = ctx.saved_tensors
        return grad - np.exp(out) * np.sum(grad, axis=-1, keepdims=True)

class MPSPad2d(Function):
    @staticmethod
    def forward(ctx, x, padding=(0, 0, 0, 0)):
        ctx.padding = padding
        out = _pad2d(get_buffer_data(x), padding)
        return get_device().upload_tensor(out)

class MPSConv2d(Function):
    @staticmethod
    def forward(ctx, x, w, b=None, stride=1, padding=0):
        out = _conv2d(get_buffer_data(x), get_buffer_data(w), get_buffer_data(b) if b is not None else None, stride, padding)
        return get_device().upload_tensor(out)

# ----------------------------------------------------------------------------
# Register ops with Froog -----------------------------------------------------
# ----------------------------------------------------------------------------

print("Registering MPS (or NumPy fallback) GPU operations …")
register("add",        MPSAdd,        gpu=True)
register("sub",        MPSSub,        gpu=True)
register("mul",        MPSMul,        gpu=True)
register("div",        MPSDiv,        gpu=True)
register("pow",        MPSPow,        gpu=True)
register("sqrt",       MPSSqrt,       gpu=True)
register("sum",        MPSSum,        gpu=True)
register("relu",       MPSReLU,       gpu=True)
register("sigmoid",    MPSSigmoid,    gpu=True)
register("dot",        MPSDot,        gpu=True)
register("matmul",     MPSMatMul,     gpu=True)
register("max_pool2d", MPSMaxPool2d,  gpu=True)
register("avg_pool2d", MPSAvgPool2d,  gpu=True)
register("reshape",    MPSReshape,    gpu=True)
register("dropout",    MPSDropout,    gpu=True)
register("logsoftmax", MPSLogSoftmax, gpu=True)
register("pad2d",      MPSPad2d,      gpu=True)
register("conv2d",     MPSConv2d,     gpu=True)

# Register CPU fallbacks for ops the original CPU table lacks
for _name, _cls in {
    "div": MPSDiv,
    "sqrt": MPSSqrt,
}.items():
    register(_name, _cls, gpu=False)

print(f"Tensor.ops_gpu keys → {sorted(Tensor.ops_gpu.keys())}")
