import os
import sys

try:
    import onnx
    from onnxsim import simplify
    from onnxconverter_common import float16
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception as e:
    print("Missing packages:", e)
    sys.exit(2)

src = 'models/facenet.onnx'
if not os.path.exists(src):
    print('Source model not found:', src)
    sys.exit(1)

def sz(path):
    return os.path.getsize(path) / (1024*1024)

print(f"Original: {src} -> {sz(src):.2f} MB")

# 1) Simplify
simp = 'models/facenet_simpl.onnx'
try:
    print('Simplifying...')
    model = onnx.load(src)
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, simp)
        print('Simplified saved ->', simp)
    else:
        print('Simplifier reported check=False, skipping save')
except Exception as e:
    print('Simplify failed:', e)

if os.path.exists(simp):
    print(f"Simplified size: {sz(simp):.2f} MB")
else:
    simp = src

# 2) FP16 conversion
fp16 = 'models/facenet_fp16.onnx'
try:
    print('Converting to float16...')
    model = onnx.load(simp)
    model16 = float16.convert_float_to_float16(model)
    onnx.save(model16, fp16)
    print('FP16 saved ->', fp16)
    print(f"FP16 size: {sz(fp16):.2f} MB")
except Exception as e:
    print('FP16 conversion failed:', e)
    fp16 = None

# 3) Dynamic quantization (int8)
quant = 'models/facenet_quant.onnx'
try:
    print('Running dynamic quantization (INT8) on simplified FP32 model...')
    # quantize the simplified (fp32) model - better accuracy
    target_for_quant = simp if simp != src else src
    quantize_dynamic(target_for_quant, quant, weight_type=QuantType.QInt8)
    print('Quantized saved ->', quant)
    print(f"Quantized size: {sz(quant):.2f} MB")
except Exception as e:
    print('Quantization failed:', e)

print('\nSummary:')
print(f"Original: {src} -> {sz(src):.2f} MB")
if os.path.exists(simp):
    print(f"Simplified: {simp} -> {sz(simp):.2f} MB")
if fp16 and os.path.exists(fp16):
    print(f"FP16: {fp16} -> {sz(fp16):.2f} MB")
if os.path.exists(quant):
    print(f"Quantized INT8: {quant} -> {sz(quant):.2f} MB")

print('\nNOTE: After creating smaller models, validate them on a few samples to ensure accuracy has not degraded unacceptably.')