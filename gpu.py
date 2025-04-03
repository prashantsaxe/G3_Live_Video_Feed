import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()

print("Torch version:", torch.__version__)
print("Device selected:", device)

if device == "cuda":
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
elif device == "mps":
    print("Using Apple Silicon MPS backend for GPU acceleration.")
else:
    print("No GPU detected. Running on CPU.")
