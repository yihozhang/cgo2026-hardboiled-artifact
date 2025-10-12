import numpy as np
from timeit import timeit

import torch
import torch.nn.functional as F

def blur_separable_numpy(input_img):
    blur_x = (input_img[:, :-2] + input_img[:, 1:-1] + input_img[:, 2:]) // 3
    
    blur_y = (blur_x[:-2, :] + blur_x[1:-1, :] + blur_x[2:, :]) // 3
    
    return blur_y

def blur_separable_torch(input_img):
    blur_x = (input_img[:, :-2] + input_img[:, 1:-1] + input_img[:, 2:]) // 3
    
    blur_y = (blur_x[:-2, :] + blur_x[1:-1, :] + blur_x[2:, :]) // 3
    
    if blur_y.device.type == 'cuda':
        torch.cuda.synchronize()

    return blur_y

N = 4322
M = 7688

input_img = np.random.randint(0xfff, size = (N, M)).astype(np.int16)

blur_time = timeit(lambda: blur_separable_numpy(input_img), number=25) / 25 * 1000  # in ms
print(f"blur time numpy {blur_time:.1f}")
print("blur throughput numpy {:.1f} Mpixels/s".format((input_img.shape[0]-2)*(input_img.shape[1]-2) / (blur_time * 1e-3) / 1e6))
numpy_output = blur_separable_numpy(input_img)

input_img = torch.as_tensor(input_img, dtype=torch.int16, device='cuda')
# input_img = torch.as_tensor(input_img, dtype=torch.int16)
blur_time = timeit(lambda: blur_separable_torch(input_img), number=25) / 25 * 1000  # in ms
print(f"blur time torch {blur_time:.1f}")
print("blur throughput torch {:.1f} Mpixels/s".format((input_img.shape[0]-2)*(input_img.shape[1]-2) / (blur_time * 1e-3) / 1e6))
torch_output = blur_separable_torch(input_img)

# move to the same device
torch_output = torch_output.cpu()

assert torch.allclose(torch_output, torch.from_numpy(numpy_output)), "Outputs do not match!"

