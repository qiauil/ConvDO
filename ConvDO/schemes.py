from .helpers import *
from .meta_type import *

class FDScheme():

    def __init__(self, kernel_weights) -> None:
        if not torch.is_tensor(kernel_weights):
            kernel_weights = torch.tensor(kernel_weights)
        self.kernel_weights=kernel_weights
        self.kernel_dx, self.kernel_dy = self.gen_kernel(kernel_weights)
        self.pad = int((kernel_weights.shape[0]-1)/2)

    def gen_kernel(self, kernel: torch.Tensor):
        len_kernel = len(kernel)
        dx = torch.zeros((len_kernel, len_kernel))
        dx[int((len_kernel-1)/2)] = kernel
        return dx.unsqueeze(0).unsqueeze(0), torch.flip(dx.T, dims=(0,)).unsqueeze(0).unsqueeze(0)


CENTRAL_INTERPOLATION_SCHEMES = {
    2: FDScheme([-1/2, 0, 1/2]),
    4: FDScheme([1/12, -2/3, 0, 2/3, -1/12]),
    6: FDScheme([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
    8: FDScheme([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
}

CENTRAL_LAPLACIAN_SCHEMES = {
    2: FDScheme([1, -2, 1]),
    4: FDScheme([-1/12, 4/3, -5/2, 4/3, -1/12]),
    6: FDScheme([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
    8: FDScheme([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),
}
