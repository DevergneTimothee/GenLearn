from mlcolvar.core.transform.descriptors import TorsionalAngle

from mlcolvar.core.transform import Transform
import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward

__all__ = ["KabschTransform"]


import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
class KabschTransform(Transform):
    """
    Torsional angle defined by a set of 4 atoms from their positions
    """
    

    def __init__(self, 
                n_atoms,
                 ref_X) -> torch.Tensor:

        super().__init__(in_features=int(n_atoms*3), out_features=n_atoms*3)

        ref_X = ref_X.reshape(-1,3)
        ref_C = ref_X.mean(dim = 0)
        ref_X = (ref_X - ref_C.unsqueeze(0))
        self.ref_X = torch.nn.Parameter(ref_X, requires_grad=False)

        
    def forward(self, x):
        reshape=False
        if len(x.shape) == 1:
           # If not, add a batch dimension
            reshape = True
            x = x.unsqueeze(0)
        x = x.reshape(x.size(0),-1,3)
        x_c = torch.mean(x, 1, True)
        # translation
        x_notran = x - x_c 

        xtmp = x_notran.permute((0,2,1))
        
        prod = torch.matmul(xtmp, self.ref_X) # batched matrix multiplication, output dimension: traj_length x 3 x 3
        u, s, vh = torch.linalg.svd(prod)
        diag_mat = torch.diag(torch.ones(3)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device, dtype=u.dtype)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)
        aligned_x = torch.matmul(x-x_c, rotate_mat).reshape(x.size(0),-1)
        return aligned_x