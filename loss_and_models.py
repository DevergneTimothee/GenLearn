import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward

__all__ = ["Generator"]


import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward

__all__ = ["Generator"]


class Generator(BaseCV, lightning.LightningModule):

    BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        gamma: float = 10000,
        cell: float = None,
        friction = None,
        options: dict = None,
        **kwargs,
    ):

        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(self.forward,
                                     eta=eta,
                                     gamma=gamma,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn = torch.nn.ModuleList([FeedForward(layers, **options[o]) for idx in range(r)])


    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):
        return torch.cat([nn(x) for nn in self.nn], dim=1)
    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))
        x.requires_grad = True

        weights = train_batch["weights"]

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights 
            )
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss
class GeneratorLoss(torch.nn.Module):
  def __init__(self, model, eta, cell, friction, gamma, n_cvs):
    super().__init__()
    self.model = model
    self.eta = eta
    self.friction = friction
    self.lambdas = torch.nn.Parameter(10*torch.randn(n_cvs), requires_grad=True)
    self.gamma = gamma
    self.cell= cell
    print(self.cell)
  def compute_covariance(self,X,weights, centering=False):
    n = X.size(0)
    pre_factor = n / (n - 1)
    mean = weights.mean()
    if X.ndim == 2:
        return   pre_factor * (torch.einsum("ij,ik,i->jk",X,X,weights/mean)/n )#(X.T @ X / n - mean @ mean.T)
    else:
        return pre_factor * (torch.einsum("ijk,ilk,i->jl",X,X,weights/mean) / n)
  def get_parameter_dict(self,model):
    return dict(model.named_parameters()) 
  def forward(self, data, output, weights):
    lambdas = self.lambdas**2
    diag_lamb = torch.diag(lambdas)
    #sorted_lambdas = lambdas[torch.argsort(lambdas)]
    r = output.shape[1]
    sample_size = output.shape[0]//2
    weights_X, weights_Y = weights[:sample_size], weights[sample_size:]
    #gradient_X = torch.autograd.grad(psi_X,X,grad_outputs=torch.ones_like(psi_X,requires_grad=True),retain_graph=True,create_graph=True)[0] * friction.unsqueeze(0)
    #gradient_Y = torch.autograd.grad(psi_Y,Y,grad_outputs=torch.ones_like(psi_Y,requires_grad=True),retain_graph=True,create_graph=True)[0] * friction.unsqueeze(0)
    gradient = torch.stack([torch.autograd.grad(outputs=output[:,idx].sum(), inputs=data, retain_graph=True, create_graph=True)[0] for idx in range(r)], dim=2).swapaxes(2,1) * self.friction
    gradient = gradient.reshape(weights.shape[0],output.shape[1],-1)
    if self.cell is not None:
       gradient /= (self.cell)
    gradient_X, gradient_Y = gradient[:sample_size], gradient[sample_size:]
    psi_X, psi_Y = output[:sample_size], output[sample_size:]
    #gradient_Y = torch.stack([torch.autograd.grad(outputs=psi_Y[:,idx].sum(), inputs=Y, retain_graph=True, create_graph=True)[0].reshape((-1,d)) for idx in range(r)], dim=2).swapaxes(2,1) * friction.unsqueeze(0).unsqueeze(1)


    
    cov_X =  self.compute_covariance(psi_X , weights_X, centering=True) 
    cov_Y =  self.compute_covariance(psi_Y , weights_Y, centering=True)


    dcov_X =  self.compute_covariance(gradient_X , weights_X) 
    dcov_Y =  self.compute_covariance(gradient_Y , weights_Y) 
    
    W1 = (self.eta *cov_X + dcov_X ) @ diag_lamb
    W2 = (self.eta *cov_Y + dcov_Y) @ diag_lamb
    
    mean_weights_x = weights_X.mean()
    mean_weights_y = weights_Y.mean()
    loss_ef = torch.trace( ((cov_X@diag_lamb) @ W2 + (cov_Y@diag_lamb)@W1)/2 - cov_X@diag_lamb - cov_Y@diag_lamb)
        # Compute loss_ortho
    loss_ortho = self.gamma * (torch.trace((torch.eye(output.shape[1], device=output.device) - cov_X).T @ (torch.eye(output.shape[1], device=output.device) - cov_X)))
    #loss_ortho = penalty
    loss = loss_ef + loss_ortho#loss_ortho
    return loss, loss_ef, loss_ortho
def compute_covariance(X,weights, centering=False):
    n = X.size(0)
    pre_factor = 1.0
    if X.ndim == 2:
        if centering==True:
          mean =  torch.einsum("ij,i->j",X,weights) / n
        else:
          mean = torch.zeros_like(X[0])
        return   pre_factor * (torch.einsum("ij,ik,i->jk",X,X,weights)/n)#(X.T @ X / n - mean @ mean.T)
    else:
        return pre_factor * (torch.einsum("ijk,ilk,i->jl",X,X,weights) / n)

def compute_eigenfunctions(model, dataset, friction, eta, r):

  #friction=friction.to("cuda")
  dataset["data"].requires_grad = True
  X= dataset["data"]
  d=dataset["data"].shape[1]
  psi_X = model(X)
  gradient_X = torch.stack([torch.autograd.grad(outputs=psi_X[:,idx].sum(), inputs=X, retain_graph=True, create_graph=True)[0].reshape((-1,d)) for idx in range(r)], dim=2).swapaxes(2,1) * torch.sqrt(friction)

  weights_X = dataset["weights"]
  cov_X =  compute_covariance(psi_X, weights_X, centering=True) 

  

  dcov_X =  compute_covariance(gradient_X, weights_X) 
  W = eta *cov_X + dcov_X

  operator = torch.linalg.inv(W + 1e-5*torch.eye(psi_X.size(1),device=psi_X.device))@cov_X
  evals, evecs = torch.linalg.eig(operator)
  return evals.detach(), evecs.detach()