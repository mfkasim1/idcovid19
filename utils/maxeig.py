import torch

class maxeig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # A: (nr, nr)
        # normalize the shape to be batched
        nr = A.shape[0]
        assert A.shape[0] == A.shape[1], "The matrix must be a square matrix"

        # get the right and left eigendecomposition
        evalues, evecs = torch.eig(A, eigenvectors=True)
        levalues, levecs = torch.eig(A.transpose(-2,-1), eigenvectors=True)

        # take the maximum eigenvalue and eigenvector
        idxmax = torch.max(evalues[:,0], dim=0)[1].item()
        lidxmax = torch.max(levalues[:,0], dim=0)[1].item()

        assert torch.allclose(evalues[idxmax,0], levalues[lidxmax,0])
        if not torch.allclose(evalues[idxmax,1], evalues[idxmax,1]*0.0):
            raise ValueError("The maximum eigenvalue cannot be a complex number")

        max_eval = evalues[idxmax, 0]
        max_evec = evecs[:,idxmax]
        max_levec = levecs[:,lidxmax]

        ctx.A = A
        ctx.max_eval = max_eval
        ctx.max_evec = max_evec
        ctx.max_levec = max_levec
        return max_eval, max_evec

    @staticmethod
    def backward(ctx, grad_eval, grad_evec):
        # grad_eval: scalar
        # grad_evec: (nr,)
        A = ctx.A
        max_evec = ctx.max_evec
        max_levec = ctx.max_levec
        nr = A.shape[0]
        ytx = (max_evec*max_levec).sum()

        # contribution from eigenvector
        g = grad_evec - (grad_evec*max_levec).sum() * max_evec / ytx
        Ameval = A - torch.eye(nr, dtype=A.dtype, device=A.device) * ctx.max_eval
        A1 = -torch.pinverse(Ameval).transpose(-2,-1)
        g1 = torch.matmul(A1, g) # (nr,)
        g1y = g1 - (g1*max_evec).sum() * max_levec / ytx # (nr,)
        grad_A_evec = torch.ger(g1y, max_evec)

        # contribution from eigenvalue
        grad_A_eval = torch.ger(max_levec, max_evec) * grad_eval / ytx

        return grad_A_evec + grad_A_eval

if __name__ == "__main__":
    from fd import finite_differences

    A = torch.tensor([
         [0.7, 0.2, 0.1],
         [0.4, 0.8, 0.1],
         [0.2, 3.0, 2.0],
    ]).to(torch.float64).requires_grad_()
    evals, evecs = maxeig.apply(A)
    print(evals, evecs)

    def getloss(A):
        evals, evecs = maxeig.apply(A)
        loss1 = (evals**2).sum()
        loss2 = (evecs.abs()**3).sum()
        loss = loss2# + loss2
        return loss

    loss = getloss(A)
    loss.backward()
    Agrad = A.grad.data

    with torch.no_grad():
        Afd = finite_differences(getloss, (A,), 0, eps=1e-6)
    print(Agrad)
    print(Afd)
    print(Agrad / Afd - 1)
