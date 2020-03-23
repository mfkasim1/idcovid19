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
        levalues, levecs = torch.eig(A.T, eigenvectors=True)

        # take the maximum eigenvalue and eigenvector
        idxmax = torch.max(evalues[:,0], dim=0)[1].item()
        lidxmax = torch.max(levalues[:,0], dim=0)[1].item()

        assert torch.allclose(evalues[idxmax,0], levalues[lidxmax,0])
        if not torch.allclose(evalues[idxmax,1], evalues[idxmax,1]*0.0):
            raise ValueError("The maximum eigenvalue cannot be a complex number")

        max_eval = evalues[idxmax, 0]
        max_leval = levalues[lidxmax, 0]
        max_evec = evecs[:,idxmax]
        max_levec = levecs[:,lidxmax]

        # make the first element positive
        max_evec = max_evec * torch.sign(max_evec[0])
        max_levec = max_levec * torch.sign(max_levec[0])

        ctx.A = A
        ctx.evals = evalues
        ctx.max_eval = max_eval
        ctx.max_leval = max_leval
        ctx.max_evec = max_evec
        ctx.max_levec = max_levec
        return max_eval, max_evec, max_levec

    @staticmethod
    def backward(ctx, grad_eval, grad_evec, grad_levec):
        # grad_eval: scalar
        # grad_evec: (nr,)
        A = ctx.A
        max_evec = ctx.max_evec
        max_levec = ctx.max_levec
        nr = A.shape[0]
        alpha = 1./torch.dot(max_evec, max_levec)

        # contribution from eigenvector
        g = grad_evec - torch.dot(grad_evec, max_evec) * max_evec
        Ameval = A - torch.eye(nr, dtype=A.dtype, device=A.device) * ctx.max_eval
        A1 = -torch.pinverse(Ameval.T)
        g1 = torch.matmul(A1, g) # (nr,)
        g1 = g1 - torch.dot(g1, max_evec) * max_levec * alpha # (nr,)
        grad_A_evec = torch.ger(g1, max_evec)

        # contribution from left eigenvector
        gl = grad_levec - torch.dot(grad_levec, max_levec) * max_levec
        Amevall = A.T - torch.eye(nr, dtype=A.dtype, device=A.device) * ctx.max_eval
        Al1 = -torch.pinverse(Amevall.T)
        gl1 = torch.matmul(Al1, gl) # (nr,)
        gl1 = gl1 - torch.dot(gl1, max_levec) * max_evec * alpha # (nr,)
        grad_A_levec = torch.ger(gl1, max_levec).T

        # contribution from eigenvalue
        grad_A_eval = torch.ger(max_levec, max_evec) * grad_eval * alpha
        return grad_A_evec + grad_A_eval + grad_A_levec

if __name__ == "__main__":
    from fd import finite_differences

    A = torch.tensor([
        [-6.8094e-02,  3.4940e-01,  3.4940e-01,  0.0000e+00,  0.0000e+00],
        [ 2.8913e-02, -4.2155e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 3.9181e-02,  0.0000e+00, -4.6650e-02,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  4.2155e-02,  0.0000e+00,  1.0000e-08,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  4.6650e-02,  0.0000e+00,  2.0000e-08]]).to(torch.float64).requires_grad_()
    evals, evecs, _ = maxeig.apply(A)
    print(evals, evecs)

    def getloss(A):
        A.requires_grad_()
        with torch.enable_grad():
            evals, evecs, _ = maxeig.apply(A)
            loss1 = (evals**2).sum()
            loss2 = (evecs.abs()**3).sum()
            lossg1 = (torch.autograd.grad(loss1, A, create_graph=True)[0]**2).sum()
            lossg2 = (torch.autograd.grad(loss2, A, create_graph=True)[0]**2).sum()
            lossgg1 = (torch.autograd.grad(lossg1, A, create_graph=True)[0]**2).sum()
            lossgg2 = (torch.autograd.grad(lossg2, A, create_graph=True)[0]**2).sum()
        loss = lossgg2# + loss2
        return loss

    loss = getloss(A)
    Agrad = torch.autograd.grad(loss, A)[0]
    # loss.backward()
    # Agrad = A.grad.data

    with torch.no_grad():
        Afd = finite_differences(getloss, (A,), 0, eps=1e-6)
    print(Agrad)
    print(Afd)
    print(Agrad / Afd - 1)
