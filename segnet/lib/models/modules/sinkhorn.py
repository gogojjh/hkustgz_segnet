import torch
import torch.nn.functional as F


def sinkhorn(M, r=1.0, c=1.0, lamda=0.005, epsilon=1e-8):
    """
    M: similarity matrix (mls/cosine larger -> similar),
    corresponding cost matrix is -M.
    """
    M = torch.exp(M / lamda).t()
    n, m = M.shape  # [num_proto, pt_num]
    sum_M = torch.sum(M)
    M /= sum_M  # normalize the cost matrix
    v = torch.ones(m).cuda()
    i = 0
    u = torch.ones(n).cuda()
    uprev = u * 2

    while torch.max(torch.abs(u - uprev)) > epsilon:
        uprev = u
        u = r / torch.matmul(M, v)  # v: [m]
        v = c / torch.matmul(M.t(), u)  # u: [n]
        i += 1
    # Log.info('sinkhorn iter: {}'.format(i))s
    M = torch.diag(u) @ M @ torch.diag(v)

    M = M.t()
    indexes = torch.argmax(M, dim=1)
    M = F.gumbel_softmax(M, tau=0.5, hard=True)

    return M, indexes


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    """
    out: similarity matrix [n num_proto]
    thres: convergence parameter

    return: L (mappping matrix)
    """
    L = torch.exp(out / epsilon).t()  # K x B [num_proto, pt_num]
    B = L.shape[1]  # num of points in a batch
    K = L.shape[0]  # num_prototype

    # make the mat element are larger than 0
    sum_L = torch.sum(L)
    L /= sum_L  # mu_s, mu_t constraint

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per component must be 1/K
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        # normalize each column: total weight per samle must be 1/B
        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B  # the columns must sum to 1 so that L is an assignment
    L = L.t()

    # select index of largest proto in this cls
    indexs = torch.argmax(L, dim=1)
    L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()

    # use gumbel_softmax to replace argmax in backpropagation
    # argmax: y = argmax(delta)
    # y = argmax(log(delta) + G): G (gumbel distribution) = -log(-log(delta))
    # L = F.gumbel_softmax(L, tau=0.5, hard=True)  # make elements 0 or 1

    return L, indexs


def distributed_greenkhorn(out, sinkhorn_iterations=100, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    K = L.shape[0]
    B = L.shape[1]

    sum_L = torch.sum(L)
    L /= sum_L

    r = torch.ones((K,), dtype=L.dtype).to(L.device) / K
    c = torch.ones((B,), dtype=L.dtype).to(L.device) / B

    r_sum = torch.sum(L, axis=1)
    c_sum = torch.sum(L, axis=0)

    r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
    c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    for _ in range(sinkhorn_iterations):
        i = torch.argmax(r_gain)
        j = torch.argmax(c_gain)
        r_gain_max = r_gain[i]
        c_gain_max = c_gain[j]

        if r_gain_max > c_gain_max:
            scaling = r[i] / r_sum[i]
            old_row = L[i, :]
            new_row = old_row * scaling
            L[i, :] = new_row

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)
        else:
            scaling = c[j] / c_sum[j]
            old_col = L[:, j]
            new_col = old_col * scaling
            L[:, j] = new_col

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    L = L.t()

    indexs = torch.argmax(L, dim=1)  # ! which protoype in this class to select
    G = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


def pot_sinkhorn(out):
    out = out.t()
    sum_out = torch.sum(out)
    out /= sum_out
    m, n = out.shape[0], out.shape[1]
    a = torch.ones((m)).cuda()
    b = (torch.ones((n)) * n / m).cuda()
    result = ot.sinkhorn(a=a, b=b, M=out, reg=50)

    return result
