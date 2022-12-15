import torch
import torch.nn.functional as F
import ot


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    """ 
    out: similarity matrix

    return: L (mappping matrix)
    """
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]  # num of points in a batch
    K = L.shape[0]  # num_prototype

    # make the mat element equal to 0 or 1
    sum_L = torch.sum(L)
    L /= sum_L  # mu_s, mu_t constraint

    # to compute L (cost matrix) while updating mu and gamma
    for _ in range(sinkhorn_iterations):
        # to keep mat element equal to 0/1
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    # use gumbel_softmax to replace argmax in backpropagation
    # argmax: y = argmax(delta)
    # y = argmax(log(delta) + G): G (gumbel distribution) = -log(-log(delta))
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


def sinkhorn(M, max_iter=3, lamda=0.05, mu_s=1, mu_t=1, thres=1e-1):
    """ 
    M: cost matrix, ivnerse of similarity matrix
    """
    K = torch.exp(-M / lamda)
    t = 0
    beta = 1.0

    for _ in range(max_iter):
        alfa1 = alfa
        alfa = mu_s / (K * beta)
        beta = mu_t / (K.t() * alfa)

        # check for threshold requirement
        delta_alfa = (alfa - alfa1).sum(-1).mean()
        if delta_alfa.item() < thres:
            break

        t += 1


def distributed_greenkhorn(out, sinkhorn_iterations=100, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    K = L.shape[0]
    B = L.shape[1]

    # make the matrix sums to 1
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


if __name__ == "__main__":
    out = torch.randn(1000, 1000).cuda()
    result1 = distributed_sinkhorn(out)
    result2 = ot.sinkhorn()
