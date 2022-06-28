import torch
import torch.nn as nn
from einops import rearrange, repeat


class LACL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, contras_table, k, dim=128, n=10240, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(LACL, self).__init__()

        self.k = k
        self.n = n
        self.m = m
        self.T = T
        self.contras_table = contras_table

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, k * n))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue = rearrange(self.queue, 'c (k n) -> c k n', k=k, n=n)

        self.register_buffer("queue_ptr", torch.zeros(k, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        for i in torch.unique(labels):
            sample_inds = torch.where(labels == i)[0]
            batch_size = sample_inds.shape[0]

            ptr = int(self.queue_ptr[i])
            # replace the keys at ptr (dequeue and enqueue)
            queue_inds = [int((x + ptr) % self.n) for x in range(batch_size)]
            self.queue[:, i, queue_inds] = keys[sample_inds].T
            # move pointer
            ptr = (ptr + batch_size) % self.n
            self.queue_ptr[i] = ptr

    @torch.no_grad()
    def _search_enqueue_samples(self, keys, labels):
        # gather all before calculating KL loss
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        # create p distribution
        p_dis = torch.einsum('b c, c k n -> b k n', [keys, self.queue.clone().detach()])
        p_dis = rearrange(p_dis, 'b k n -> b (k n)')
        p_dis = torch.softmax(p_dis / self.T, dim=-1)

        # create q distribution
        q_dis = torch.ones((keys.shape[0], self.k, self.n)).cuda()
        q_dis = torch.einsum('b k n, b k -> b k n', [q_dis, torch.nn.functional.one_hot(labels, self.k)])

        q_dis = rearrange(q_dis, 'b k n -> b (k n)')
        q_dis = torch.softmax(q_dis, dim=-1)

        kl_loss = nn.functional.kl_div(p_dis.log(), q_dis, reduction="none").sum(dim=-1)
        kl_mean = torch.mean(kl_loss)
        enqueue_inds = torch.where(kl_loss <= kl_mean)

        return keys[enqueue_inds], labels[enqueue_inds]

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Bx1
        l_pos = torch.einsum('b c, b c -> b', [q, k]).unsqueeze(-1)
        # negative logits: BxKxN
        l_neg = torch.einsum('b c, c k n -> b k n', [q, self.queue.clone().detach()])
        print(self.contras_table)
        print(self.contras_table[labels])
        l_neg[self.contras_table[labels]] = float('-inf')
        print(l_neg)
        l_neg = rearrange(l_neg, 'b k n -> b (k n)')

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        enqueue_keys, enqueue_labels = self._search_enqueue_samples(k, labels)
        self._dequeue_and_enqueue(enqueue_keys, enqueue_labels)

        return logits, targets


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
