# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

# InfoNCE-style contrastive loss using cosine similarity
class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1, project=False, in_dim_1=None, in_dim_2=None, out_dim=None) -> None:
        super().__init__()
        self.tau = tau
        self.project = project

        if project:
            if not in_dim_1:
                return ValueError
            self.x_projector = nn.Linear(in_dim_1, out_dim)
            self.y_projector = nn.Linear(in_dim_2, out_dim)


    def forward(self, X, Y):
        
        if self.project:
            X = self.x_projector(X)
            Y = self.y_projector(Y)

        loss = self.compute_cl(X, Y) + self.compute_cl(Y, X)

        return loss
    

    def compute_cl(self, X, Y):

        '''
        X: (bs, hidden_size), Y: (bs, hidden_size)
        tau: the temperature factor
        '''
        #sim_matrix = X.mm(Y.t())    # (bs, bs)
        sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
        pos = torch.exp(torch.diag(sim_matrix) / self.tau).unsqueeze(0)   # (1, bs)
        neg = torch.sum(torch.exp(sim_matrix / self.tau), dim=0) - pos     # (1, bs)
        # TODO: whether to subtract pos depends on the exact formulation
        loss = - torch.log(pos / neg)
        loss = loss.view(X.shape[0], -1)

        return loss
    

# Soft-label contrastive loss using dot-product similarity
class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    






class Contrastive_Loss_Cos(nn.Module):
    """
    Cosine similarity loss for positive pairs with optional projection.
    """
    def __init__(self, tau=1, project=True, in_dim_1=768, in_dim_2=512, out_dim=128) -> None:
        super().__init__()
        self.tau = tau
        self.project = project


        # Create projector layers when projection is enabled
        if project:
            if not in_dim_1:
                raise ValueError("Input dimension must be specified when project=True")
            self.x_projector = nn.Linear(in_dim_1, out_dim)

            self.y_projector = nn.Sequential(
            nn.Linear(in_dim_1, int(in_dim_1 / 2)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(in_dim_1 / 2), out_dim)
        )

    def forward(self, X, Y):
        """
        Compute cosine-similarity loss between two embedding sets.
        X: [batch_size, hidden_dim]
        Y: [batch_size, hidden_dim]
        
        Loss = 1 - cos(MLP(X), MLP(Y))
        """

        if self.project:
            # X = self.x_projector(X)
            Y = self.y_projector(Y)
            
        # L2-normalize embeddings
        X_norm = F.normalize(X, p=2, dim=1)
        Y_norm = F.normalize(Y, p=2, dim=1)
        
        # Compute per-pair cosine similarity (aligned pairs)
        cos_sim = torch.sum(X_norm * Y_norm, dim=1)  # [batch_size]
        
        # Loss: 1 - cosine similarity
        loss = 1 - cos_sim
        
        # Return mean loss
        return loss.mean()







class Contrastive_Loss_L2(nn.Module):
    """
    L2 distance loss for positive pairs with optional projection.
    """
    def __init__(self, tau=1, project=True, in_dim_1=768, in_dim_2=512, out_dim=128) -> None:
        super().__init__()
        self.tau = tau
        self.project = project

        # Create projector layers when projection is enabled
        if project:
            if not in_dim_1:
                raise ValueError("Input dimension must be specified when project=True")
            self.x_projector = nn.Linear(in_dim_1, out_dim)

            self.y_projector = nn.Sequential(
                nn.Linear(in_dim_1, int(in_dim_1 / 2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(in_dim_1 / 2), out_dim)
            )

    def forward(self, X, Y):
        """
        Compute L2 distance loss between two embedding sets.
        X: [batch_size, hidden_dim]
        Y: [batch_size, hidden_dim]
        
        Loss = ||MLP(X) - MLP(Y)||_2^2
        """
        # 如果需要，应用投影
        if self.project:
            # X = self.x_projector(X)
            Y = self.y_projector(Y)
            
        # Optional: normalize for scale consistency
        X_norm = F.normalize(X, p=2, dim=1)
        Y_norm = F.normalize(Y, p=2, dim=1)
        
        # Compute per-pair squared L2 distance (aligned pairs)
        l2_dist = torch.sum((X_norm - Y_norm) ** 2, dim=1)  # [batch_size]
        
        # Return mean loss
        return l2_dist.mean()




class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    In forward, the first argument is for query, the second for key/value.
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size
        self.all_head_size  = all_head_size
        self.num_heads      = head_num
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):


        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



class Attention(nn.Module):

    def __init__(self, hidden_size, method="dot"):

        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size,bias=False)


    def forward(self, query, key):
        """
        query: [bs, hidden_size]
        key: [bs, seq_len, hidden_size]
        weight: [bs, seq_len, 1]
        """

        if self.method == "dot":
            return self.dot_score(query, key)
        elif self.method == "general":
            return self.general_score(query, key)


    def dot_score(self, query, key):
        
        query = query.unsqueeze(2)  #[bs, hidden_size, 1]
        attn_energies = torch.bmm(key, query) # (bs, seq_len, hidden_size) * (bs, hidden_size, 1) --> (bs, seq_len, 1)
        attn_energies = attn_energies.squeeze(-1) # (bs, seq_len)

        return F.softmax(attn_energies, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
    

    def general_score(self, query, key):

        query = self.Wa(query).unsqueeze(2) # (bs, hidden_size, 1)
        attn_energies = torch.bmm(key, query).squeeze(-1) 
        
        return F.softmax(attn_energies,dim=-1).unsqueeze(-1)
    

def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return torch.sum(F.softplus(neg_preds - pos_preds))



class SpAdjEdgeDrop(nn.Module):

    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask]# / keep_rate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class CalibrationLoss(nn.Module):
    def __init__(self, t=2.0, co_occurrence_file=None):
        super().__init__()
        self.gamma = t  
        self.co_occurrence_weights = {}  
        self.weight_cache = {}  
        
        if co_occurrence_file:
            self.load_co_occurrence_weights(co_occurrence_file)
    
    def load_co_occurrence_weights(self, file_path):
        """Load log-normalized co-occurrence strengths from a NPZ file
        produced by ``data/getCo-occer.ipynb`` (keys ``pairs`` int32 of
        shape ``[N, 2]`` and ``weights`` float32 of shape ``[N]`` in
        ``[0, 1]``). ``exp(-gamma * s_ij)`` is applied lazily in
        :meth:`get_weight` so ``gamma`` can change without reloading.
        """
        try:
            with np.load(file_path) as data:
                pairs = data["pairs"]
                s_values = data["weights"].astype(np.float32)

            self.co_occurrence_weights = {}
            for (i, j), s in zip(pairs.tolist(), s_values.tolist()):
                self.co_occurrence_weights[(i, j)] = s
                self.co_occurrence_weights[(j, i)] = s
            self.weight_cache = {}
            print(f"Loaded {len(self.co_occurrence_weights) // 2} co-occurrence pair weights from {file_path}")
        except Exception as e:
            print(f"Error loading co-occurrence weights: {e}")
            self.co_occurrence_weights = {}
    
    def get_weight(self, item_id1, item_id2):
        key = (item_id1, item_id2)
        if key in self.weight_cache:
            return self.weight_cache[key]
        
        s_ij = self.co_occurrence_weights.get(key)
        if s_ij is not None:
            weight_value = math.exp(-self.gamma * s_ij)
        else:
            weight_value = 1.0
        
        self.weight_cache[key] = weight_value
        return weight_value
    
    def forward(self, embeddings, item_ids):
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        

        n = embeddings.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        

        item_ids_np = item_ids.cpu().numpy()
        weights = []
        for i in range(n):
            for j in range(i+1, n):
                weight = self.get_weight(int(item_ids_np[i]), int(item_ids_np[j]))
                weights.append(weight)
        

        weights_tensor = torch.tensor(weights, device=embeddings.device, dtype=embeddings.dtype)
        weight_matrix = torch.ones((n, n), device=embeddings.device, dtype=embeddings.dtype)
        idx = 0
        triu_indices = torch.triu_indices(n, n, offset=1)
        weight_matrix[triu_indices[0], triu_indices[1]] = weights_tensor
        weight_matrix[triu_indices[1], triu_indices[0]] = weights_tensor  # Symmetric
        
        mask = 1.0 - torch.eye(n, device=embeddings.device, dtype=embeddings.dtype)
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # [n, n, dim]
        distances = torch.sum(diff**2, dim=2)  # [n, n]
        weighted_distances = distances * weight_matrix * mask
        

        num_pairs = n * (n - 1)
        calibration_loss = -torch.sum(weighted_distances) / num_pairs
        
        return calibration_loss


class FastCalibrationLoss(nn.Module):
    def __init__(self, t=2.0, co_occurrence_file=None, item_num=None):
        super().__init__()
        self.gamma = t
        self.item_num = item_num
        self.weight_dict = {}

        if co_occurrence_file:
            self.load_co_occurrence_weights(co_occurrence_file)

    def load_co_occurrence_weights(self, file_path):
        """Load log-normalized co-occurrence strengths from a NPZ file
        produced by ``data/getCo-occer.ipynb`` (keys ``pairs`` int32 of
        shape ``[N, 2]`` and ``weights`` float32 of shape ``[N]`` in
        ``[0, 1]``) and precompute ``exp(-gamma * s_ij)`` so the forward
        pass only has to do dictionary lookups.
        """
        try:
            with np.load(file_path) as data:
                pairs = data["pairs"]
                s_values = data["weights"].astype(np.float32)
            weights = np.exp(-self.gamma * s_values)

            self.weight_dict = {}
            for (i, j), w in zip(pairs.tolist(), weights.tolist()):
                self.weight_dict[(i, j)] = w
                self.weight_dict[(j, i)] = w
            print(f"Loaded {len(self.weight_dict) // 2} co-occurrence pair weights from {file_path}")
        except Exception as e:
            print(f"Error loading co-occurrence weights: {e}")
            self.weight_dict = {}

    def forward(self, embeddings, item_ids):

        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        n = embeddings.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        

        item_ids_cpu = item_ids.cpu().numpy()
        i_indices, j_indices = np.triu_indices(n, k=1)

        weights = [
            self.weight_dict.get((int(item_ids_cpu[i]), int(item_ids_cpu[j])), 1.0)
            for i, j in zip(i_indices, j_indices)
        ]

        weights_tensor = torch.tensor(weights, device=embeddings.device, dtype=embeddings.dtype)
        
        # Build symmetric weight matrix
        weight_matrix = torch.ones((n, n), device=embeddings.device, dtype=embeddings.dtype)
        triu_i = torch.from_numpy(i_indices).to(embeddings.device)
        triu_j = torch.from_numpy(j_indices).to(embeddings.device)
        
        weight_matrix[triu_i, triu_j] = weights_tensor
        weight_matrix[triu_j, triu_i] = weights_tensor
        

        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        distances = torch.sum(diff**2, dim=2)
        mask = 1.0 - torch.eye(n, device=embeddings.device, dtype=embeddings.dtype)
        weighted_distances = distances * weight_matrix * mask
        
        num_pairs = n * (n - 1)
        calibration_loss = -torch.sum(weighted_distances) / num_pairs
        
        return calibration_loss


