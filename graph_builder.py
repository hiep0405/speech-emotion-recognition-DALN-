import torch
import numpy as np
from torch_geometric.data import Data

def build_temporal_graph(
    features: torch.Tensor,
    similarity_threshold: float = 0.5,
    min_degree: int = 2,
    add_self_loops: bool = True
) -> Data:
    """
    Xây dựng Temporal Graph từ ma trận đặc trưng wav2vec 2.0 (N x 1024)
    theo đúng công thức trong Paper ICASSP 2026:
    
    1. Khoảng cách Euclid: d_{j,k} = ||x_j - x_k||_2
    2. Độ rộng băng thông kernel: \sigma = median(d_{j,k}) / 2
    3. Trọng số tương đồng Gaussian RBF: W_{j,k} = exp(-d_{j,k}^2 / (2\sigma^2))
    4. Ngưỡng lọc cạnh: W_{j,k} >= similarity_threshold (0.5)
    5. Ràng buộc bậc tối thiểu: Bậc node >= min_degree (2)
    6. Tự nối (Self-loops): W_{j,j} = 1
    7. Sắp xếp láng giềng theo thứ tự thời gian: t_{j_1} < t_{j_2} < ... < t_{j_m}
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor (N, D), got shape {features.shape}")
        
    N, D = features.shape
    
    # 0. Chuẩn hóa CMVN theo từng câu (Utterance-level Z-Score Normalization)
    # Loại bỏ đặc trưng âm sắc cố định của người nói (Speaker Bias / Pitch Level),
    # bảo toàn động học cảm xúc tương đối giữa các khung thời gian
    if N > 1:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-6
        features = (features - mean) / std
    
    # 1. Tính ma trận khoảng cách Euclid d_{j,k}
    dist_matrix = torch.cdist(features, features, p=2.0)
    
    # 2. Tính \sigma theo Heuristic Trung vị: \sigma = median(d_{j,k}) / 2
    # Chỉ lấy phần tam giác trên (không bao gồm đường chéo) để tính median chính xác
    triu_indices = torch.triu_indices(N, N, offset=1)
    if triu_indices.shape[1] > 0:
        pairwise_dists = dist_matrix[triu_indices[0], triu_indices[1]]
        median_dist = torch.median(pairwise_dists)
        sigma = median_dist / 2.0
        # Tránh trường hợp sigma = 0
        if sigma < 1e-6:
            sigma = torch.tensor(1.0, device=features.device)
    else:
        sigma = torch.tensor(1.0, device=features.device)
        
    # 3. Trọng số Gaussian RBF: W_{j,k} = exp(-d_{j,k}^2 / (2\sigma^2))
    sim_matrix = torch.exp(-(dist_matrix ** 2) / (2.0 * (sigma ** 2)))
    
    # 4. Ngưỡng lọc cạnh W_{j,k} >= threshold
    adj_matrix = (sim_matrix >= similarity_threshold).clone()
    
    # Bỏ đường chéo tạm thời để xét bậc thực tế của láng giềng
    adj_matrix.fill_diagonal_(False)
    
    # 5. Đảm bảo bậc tối thiểu (min_degree constraint)
    # Nếu node có ít hơn min_degree láng giềng, nối với các node gần nhất theo khoảng cách
    degrees = adj_matrix.sum(dim=-1)
    under_degree_nodes = torch.where(degrees < min_degree)[0]
    
    if len(under_degree_nodes) > 0 and N > 1:
        # Lấy top k node gần nhất (khoảng cách nhỏ nhất, loại bỏ chính nó)
        dist_no_diag = dist_matrix.clone()
        dist_no_diag.fill_diagonal_(float("inf"))
        k = min(min_degree, N - 1)
        _, nearest_neighbors = torch.topk(dist_no_diag[under_degree_nodes], k=k, largest=False, dim=-1)
        
        for idx, node in enumerate(under_degree_nodes):
            for neighbor in nearest_neighbors[idx]:
                adj_matrix[node, neighbor] = True
                adj_matrix[neighbor, node] = True # Đảm bảo vô hướng
                
    # 6. Thêm self-loops nếu cần
    if add_self_loops:
        adj_matrix.fill_diagonal_(True)
        sim_matrix.fill_diagonal_(1.0)
        
    # 7. Trích xuất edge_index và edge_weight
    # torch.nonzero trả về các cặp (i, j)
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t() # (2, E)
    
    # Quan trọng: Sắp xếp các cạnh sao cho với mỗi node nguồn i,
    # các láng giềng đích j được sắp xếp theo đúng thứ tự thời gian tăng dần j_1 < j_2 < ...
    # Vì j chính là chỉ số khung thời gian (frame index: 0, 1, ..., N-1),
    # việc sắp xếp lexicographical theo (src, dst) đảm bảo 100% tính tuần tự thời gian!
    src = edge_indices[0]
    dst = edge_indices[1]
    sort_idx = torch.argsort(src * N + dst)
    edge_index = edge_indices[:, sort_idx]
    
    edge_weight = sim_matrix[edge_index[0], edge_index[1]]
    
    # Đóng gói thành đối tượng Data của PyTorch Geometric
    graph_data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(-1), # (E, 1)
        num_nodes=N
    )
    
    return graph_data
