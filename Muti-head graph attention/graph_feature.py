import torch.optim as optim
from utils import *
from model import Graph


def graph_feature(n_cir,c_fusion_sim,new_association,train_samples,val_samples):

    n_drug=new_association.shape[1]
    set_seed(666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_model={
        'num_heads_per_layer': 3,
        'num_embedding_features': 64,
        'num_hidden_layers': 2,
        'num_epoch': 1000,
        'lr': 1e-3,
        'weight_decay': 5e-3,
        'add_layer_attn': True,
        'residual': True,
    }
    lr = args_model['lr']
    weight_decay = args_model['weight_decay']
    residual = args_model['residual']
    add_layer_attn = args_model['add_layer_attn']
    num_epoch = args_model['num_epoch']
    num_hidden_layers = args_model['num_hidden_layers']
    num_heads_per_layer = [args_model['num_heads_per_layer'] for _ in range(num_hidden_layers)]
    num_embedding_features = [args_model['num_embedding_features'] for _ in range(num_hidden_layers)]

    train_samples_graph = train_samples.T
    val_samples_graph = val_samples.T
    edge_idx_cir = np.array(tuple(np.where(c_fusion_sim != 0)))
    edge_idx_cir = torch.tensor(edge_idx_cir, dtype=torch.long, device=device)
    model = Graph(n_drug + n_cir, num_hidden_layers, num_embedding_features, num_heads_per_layer,n_drug, n_cir, add_layer_attn, residual).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lr = 5e-5
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr=lr, step_size_up=200,
                                            step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                            cycle_momentum=False, last_epoch=-1)
    het_mat = construct_het_mat(new_association.T, c_fusion_sim)
    adj_mat = construct_adj_mat(new_association.T)

    cir_sim = torch.tensor(c_fusion_sim, dtype=torch.float, device=device)
    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
    het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)
    for epoch in range(num_epoch):
        model.train()
        pred_mat,ass_c = model(het_mat_device, edge_idx_device, cir_sim, edge_idx_cir)
        loss = calculate_loss(pred_mat.cpu(), train_samples_graph, val_samples_graph)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return ass_c.cpu()

