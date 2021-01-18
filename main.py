import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN

from sklearn.metrics import roc_auc_score

from algorithms import MDS_LOCAL

def train(args, model, device, train_graphs, optimizer, criterion, get_labels, epoch):
    model.train()
    
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = get_labels(batch_graph, model).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, get_labels, minibatch_size=64):
    model.eval()
    output = []
    labels = torch.zeros(0).long()
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
        labels = torch.cat([labels, get_labels([graphs[j] for j in sampled_idx], model)], dim=0)
    return torch.cat(output, 0), labels

def test(args, model, device, train_graphs, test_graphs, num_classes, get_labels, epoch):
    model.eval()
    
    output, labels = pass_data_iteratively(model, train_graphs, get_labels)
    labels = labels.to(device)
    acc_train = 0
    for i in range(num_classes):
        acc_train += roc_auc_score((labels == i).long().cpu().numpy(), output[:, i].cpu().numpy()) / num_classes

    output, labels = pass_data_iteratively(model, test_graphs, get_labels)
    labels = labels.to(device)
    acc_test = 0
    for i in range(num_classes):
        acc_test += roc_auc_score((labels == i).long().cpu().numpy(), output[:, i].cpu().numpy()) / num_classes

    return acc_train, acc_test


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--random', type=int, default=None,
                                        help='the range of random features (default: None). None means it does not add random features.')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.dataset in ['TRIANGLE', 'TRIANGLE_EX', 'LCC', 'LCC_EX', 'MDS', 'MDS_EX']:
        node_classification = True
        train_graphs, _ = load_data(f'dataset/{args.dataset}/{args.dataset}_train.txt', args.degree_as_tag)
        test_graphs, _ = load_data(f'dataset/{args.dataset}/{args.dataset}_test.txt', args.degree_as_tag)
        for g in train_graphs + test_graphs:
            if args.random:
                g.node_features = torch.ones(g.node_features.shape[0], 0)
            else:
                g.node_features = torch.ones(g.node_features.shape[0], 1)
        if args.dataset in ['TRIANGLE', 'TRIANGLE_EX', 'MDS', 'MDS_EX']:
            num_classes = 2
        elif args.dataset in ['LCC', 'LCC_EX']:
            num_classes = 3
        else:
            assert(False)
        if args.dataset in ['MDS', 'MDS_EX']:
            get_labels = lambda batch_graph, model: torch.LongTensor(MDS_LOCAL(model, batch_graph))
            criterion = nn.CrossEntropyLoss()
        else:
            get_labels = lambda batch_graph, model: torch.LongTensor(sum([graph.node_tags for graph in batch_graph], []))
            bc = [0 for i in range(num_classes)]
            for G in train_graphs:
                for t in G.node_tags:
                    bc[t] += 1
            w = torch.FloatTensor([max(bc) / bc[i] for i in range(num_classes)]).to(device)
            criterion = nn.CrossEntropyLoss(weight=w)
    else:
        node_classification = False
        graphs, num_classes = load_data(f'dataset/{args.dataset}/{args.dataset}.txt', args.degree_as_tag)
        
        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
        
        criterion = nn.CrossEntropyLoss()
        get_labels = lambda batch_graph, model: torch.LongTensor([graph.label for graph in batch_graph])

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, args.random, node_classification, device).to(device)

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        
        avg_loss = train(args, model, device, train_graphs, optimizer, criterion, get_labels, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, num_classes, get_labels, epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        print(model.eps)
    

if __name__ == '__main__':
    main()
