import json
import os
import sys
# for linux env.
sys.path.insert(0,'..')
import argparse
from distutils.util import strtobool
import torch
import torch.nn as nn

from data.data_loader import NumpyTupleDataset
from mflow.models.hyperparams import Hyperparameters
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import check_validity, save_mol_png

import time
from mflow.utils.timereport import TimeReport
from mflow.generate import generate_mols

import functools
print = functools.partial(print, flush=True)

def train(args):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    # Device configurationradha ramanam
    device = -1
    multigpu = False
    if args.gpu >= 0:
        # signle gpu
        # device = args.gpu
        device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    elif args.gpu == -1:
        # cpu
        device = torch.device('cpu')
    else:
        # multigpu, can be slower than using just 1 gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multigpu = True

    debug = args.debug

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # Model configuration
    b_hidden_ch = [int(d) for d in args.b_hidden_ch.strip(',').split(',')]
    a_hidden_gnn = [int(d) for d in args.a_hidden_gnn.strip(',').split(',')]
    a_hidden_lin = [int(d) for d in args.a_hidden_lin.strip(',').split(',')]
    mask_row_size_list = [int(d) for d in args.mask_row_size_list.strip(',').split(',')]
    mask_row_stride_list = [int(d) for d in args.mask_row_stride_list.strip(',').split(',')]
    if args.data_name == 'qm9':
        from data import transform_qm9
        data_file = 'qm9_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]
        b_n_type = 4
        b_n_squeeze = 3
        a_n_node = 9
        a_n_type = len(atomic_num_list)  # 5
        valid_idx = transform_qm9.get_val_ids()  # len: 13,082, total data: 133,885
    elif args.data_name == 'zinc250k':
        from data import transform_zinc250k
        data_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list  # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        # mlp_channels = [1024, 512]
        # gnn_channels = {'gcn': [16, 128], 'hidden': [256, 64]}
        b_n_type = 4
        b_n_squeeze = 19   # 2
        a_n_node = 38
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = transform_zinc250k.get_val_ids()
    else:
        raise ValueError('Only support qm9 and zinc250k right now. '
                         'Parameters need change a little bit for other dataset.')

    model_params = Hyperparameters(b_n_type=b_n_type,  # 4,
                                   b_n_flow=args.b_n_flow,
                                   b_n_block=args.b_n_block,
                                   b_n_squeeze=b_n_squeeze,
                                   b_hidden_ch=b_hidden_ch,
                                   b_affine=True,
                                   b_conv_lu=args.b_conv_lu,
                                   a_n_node=a_n_node,
                                   a_n_type=a_n_type,
                                   a_hidden_gnn=a_hidden_gnn,
                                   a_hidden_lin=a_hidden_lin,
                                   a_n_flow=args.a_n_flow,
                                   a_n_block=args.a_n_block,
                                   mask_row_size_list=mask_row_size_list,
                                   mask_row_stride_list=mask_row_stride_list,
                                   a_affine=True,
                                   learn_dist=args.learn_dist,
                                   seed=args.seed,
                                   noise_scale=args.noise_scale
                                   )
    print('Model params:')
    model_params.print()
    model = MoFlow(model_params)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_hyperparams(os.path.join(args.save_dir, 'moflow-params.json'))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    else:
        multigpu = False
    model = model.to(device)

    # Datasets:
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn)  # 133885
    if len(valid_idx) > 0:
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
        # n_train = len(train_idx)  # 120803
        train = torch.utils.data.Subset(dataset, train_idx)  # 120,803
        test = torch.utils.data.Subset(dataset, valid_idx)  # 13,082
    else:
        torch.manual_seed(args.seed)
        train, test = torch.utils.data.random_split(
            dataset,
            [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                                   shuffle=args.shuffle, num_workers=args.num_workers)

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(time.time() - start))
    print('Data shuffle: {}, Number of data loader workers: {}!'.format(args.shuffle, args.num_workers))
    if args.gpu >= 0:
        print('Using GPU device:{}!'.format(args.gpu))
    print('Num Train-size: {}'.format(len(train)))
    print('Num Minibatch-size: {}'.format(args.batch_size))
    print('Num Iter/Epoch: {}'.format(len(train_dataloader)))
    print('Num epoch: {}'.format(args.max_epochs))
    print('==========================================')


    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the models
    iter_per_epoch = len(train_dataloader)
    log_step = args.save_interval  # 20 default
    tr = TimeReport(total_iter=args.max_epochs * iter_per_epoch)
    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch+1, time.ctime()))
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # turn off shuffle to see the order with original code
            x = batch[0].to(device)  # (256,9,5)
            adj = batch[1].to(device)  # (256,4,9, 9)
            adj_normalized = rescale_adj(adj).to(device)

            # Forward, backward and optimize
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            if multigpu:
                nll = model.module.log_prob(z, sum_log_det_jacs)
            else:
                nll = model.log_prob(z, sum_log_det_jacs)
            loss = nll[0] + nll[1]
            loss.backward()
            optimizer.step()
            tr.update()

            # Print log info
            if (i+1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},'
                      ' nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                      format(epoch+1, args.max_epochs, i+1, iter_per_epoch,
                             loss.item(), nll[0].item(), nll[1].item(),
                             tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()

        if debug:
            def print_validity(ith):
                model.eval()
                if multigpu:
                    adj, x = generate_mols(model.module, batch_size=100, device=device)
                else:
                    adj, x = generate_mols(model, batch_size=100, device=device)
                valid_mols = check_validity(adj, x, atomic_num_list)['valid_mols']
                mol_dir = os.path.join(args.save_dir, 'generated_{}'.format(ith))
                os.makedirs(mol_dir, exist_ok=True)
                for ind, mol in enumerate(valid_mols):
                    save_mol_png(mol, os.path.join(mol_dir, '{}.png'.format(ind)))
                model.train()
            print_validity(epoch+1)

        # The same report for each epoch
        print('Epoch [{}/{}], Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},'
              ' nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch + 1, args.max_epochs, -1, iter_per_epoch,
                     loss.item(), nll[0].item(), nll[1].item(),
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()

        # Save the model checkpoints
        save_epochs = args.save_epochs
        if save_epochs == -1:
            save_epochs = args.max_epochs
        if (epoch + 1) % save_epochs == 0:
            if multigpu:
                torch.save(model.module.state_dict(), os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch + 1)))
            else:
                torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch + 1)))
            tr.end()

    print("[Training Ends], Start at {}, End at {}".format(time.ctime(start), time.ctime()))

