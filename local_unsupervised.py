from torch.utils.data import DataLoader
import copy
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from options import args_parser
from utils import losses
from utils.util import get_timestamp, calculate_bank

args = args_parser()


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset, Pi, priors_corr):
        self.dataset = dataset
        self.ldr_train = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4
        self.Pi = Pi
        self.priors_corr = priors_corr
        self.temp_bank = []
        self.permanent_bank = set()
        # self.real_Pi = list(Pi.numpy())

    def train(self, args, net, op_dict, epoch,logging):
        net.cuda()
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        loss_fun = nn.CrossEntropyLoss()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        self.epoch = epoch
        epoch_loss = []

        print(' Unsupervised training')

        for epoch in range(args.local_ep):

            batch_loss = []
            iter_max = len(self.ldr_train)

            for i, (_, _, (image_batch_w, image_batch_s), label_batch) in enumerate(self.ldr_train):
                image_batch_w, image_batch_s, label_batch = image_batch_w.cuda(), image_batch_s.cuda(), label_batch.cuda()


                model_inputs = torch.cat((image_batch_w, image_batch_s))

                _, logits, _ = net(model_inputs)

                logits_u_w, logits_u_s = logits.chunk(2)

                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()
                loss = args.lambda_u * Lu

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(f' Local Loss: {epoch_loss}')
        net.cpu()
        net_states = net.state_dict()
        return net_states, sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
