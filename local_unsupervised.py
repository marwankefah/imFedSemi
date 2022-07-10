from torch.utils.data import DataLoader
import copy
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from options import args_parser
from ramp import LinearRampUp
from utils import losses, ramps
from utils.util import get_timestamp, calculate_bank
from networks.models import DenseNet121
from utils_SimPLE import label_guessing, sharpen

args = args_parser()
args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset, Pi, priors_corr):
        self.dataset = dataset
        self.ldr_train = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4
        self.Pi = Pi
        self.iter_num = 0
        self.priors_corr = priors_corr
        self.temp_bank = []
        self.permanent_bank = set()
        self.flag = True
        net_ema = DenseNet121(out_size=args.class_num, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        self.ema_model = net_ema.cuda()
        for param in self.ema_model.parameters():
            param.detach_()

        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        print(len(dataset))
        self.max_warmup_step = round(len(dataset) / args.batch_size) * args.warmup
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)

        # self.real_Pi = list(Pi.numpy())

    def train(self, args, net, op_dict, epoch, logging):
        net.cuda()
        net.train()
        self.ema_model.cuda()
        self.ema_model.eval()

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net))
            self.flag = False

        self.epoch = epoch
        epoch_loss = []

        print(' Unsupervised training')

        for epoch in range(args.local_ep):

            batch_loss = []
            iter_max = len(self.ldr_train)

            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()

                with torch.no_grad():
                    guessed = label_guessing(self.ema_model, [ema_image_batch])
                    sharpened = sharpen(guessed)

                # pseu = torch.argmax(sharpened, dim=1)
                # label = label_batch.squeeze()
                # if len(label.shape) == 0:
                #     label = label.unsqueeze(dim=0)
                # correct_pseu += torch.sum(label[torch.max(sharpened, dim=1)[0] > args.confidence_threshold] == pseu[
                #     torch.max(sharpened, dim=1)[0] > args.confidence_threshold].cpu()).item()
                # all_pseu += len(pseu[torch.max(sharpened, dim=1)[0] > args.confidence_threshold])
                # train_right += sum([pseu[i].cpu() == label_batch[i].int() for i in range(label_batch.shape[0])])

                _, logits_str, probs_str = net(image_batch)
                # probs_str = F.softmax(logits_str, dim=1)
                # pred_label = torch.argmax(logits_str, dim=1)

                # same_total += sum([pred_label[sam] == pseu[sam] for sam in range(logits_str.shape[0])])

                loss_u = torch.sum(losses.softmax_mse_loss(probs_str, sharpened)) / args.batch_size

                ramp_up_value = self.ramp_up(current=self.epoch)

                loss = ramp_up_value * args.lambda_u * loss_u

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()

                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)

                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.epoch = self.epoch + 1

        self.ema_model.cpu()
        net.cpu()
        net_states = net.state_dict()

        return net_states, sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())
