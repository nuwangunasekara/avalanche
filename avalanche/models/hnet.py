import numpy as np
import torch
from torch import nn
from hypnettorch.mnets import LeNet
from hypnettorch.hnets import HMLP
from avalanche.models.MultiMLP import create_static_feature_extractor, preprocessor, train_nb
from skmultiflow.bayes import NaiveBayes

class HCNN(nn.Module):
    """
    Args:
    in_shape (tuple or list): The shape of an input sample.

        .. note::
            We assume the Tensorflow format, where the last entry
            denotes the number of channels.

    num_classes (int): The number of output neurons.
    device: The device to initialize the networks.
    lr: learning rate.
    num_of_tasks: Number of tasks.
    """
    def __init__(self,
                 in_shape,
                 num_classes,
                 device,
                 lr=1e-4,
                 num_of_tasks=4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.num_of_tasks = num_of_tasks
        self.device = device
        self.hnet = None
        self.mnet = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.nb_or_ht = NaiveBayes()

        self.training_exp = 0
        self.detected_task_id = 0
        self.call_predict = False
        self.mb_yy = None

        self.mb_task_id = None
        self.auto_detect_tasks = False

        self.f_ex_device, self.f_ex = create_static_feature_extractor(
            device=self.device,
            use_single_channel_fx=False,
            quantize=False
        )

    def forward(self, x):
        y = self.mb_yy
        t_id = self.detected_task_id

        static_features = self.get_static_features(x, self.f_ex, fx_device=self.f_ex_device)

        # yy = nn.functional.one_hot(y, num_classes=self.num_classes)
        xx = torch.swapdims(x, 1, 3)
        xx = xx.reshape(xx.shape[0], -1)

        if self.mnet is None:
            in_shape = (x.shape[2], x.shape[3], x.shape[1])
            self.mnet = LeNet(in_shape=in_shape, num_classes=self.num_classes,
                         arch='cifar', no_weights=True).to(self.device)
            self.hnet = HMLP(self.mnet.param_shapes,
                             uncond_in_size=1,
                             cond_in_size=0,
                             layers=[100, 100],
                             # no_cond_weights=True
                             # num_cond_embs=self.num_of_tasks
                             ).to(self.device)
            self.hnet.apply_hyperfan_init(mnet=self.mnet)
            # Adam usually works well in combination with hypernetwork training.
            self.optimizer = torch.optim.Adam(self.hnet.internal_params, lr=self.lr)

        if self.call_predict:
            nb_preds_t_id = self.nb_or_ht_predict(static_features)
            pred_t_id = torch.tensor(np.argmax(nb_preds_t_id.mean(axis=0), axis=0), dtype=torch.float).to(self.device) * \
                        torch.ones(static_features.shape[0]).to(self.device)
            # pred_t_id = np.argmax(nb_preds_t_id.mean(axis=0), axis=0).item()
                        # * torch.ones((xx.shape[0], 1)).to(self.device)
            # add pred_t_id to static_features
            # static_features = torch.cat(
            #     (static_features.to(self.device), pred_t_id.view(pred_t_id.shape[0], -1))
            #     , 1)
            with torch.no_grad():
                # w = self.hnet(cond_id=pred_t_id)
                w = self.hnet(uncond_input=pred_t_id)
                p = self.mnet.forward(xx, weights=w)
                # Note, the network outputs are logits, and thus not normalized.
                # preds = torch.softmax(preds, dim=1)
                # preds = preds.max(dim=1)[1]
                return p.detach()
        else: # train
            self.nb_train(static_features, t_id)

            self.optimizer.zero_grad()

            # w = self.hnet(cond_id=t_id)
            # static_features = torch.cat(
            #     (static_features.to(self.device), torch.ones(static_features.shape[0]).view(static_features.shape[0], -1).to(self.device) * t_id)
            #     , 1)
            t_id = t_id * torch.ones((xx.shape[0], 1)).to(self.device)
            w = self.hnet(uncond_input=t_id)
            loss = None
            preds = []
            for i in range(len(w)):
                p = self.mnet.forward(xx[i], weights=w[i])
                if loss is None:
                    loss = self.criterion(p, y[i].view(-1))
                else:
                    loss += self.criterion(p, y[i].view(-1))
                preds.append(p.detach())
            loss.backward()
            self.optimizer.step()
            return torch.cat(tuple(preds), 0).detach()

    def get_static_features(self, x, feature_extractor, fx_device):
        x = x.to(fx_device)
        if x.shape[1] == 1:  # has les than 3 channels
            preprocessed_x = preprocessor(x)  # repeat channel 1
            # preprocessed_x = x
        else:
            preprocessed_x = x
        return feature_extractor(preprocessed_x)

    def nb_train(self, features, task_id):
        train_nb(features, self.nb_or_ht, task_id)

    def nb_or_ht_predict(self, static_features):
        return self.nb_or_ht.predict_proba(static_features.detach().cpu().numpy())

    def add_to_frozen_pool(self):
        self.detected_task_id += 1

    def load_frozen_pool(self):
        return

    def clear_frozen_pool(self):
        return

    def print_stats(self, dumped_at=None):
        return

    def save_nb_predictions(self):
        return