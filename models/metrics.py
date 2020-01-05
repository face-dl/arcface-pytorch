from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NoiseTolerant(object):
    def __init__(self):
        self.consines = []
        self.slide_batch_num_ = 1000

        self.l_bin_id_ = -1
        self.r_bin_id_ = -1
        self.lt_bin_id_ = -1
        self.rt_bin_id_ = -1
        self.t_bin_ids_ = []

        self.bins_ = 200
        self.value_low_ = -1
        self.value_high_ = 1

        self.s = 1.0 - 0.99
        self.r = 2.576

        self.pdf_ = [0] * (self.bins_ + 1)
        self.fr = 2

        self.noise_ratio_ = 0
        self.iters = 0

    def clamp(self, val, min_val, max_val):
        val = max(val, min_val)
        val = min(val, max_val)
        return val

    def get_bin_id(self, cos):
        bin_id = self.bins_ * (cos - self.value_low_) / (self.value_high_ - self.value_low_)
        bin_id = self.clamp(bin_id, 0, self.bins_)
        return int(bin_id)

    def get_cos(self, bin_id):
        cos = self.value_low_ + (self.value_high_ - self.value_low_) * bin_id / self.bins_
        cos = self.clamp(cos, self.value_low_, self.value_high_)
        return cos

    def softplus(self, x):
        return np.log(1.0 + np.exp(x))

    def weight2(self, bin_id):
        z = (bin_id - self.lt_bin_id_) / (self.r_bin_id_ - self.lt_bin_id_)
        upper = self.softplus(10.0 * z)
        down = self.softplus(10.0)
        weight2 = self.clamp(upper / down, 0.0, 1.0)
        return weight2

    def weight3(self, bin_id):
        a = ((self.r_bin_id_ - self.rt_bin_id_) / self.r) if bin_id > self.rt_bin_id_ else  ((self.rt_bin_id_ - self.l_bin_id_) / self.r)
        weight3 = np.exp(-1.0 * (bin_id - self.rt_bin_id_) * (bin_id - self.rt_bin_id_) / (2 * a * a))
        return weight3

    def alpha(self, r_cos=None):
        if r_cos is None:
            r_cos = self.clamp(self.get_cos(self.r_bin_id_), 0.0, 1.0)
        alpha = 2.0 - 1.0 / (1.0 + np.exp(5 - 20 * r_cos)) - 1.0 / (1.0 + np.exp(20 * r_cos - 15))
        return alpha

    def cos2weight(self, cos):
        # lt_bin_id_, rt_bin_id_ r_bin_id_
        bin_id = self.get_bin_id(cos)
        weight1 = 1.0

        weight2 = self.weight2(bin_id)
        weight3 = self.weight3(bin_id)

        r_cos = self.clamp(self.get_cos(self.r_bin_id_), 0.0, 1.0)
        alpha = self.alpha(r_cos)
        # weight = alpha * (r_cos < 0.5) * weight1 + (1 - alpha) * weight2 + alpha * (r_cos > 0.5) * weight3
        weight = (1 - alpha) * weight2 + alpha * (r_cos > 0.5) * weight3
        # weight = alpha * (r_cos < 0.5) * weight1
        return weight

    def delta(self, a, b):
        if b == -1:
            return a - b
        if (a > b):
            return +1
        if (a < b):
            return -1
        return 0

    def drow_pic(self, func, file_path, is_cosine=True):
        x = np.arange(200).astype(np.float32)
        if is_cosine:
            x = np.arange(200) / 100 - 1.0
        y = np.zeros_like(x)
        for i in range(200):
            y[i] = func(x[i])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        plt.plot(x, y, color="red", linewidth=1)
        plt.title('余弦柱状图')
        ###保存
        plt.savefig(file_path)

    def get_mul_weight(self, consines, labels):
        self.iters += 1
        batch_size = len(consines)
        consines = consines.gather(1, labels.unsqueeze(dim=1).long())
        consines = consines.squeeze().data.cpu().numpy()
        self.consines.append(consines)

        # add
        for consine in consines:
            bin_id = self.get_bin_id(consine)
            self.pdf_[bin_id] += 1

        if len(self.consines) < self.slide_batch_num_:
            return torch.ones(batch_size)

        # del
        while len(self.consines) > self.slide_batch_num_:
            del_consines = self.consines.pop(0)
            for consine in del_consines:
                bin_id = self.get_bin_id(consine)
                self.pdf_[bin_id] -= 1
        # mean
        sum_filter_pdf = 0.0
        filter_pdf = [0] * (self.bins_ + 1)
        for i in range(self.fr, self.bins_ - self.fr):
            count = self.fr + self.fr + 1
            for j in range(i - self.fr, i + self.fr + 1):
                filter_pdf[i] += self.pdf_[j] / count
            sum_filter_pdf += filter_pdf[i]

        pcf_ = [0] * (self.bins_ + 1)
        pcf_[0] = filter_pdf[0] / sum_filter_pdf
        for i in range(1, self.bins_ + 1):
            pcf_[i] = pcf_[i - 1] + filter_pdf[i] / sum_filter_pdf

        origin_l_bin_id = self.l_bin_id_
        origin_r_bin_id = self.r_bin_id_
        origin_lt_bin_id = self.lt_bin_id_
        origin_rt_bin_id = self.rt_bin_id_

        for i in range(self.bins_ + 1):
            if pcf_[i] > self.s * 0.5:
                break
            self.l_bin_id_ = i
        for i in range(self.bins_, -1, -1):
            if pcf_[i] < 1.0 - self.s * 0.5:
                break
            self.r_bin_id_ = i
        if self.l_bin_id_ >= self.r_bin_id_:
            logging.info("l_bin_id_ %s >= r_bin_id_ %s", self.l_bin_id_, self.r_bin_id_)
            return torch.ones(batch_size)

        m_bin_id_ = (self.l_bin_id_ + self.r_bin_id_) / 2
        t_bin_id_ = np.argmax(filter_pdf)

        t_bin_ids_ = []
        for i in range(max(self.l_bin_id_, 5), min(self.r_bin_id_, self.bins_ - 5)):
            if filter_pdf[i] >= filter_pdf[i - 1] and filter_pdf[i] >= filter_pdf[i + 1] and \
                            filter_pdf[i] > filter_pdf[i - 2] and filter_pdf[i] > filter_pdf[i + 2] and \
                            filter_pdf[i] > filter_pdf[i - 3] + 1 and filter_pdf[i] > filter_pdf[i + 3] + 1 and \
                            filter_pdf[i] > filter_pdf[i - 4] + 2 and filter_pdf[i] > filter_pdf[i + 4] + 2 and \
                            filter_pdf[i] > filter_pdf[i - 5] + 3 and filter_pdf[i] > filter_pdf[i + 5] + 3:
                t_bin_ids_.push_back(i)
                i += 5
        if len(t_bin_ids_) == 0:
            t_bin_ids_.append(t_bin_id_)
        if t_bin_id_ < m_bin_id_:
            self.lt_bin_id = t_bin_id_
            self.rt_bin_id = max(t_bin_ids_[-1], m_bin_id_)
        else:
            self.rt_bin_id = t_bin_id_
            self.lt_bin_id = min(t_bin_ids_[0], m_bin_id_)

        self.l_bin_id_ = origin_l_bin_id + self.delta(self.l_bin_id_, origin_l_bin_id)
        self.r_bin_id_ = origin_r_bin_id + self.delta(self.r_bin_id_, origin_r_bin_id)
        self.lt_bin_id_ = origin_lt_bin_id + self.delta(self.lt_bin_id_, origin_lt_bin_id)
        self.rt_bin_id_ = origin_rt_bin_id + self.delta(self.rt_bin_id_, origin_rt_bin_id)

        if (self.lt_bin_id_ < m_bin_id_):
            self.noise_ratio_ = 2.0 * pcf_[self.lt_bin_id_]
        else:
            self.noise_ratio_ = 0.0

        weights = torch.zeros(batch_size)
        for i in range(batch_size):
            weights[i] = self.cos2weight(consines[i])
        if self.iters % 100 == 0:
            logging.info("tolerant iters %s weight %s", self.iters, weights)
        return weights


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, noise_tolerant=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        _weight = torch.FloatTensor(out_features, in_features)
        nn.init.xavier_uniform_(_weight)
        self.weight = Parameter(_weight / _weight.norm() * 0.2)

        self.easy_margin = easy_margin
        self.noise_tolerant = noise_tolerant
        if noise_tolerant:
            self.nt = NoiseTolerant()
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        if self.noise_tolerant:
            w = self.nt.get_mul_weight(cosine, label)
            output *= w.unsqueeze(dim=1).cuda()
        return (cosine, output)


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


if __name__ == '__main__':
    t = NoiseTolerant()

    t.l_bin_id_ = 60
    t.lt_bin_id_ = 80

    t.rt_bin_id_ = 110
    t.r_bin_id_ = 120
    # print(t.weight2(200))
    t.drow_pic(t.cos2weight, "3.jpg", True)
