from __future__ import division
from __future__ import print_function

import numpy as np
import torch


class NoiseTolerant(object):
    def __init__(self):
        self.consines = []
        self.slide_batch_num_ = 1000

        self.l_bin_id_ = 0
        self.r_bin_id_ = 0
        self.lt_bin_id_ = 0
        self.rt_bin_id_ = 0
        self.t_bin_ids_ = []

        self.bins_ = 200
        self.value_low_ = -1
        self.value_high_ = 1

        self.s = 1.0 - 0.99
        self.r = 2.576

        self.pdf_ = [0] * (self.bins_ + 1)
        self.fr = 2

        self.noise_ratio_ = 0

    def clamp(self, val, min_val, max_val):
        val = torch.max(torch.as_tensor(val, dtype=torch.float32), torch.as_tensor(min_val, dtype=torch.float32))
        val = torch.min(val, torch.as_tensor(max_val, dtype=torch.float32))
        return val

    def get_bin_id(self, cos):
        bin_id = self.bins_ * (cos - self.value_low_) / (self.value_high_ - self.value_low_)
        bin_id = self.clamp(bin_id, torch.Tensor([0]), torch.Tensor([self.bins_]))
        return bin_id

    def get_cos(self, bin_id):
        cos = self.value_low_ + (self.value_high_ - self.value_low_) * bin_id / self.bins_
        cos = self.clamp(cos, self.value_low_, self.value_high_)
        return cos

    def softplus(self, x):
        return torch.log(1.0 + torch.exp(torch.as_tensor(x)))
        # return torch.exp(torch.as_tensor(x))

    def weight2(self, bin_id):
        z = (bin_id - self.lt_bin_id_) / (self.r_bin_id_ - self.lt_bin_id_)
        upper = self.softplus(10.0 * z)
        down = self.softplus(10.0)
        weight2 = self.clamp(upper / down, 0.0, 1.0)
        return weight2

    def weight3(self, bin_id):
        a = ((self.r_bin_id_ - self.rt_bin_id_) / self.r) if bin_id > self.rt_bin_id_ else  ((self.rt_bin_id_ - self.l_bin_id_) / self.r)
        weight3 = torch.exp(-1.0 * (bin_id - self.rt_bin_id_) * (bin_id - self.rt_bin_id_) / (2 * a * a))
        return weight3

    def alpha(self, r_cos=None):
        if r_cos is None:
            r_cos = self.clamp(self.get_cos(self.r_bin_id_), 0.0, 1.0)
        alpha = 2.0 - 1.0 / (1.0 + torch.exp(5 - 20 * r_cos)) - 1.0 / (1.0 + torch.exp(20 * r_cos - 15))
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
            x_torch = torch.Tensor([x[i]])
            y[i] = func(x_torch).data.numpy()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        plt.plot(x, y, color="red", linewidth=1)
        plt.title('余弦柱状图')
        ###保存
        plt.savefig(file_path)

    def get_mul_weight(self, consines, labels):
        batch_size = len(consines)
        consines = consines.gather(1, labels.unsqueeze(dim=1).long())
        consines = consines.data.cpu()
        self.consines.append(consines)

        # add
        for consine in consines:
            bin_id = self.get_bin_id(consine)
            self.pdf_[bin_id] += 1

        if len(self.pdf_) < self.slide_batch_num_:
            return torch.ones(batch_size)

        # del
        while len(self.pdf_) > self.slide_batch_num_:
            del_consines = self.pdf_.pop(0)
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
            if pcf_[i] > 1.0 - self.s * 0.5:
                break
            self.r_bin_id_ = i
        if self.l_bin_id_ >= self.r_bin_id_:
            return torch.ones(batch_size)

        m_bin_id_ = (self.l_bin_id_ + self.r_bin_id_) / 2
        t_bin_id_ = torch.topk(filter_pdf, 1)[1]

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
        return weights
