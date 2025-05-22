import torch
import numpy as np
import math


class AdversarialOptimizer(torch.optim.Optimizer):
    def __init__(self, params, data_len, batch_size, use_adam=True,
                 betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-3,
                 pi_decay=1e-3,  # eps=np.finfo(np.float32).eps,
                 eps=1e-20,
                 amsgrad=False,
                 mode='extragrad', log=False, pi_lr=None, pi_reg=None,
                 is_min_min = False,
                 use_momentum=False, momentum=0.9):

        defaults = dict(lr=lr, betas=betas, amsgrad=amsgrad)
        super(AdversarialOptimizer, self).__init__(params, defaults=defaults)
        
        assert not use_adam or not use_momentum

        self.use_adam = use_adam
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.lr = lr
        if pi_lr is None:
            self.pi_lr = lr
        else:
            self.pi_lr = pi_lr
        # print("pi_lr:", self.pi_lr)
        self.weight_decay = weight_decay

        self.loss_scale = data_len / batch_size # [TODO] ???
        # self.loss_scale = 1.

        if pi_reg is None:
            self.pi_reg = torch.ones(data_len, requires_grad=False) / data_len
        else:
            self.pi_reg = pi_reg / torch.sum(pi_reg)
        self.pi = self.pi_reg.clone()
        self.pi_decay = pi_decay
        # print("self.pi_decay:", self.pi_decay)
        self.__eps = eps
        # print("eps:", self.__eps)
        self.mode = mode
        self.log = log
        self.loss = None

        # Min-min
        self.is_min_min = is_min_min

        self.__previous_params = None
        self.__pi_intermediate = None
        if self.mode == 'extragrad':
            self.__pi_intermediate = self.pi_reg
        self.__prev_grads_pi = None
        self.__prev_grads_theta = []
        self.logs = {}

        self.__called_main_step = False

    def __move_pi_to_device(self, device):
        self.pi = self.pi.to(device)
        self.pi_reg = self.pi_reg.to(device)
        if self.__pi_intermediate is not None:
            self.__pi_intermediate = self.__pi_intermediate.to(device)

    def __add_logs(self):
        current_loss = self.loss
        kl_1 = (self.pi * torch.log(self.pi / (self.pi_reg))).mean().item()
        kl_2 = (self.pi_reg * torch.log(self.pi_reg / (self.pi))).mean().item()
        max_ape = (torch.abs(self.pi - self.pi_reg) / (self.pi_reg)).max().item()
        mean_ape = (torch.abs(self.pi - self.pi_reg) / (self.pi_reg)).mean().item()

        if len(self.logs.keys()) == 0:
            self.logs['loss'] = [current_loss]
            self.logs['kl_1'] = [kl_1]
            self.logs['kl_2'] = [kl_2]
            self.logs['max_ape'] = [max_ape]
            self.logs['mean_ape'] = [mean_ape]
        else:
            self.logs['loss'].append(current_loss)
            self.logs['kl_1'].append(kl_1)
            self.logs['kl_2'].append(kl_2)
            self.logs['max_ape'].append(max_ape)
            self.logs['mean_ape'].append(mean_ape)

    def __setstate__(self, state):
        super(AdversarialOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def __adam_grad(self, p, group, suffix=1):
        if p.grad is None:
            return None
        grad = p.grad.data
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if f'step_{suffix}' not in state:
            state[f'step_{suffix}'] = 0
            # Exponential moving average of gradient values
            state[f'exp_avg_{suffix}'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state[f'exp_avg_sq_{suffix}'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state[f'max_exp_avg_sq_{suffix}'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state[f'exp_avg_{suffix}'], state[f'exp_avg_sq_{suffix}']
        if amsgrad:
            max_exp_avg_sq = state[f'max_exp_avg_sq_{suffix}']
        beta1, beta2 = group['betas']

        state[f'step_{suffix}'] += 1

        if self.weight_decay != 0:
            grad = grad.add(self.weight_decay, p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(self.__eps)
        else:
            denom = exp_avg_sq.sqrt().add_(self.__eps)

        bias_correction1 = 1 - beta1 ** state[f'step_{suffix}']
        bias_correction2 = 1 - beta2 ** state[f'step_{suffix}']
        step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

        return -step_size * exp_avg / denom
    
    def __sgd_grad(self, p, group):
        if p.grad is None:
            return None
        
        d_p = p.grad.data

        if self.weight_decay != 0:
            d_p = d_p.add(p.data, alpha=self.weight_decay)

        if self.use_momentum and self.momentum != 0:
            state = self.state[p]
            if 'momentum_buffer' not in state:
                buf = torch.clone(d_p).detach()
                state['momentum_buffer'] = buf
            else:
                buf = state['momentum_buffer']
                buf.mul_(self.momentum).add_(d_p)
            d_p = buf
        
        return -d_p * self.lr

    def __extragrad_intermediate_step(self, closure, dataset_indexes):
        pi_selected = self.pi[dataset_indexes]
        losses, self.loss = closure(pi_selected, self.loss_scale)
        self.__previous_params = []
        for group in self.param_groups:
            for p in group['params']:
                self.__previous_params.append(p.data)
                if self.use_adam:
                    p.data = p.data + self.__adam_grad(p, group, suffix=0.5)
                elif self.use_momentum:
                    p.data = p.data + self.__sgd_grad(p, group)
                else:
                    p.data = p.data - self.lr * (p.grad.data + self.weight_decay * p.data)

        if not self.is_min_min:
            losses = -losses.clone().detach()
        else:   
            losses = +losses.clone().detach()  

        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad[dataset_indexes] = losses
        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(
            self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.__pi_intermediate = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def __extragrad_main_step(self, closure, dataset_indexes):
        pi_selected = self.__pi_intermediate[dataset_indexes]
        losses, self.loss = closure(pi_selected, self.loss_scale)
        t = 0
        for group in self.param_groups:
            for p in group['params']:
                if self.use_adam:
                    p.data = self.__previous_params[t] + self.__adam_grad(p, group, suffix=1)
                elif self.use_momentum:
                    p.data = self.__previous_params[t] + self.__sgd_grad(p, group)
                else:
                    p.data = self.__previous_params[t] - self.lr * (p.grad.data + self.weight_decay * p.data)
                t += 1

        if not self.is_min_min:
            losses = -losses.clone().detach()
        else:   
            losses = +losses.clone().detach()  
        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad[dataset_indexes] = losses
        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(
            self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.pi = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def __slowed_optimistic_step(self, closure, dataset_indexes):
        pi_selected = self.pi[dataset_indexes]
        losses, self.loss = closure(pi_selected, self.loss_scale)

        t = 0
        for group in self.param_groups:
            for p in group['params']:
                current_grad = p.grad.data
                if len(self.__prev_grads_theta) < t + 1:
                    grad = current_grad
                else:
                    grad = current_grad - 0.5 * self.__prev_grads_theta[t]
                self.__prev_grads_theta.append(current_grad)

                if self.use_adam:
                    p.grad.data = current_grad
                    p.data = p.data + self.__adam_grad(p, group)
                elif self.use_momentum:
                    p.grad.data = current_grad
                    p.data = p.data + self.__sgd_grad(p, group)
                else:
                    p.data = p.data - self.lr * (current_grad + self.weight_decay * p.data)
                t += 1
        self.__prev_grads_theta = self.__prev_grads_theta[:t]

        if not self.is_min_min:
            losses = -losses.clone().detach()
        else:   
            losses = +losses.clone().detach()  

        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad[dataset_indexes] = losses
        if self.__prev_grads_pi is not None:
            prev = self.__prev_grads_pi
            self.__prev_grads_pi = pi_grad.clone()
            pi_grad = pi_grad - 0.5 * prev
        else:
            self.__prev_grads_pi = pi_grad.clone()

        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(
            self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.pi = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def __optimistic_intermediate_step(self, closure, dataset_indexes):
        t = 0
        self.__previous_params = []
        for group in self.param_groups:
            for p in group['params']:
                self.__previous_params.append(p.data)
                if len(self.__prev_grads_theta) > 0:
                    p.grad.data = self.__prev_grads_theta[t]
                    t += 1
                    if self.use_adam:
                        p.data = p.data + self.__adam_grad(p, group)
                    elif self.use_momentum:
                        p.data = p.data + self.__sgd_grad(p, group)
                    else:
                        p.data = p.data - self.lr * (p.grad.data + self.weight_decay * p.data)
        if self.__prev_grads_pi is not None:
            pi_grad = self.__prev_grads_pi
            pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(
                self.pi_reg) - self.pi_lr * pi_grad
            pi_new_log /= 1 + self.pi_decay * self.pi_lr
            self.__pi_intermediate = torch.nn.functional.softmax(pi_new_log, dim=-1)
        else:
            self.__pi_intermediate = self.pi.clone()

    def __optimistic_main_step(self, closure, dataset_indexes):
        pi_selected = self.__pi_intermediate[dataset_indexes]
        losses, self.loss = closure(pi_selected, self.loss_scale)
        t = 0
        self.__prev_grads_theta = []
        for group in self.param_groups:
            for p in group['params']:
                self.__prev_grads_theta.append(p.grad.data)
                if self.use_adam:
                    p.data = self.__previous_params[t] + self.__adam_grad(p, group)
                elif self.use_momentum:
                    p.data = self.__previous_params[t] + self.__sgd_grad(p, group)
                else:
                    p.data = self.__previous_params[t] - self.lr * (p.grad.data + self.weight_decay * p.data)
                t += 1
        if not self.is_min_min:
            losses = -losses.clone().detach()
        else:   
            losses = +losses.clone().detach()  
        self.__prev_grads_pi
        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad[dataset_indexes] = losses
        self.__prev_grads_pi = pi_grad
        pi_new_log = torch.log(self.pi + self.__eps) + self.pi_decay * self.pi_lr * torch.log(
            self.pi_reg) - self.pi_lr * pi_grad
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        self.pi = torch.nn.functional.softmax(pi_new_log, dim=-1)

    def step(self, closure, dataset_indexes):
        self.__move_pi_to_device(closure.device)
        if self.mode == 'extragrad':
            self.__extragrad_intermediate_step(closure, dataset_indexes)
            self.__extragrad_main_step(closure, dataset_indexes)
        if self.mode == 'optimistic':
            self.__optimistic_intermediate_step(closure, dataset_indexes)
            self.__optimistic_main_step(closure, dataset_indexes)
        if self.mode == 'slowed-optimistic':
            self.__slowed_optimistic_step(closure, dataset_indexes)
        if self.log:
            self.__add_logs()
