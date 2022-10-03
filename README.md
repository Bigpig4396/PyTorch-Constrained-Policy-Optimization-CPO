# PyTorch-Constrained-Policy-Optimization-CPO

CPO for discrete action, change the original loss to something like policy gradient (original TRPO loss is cool in math but bad in performance). Maybe upload continuous action version in the future (if I am happy). Step function of environment should also return a cost value. More dangerous action should return higher cost.

https://arxiv.org/abs/1705.10528


inherit some functions from https://github.com/ajlangley/cpo-pytorch


Tune self.max_kl, the smaller the value, the slower the training speed, but more stable. If you have stability problem, decrease this parameter. 


self.max_J_c can be increased a little.
