from env_cost import EnvGoTogether
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.autograd import grad
from torch.distributions.kl import kl_divergence

def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9, max_iter=10):
    '''
    Perform a backtracking line search that terminates when constraints_satisfied
    return True and return the calculated step length. Return 0.0 if no step
    length can be found for which constraints_satisfied returns True

    Parameters
    ----------
    search_dir : torch.FloatTensor
        the search direction along which the line search is done

    max_step_len : torch.FloatTensor
        the maximum step length to consider in the line search

    constraints_satisfied : callable
        a function that returns a boolean indicating whether the constraints
        are met by the current step length

    line_search_coef : float
        the proportion by which to reduce the step length after each iteration

    max_iter : int
        the maximum number of backtracks to do before return 0.0

    Returns
    -------
    the maximum step length coefficient for which constraints_satisfied evaluates
    to True
    '''

    step_len = max_step_len / line_search_coef

    for i in range(max_iter):
        step_len *= line_search_coef

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    return torch.tensor(0.0)

def flatten(vecs):
    '''
    Return an unrolled, concatenated copy of vecs

    Parameters
    ----------
    vecs : list
        a list of Pytorch Tensor objects

    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    '''

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened

def set_params(parameterized_fun, new_params):
    '''
    Set the parameters of parameterized_fun to new_params

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated

    update : torch.FloatTensor
        a flattened version of the parameters to be set
    '''

    n = 0
    for param in parameterized_fun.parameters():
        numel = param.numel()
        new_param = new_params[n:n + numel].view(param.size())
        param.data = new_param
        n += numel

def get_flat_params(parameterized_fun):
    '''
    Get a flattened view of the parameters of a function approximator

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator for which the parameters are to be returned

    Returns
    -------
    flat_params : torch.FloatTensor
        a flattened view of the parameters of parameterized_fun
    '''
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params

def cg_solver(Avp_fun, b, max_iter=10):
    '''
    Finds an approximate solution to a set of linear equations Ax = b

    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector

    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b

    max_iter : int
        the maximum number of iterations (default is 10)

    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    '''

    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p

def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    '''
    Return a flattened view of the gradients of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated

    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed

    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)

    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself

    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    '''

    if create_graph == True:
        retain_graph = True
    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = torch.cat([v.view(-1) for v in grads])
    return flat_grads

def detach_dist(dist):
    '''
    Return a copy of dist with the distribution parameters detached from the
    computational graph

    Parameters
    ----------
    dist: torch.distributions.distribution.Distribution
        the distribution object for which the detached copy is to be returned

    Returns
    -------
    detached_dist
        the detached distribution
    '''

    detached_dist = Categorical(logits=dist.logits.detach())
    return detached_dist

def mean_kl_first_fixed(dist_1, dist_2):
    '''
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph

    Parameters
    ----------
    dist_1 : torch.distributions.distribution.Distribution
        the first argument to the kl-divergence function (will be fixed)

    dist_2 : torch.distributions.distribution.Distribution
        the second argument to the kl-divergence function (will not be fixed)

    Returns
    -------
    mean_kl : torch.float
        the kl-divergence between dist_1 and dist_2
    '''
    dist_1_detached = detach_dist(dist_1)
    mean_kl = torch.mean(kl_divergence(dist_1_detached, dist_2))
    return mean_kl

def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
    '''
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor (with requires_grad=True)
        the output of the function of which the Hessian is calculated

    inputs : torch.FloatTensor
        the inputs w.r.t. which the Hessian is calculated

    damping_coef : float
        the multiple of the identity matrix to be added to the Hessian
    '''

    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)
    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v
        return Hvp
    return Hvp_fun

class P_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(P_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_score = self.fc3(x)
        return F.softmax(action_score, dim=-1)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class CPO():
    def __init__(self, state_dim, action_dim):
        super(CPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_net = P_net(state_dim, action_dim)
        self.q_net = Q_net(state_dim, action_dim)
        self.c_net = Q_net(state_dim, action_dim)
        self.gamma = 0.99
        self.max_J_c = 0.1
        self.max_kl = 1e-2
        self.line_search_coef=0.9
        self.line_search_accept_ratio=0.1
        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)
        self.c_optimizer = torch.optim.Adam(self.c_net.parameters(), lr=1e-2)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=1e-3)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.p_net.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def train_CPO(self, state_list, action_list, prob_list, reward_list, cost_list, next_state_list):
        # train cost
        state = state_list[0]
        next_state = next_state_list[0]
        for i in range(1, len(state_list)):
            state = np.vstack((state, state_list[i]))
            next_state = np.vstack((next_state, next_state_list[i]))
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        next_a_prob = self.p_net.forward(next_state)
        for epoch in range(5):
            q = self.q_net.forward(state)
            next_q = self.q_net.forward(next_state)
            expect_q = q.clone()
            for i in range(len(state_list)):
                expect_q[i, action_list[i]] = reward_list[i] + self.gamma * torch.sum(next_a_prob[i, :] * next_q[i, :])
            loss = self.loss_fn(q, expect_q.detach())
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()

            q_c = self.c_net.forward(state)
            next_q_c = self.c_net.forward(next_state)
            expect_q_c = q_c.clone()
            for i in range(len(state_list)):
                expect_q_c[i, action_list[i]] = cost_list[i] + self.gamma * torch.sum(next_a_prob[i, :] * next_q_c[i, :])
            loss = self.loss_fn(q_c, expect_q_c.detach())
            self.c_optimizer.zero_grad()
            loss.backward()
            self.c_optimizer.step()

        q = self.q_net.forward(state)
        q_c = self.c_net.forward(state)
        a_prob = self.p_net.forward(state)
        v = torch.sum(a_prob * q, 1)
        v_c = torch.sum(a_prob * q_c, 1)
        gae = torch.zeros(len(state_list), )
        gae_c = torch.zeros(len(state_list), )
        for i in range(len(state_list)):
            gae[i] = q[i, action_list[i]] - v[i]
            gae_c[i] = q_c[i, action_list[i]] - v_c[i]

        # train policy
        log_a_probs = torch.zeros(len(state_list), )
        for i in range(len(state_list)):
            log_a_probs[i] = torch.log(a_prob[i, action_list[i]])
        r_loss = torch.mean(gae *log_a_probs)
        g = flat_grad(r_loss, self.p_net.parameters(), retain_graph=True)
        c_loss = torch.mean(gae_c *log_a_probs)
        b = flat_grad(c_loss, self.p_net.parameters(), retain_graph=True)

        action_dists = Categorical(probs=a_prob)
        mean_kl = mean_kl_first_fixed(action_dists, action_dists)  # kl-divergence between dist_1 and dist_2
        Fvp_fun = get_Hvp_fun(mean_kl, self.p_net.parameters())
        H_inv_g = cg_solver(Fvp_fun, g)    # H**-1*g
        H_inv_b = cg_solver(Fvp_fun, b)    # H**-1*b
        q = torch.matmul(g, H_inv_g)
        r = torch.matmul(g, H_inv_b)
        s = torch.matmul(b, H_inv_b)

        J_c = 0
        for i in range(len(state_list)-1, -1, -1):
            J_c = cost_list[i] + self.gamma * J_c
        c = (J_c - self.max_J_c)

        is_feasible = False if c > 0 and c ** 2 / s - self.max_kl > 0 else True
        if is_feasible:
            lam, nu = self.calc_dual_vars(q, r, s, c)
            search_dir = lam ** -1 * (H_inv_g + nu * H_inv_b)
        else:
            search_dir = -torch.sqrt(2 * self.max_kl / s) * H_inv_b

        # Should be positive
        current_policy = get_flat_params(self.p_net)
        new_policy = current_policy + search_dir
        set_params(self.p_net, new_policy)

    def calc_dual_vars(self, q, r, s, c):
        if c < 0.0 and c ** 2 / s - self.max_kl > 0.0:
            lam = torch.sqrt(q / (self.max_kl))
            nu = 0.0
            return lam, nu
        A = q - r ** 2 / s
        B = self.max_kl - c ** 2 / s
        lam_mid = r / c
        lam_a = torch.sqrt(A / B)
        lam_b = torch.sqrt(q / (self.max_kl))
        f_mid = -0.5 * (q / lam_mid + lam_mid * self.max_kl)
        f_a = -torch.sqrt(A * B) - r * c / s
        f_b = -torch.sqrt(q * self.max_kl)
        if lam_mid > 0:
            if c < 0:
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
            else:
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
        else:
            if c < 0:
                lam = lam_b
            else:
                lam = lam_a
        lam = lam_a if f_a >= f_b else lam_b
        nu = max(0.0, (lam * c - r) / s)
        return lam, nu

if __name__ == '__main__':
    state_dim = 169
    action_dim = 4
    max_epi = 1000
    max_mc = 500
    epi_iter = 0
    mc_iter = 0
    acc_reward = 0
    acc_cost = 0
    reward_curve = []
    env = EnvGoTogether(13)
    agent = CPO(state_dim, action_dim)
    for epi_iter in range(max_epi):
        state_list = []
        action_list = []
        prob_list = []
        reward_list = []
        cost_list = []
        next_state_list = []
        for mc_iter in range(max_mc):
            # env.render()
            state = np.zeros((env.map_size, env.map_size))
            state[env.agt1_pos[0], env.agt1_pos[1]] = 1
            state = state.reshape((1, env.map_size * env.map_size))
            action, action_prob = agent.get_action(state)
            group_list = [action, 2]
            reward, done, cost = env.step(group_list)
            next_state = np.zeros((env.map_size, env.map_size))
            next_state[env.agt1_pos[0], env.agt1_pos[1]] = 1
            next_state = next_state.reshape((1, env.map_size * env.map_size,))
            acc_reward += reward
            acc_cost += cost
            state_list.append(state)
            action_list.append(action)
            prob_list.append(action_prob)
            reward_list.append(reward)
            cost_list.append(cost)
            next_state_list.append(next_state)
            if done:
                break
        agent.train_CPO(state_list, action_list, prob_list, reward_list, cost_list, next_state_list)
        print('epi', epi_iter, 'reward', acc_reward / mc_iter, 'cost', acc_cost / mc_iter, 'MC', mc_iter)
        env.reset()
        acc_reward = 0
        acc_cost = 0