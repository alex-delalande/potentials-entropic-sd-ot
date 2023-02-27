#!/usr/bin/env python
# coding: utf-8

# # Behavior of $\epsilon \mapsto \psi^\epsilon$

# In[1]:


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

parameters = {'axes.labelsize': 20,
             'axes.titlesize': 27,
             'legend.fontsize': 20}
plt.rcParams.update(parameters)


# # Resolution of unregularized OT
# 
# Let $\rho$ be a probability density on $[m, M]$ and $\mu$ a discrete probability measure with support of size $N$.
# 
# We solve the following optimization problem:
# 
# $$ (1) \qquad \min_{\psi \in \mathbb{R}^N} F(\psi), $$
# where $$F(\psi) = \int_m^M \psi^*(x) \rho(x)dx + \langle \psi, \mu \rangle.$$
# The function $F$ is convex on $\mathbb{R}^N$ and stricly convex on $(\mathbb{1}_N)^\perp$. Its gradient reads:
# $$ \frac{ \partial F}{\partial \psi_i} (\psi) = \mu_i - \int_{Lag_i(\psi)} \rho(x) dx.$$
# 
# Both the cost function $F$ and the gradient $\nabla F = (\frac{ \partial F}{\partial \psi_i})_{i=1, \dots, N}$ are computed using ```scipy.integrate.quad```.
# 
# We solve $(1)$ using ```scipy.optimize.minimize``` with the BFGS method.

# In[2]:


def h(x, j, psi, rho, b, Y_target):
    if j==np.argmax(x*Y_target - psi): pi=1
    else: pi=0
    return (b[j] - pi)*rho(x)

def f(x, psi, rho, b, Y_target):
    return np.max(x*Y_target - psi)*rho(x)

def get_cost_unreg(psi, rho, m, M, b, Y_target):
    A = quad(f, m, M, args=(psi, rho, b, Y_target), limit=100, epsabs=1e-18, epsrel=1e-18)[0]
    B = np.sum(b*psi)
    return A + B

def gradient_unreg(psi, rho, m, M, b, Y_target):
    grad = np.zeros(len(Y_target))
    for j in range(len(Y_target)):
        grad[j] = quad(h, m, M, args=(j, psi, rho, b, Y_target), limit=100, epsabs=1e-18, epsrel=1e-18)[0]
    return grad

def solve_ot_unreg(psi_init, rho, m, M, b, Y_target, disp=False):
    res = minimize(get_cost_unreg, psi_init, args=(rho, m, M, b, Y_target), method='L-BFGS-B',                   jac=gradient_unreg, tol=None, options={'ftol': 1e-12, 'gtol': 1e-12})

    if disp:
        print("Final gradient:", gradient_unreg(res['x'], rho, m, M, b, Y_target))
        print("Cost:", get_cost_unreg(res['x'], rho, m, M, b, Y_target))
        
    return res['x'] - (1/len(b))*np.sum(res['x'])


# # Resolution of regularized OT
# 
# Here we solve the following optimization problem:
# 
# $$ (2) \qquad \min_{\psi \in \mathbb{R}^N} F^\epsilon(\psi), $$
# where $$F^\epsilon(\psi) = \int_{m}^M \epsilon \log \left( \sum_{i=1}^N e^{\frac{xy_i - \psi_i}{\epsilon}} \right) \rho(x)dx + \langle \psi, \mu \rangle + \epsilon = \int_{m}^M  f^\epsilon(x, \psi) dx + \langle \psi, \mu \rangle + \epsilon.$$
# The function $F^\epsilon$ is convex on $\mathbb{R}^N$ and stricly convex on $(\mathbb{1}_N)^\perp$. Its gradient reads:
# $$ \frac{ \partial F^\epsilon}{\partial \psi_i} (\psi) = \mu - \int_0^1 \frac{ e^{\frac{xy_i - \psi_i}{\epsilon}} }{ \sum_{j=1}^N e^{\frac{xy_j - \psi_j}{\epsilon}} } \rho(x) dx.$$
# 
# Both the cost function $F^\epsilon$ and the gradient $\nabla F^\epsilon = (\frac{ \partial F^\epsilon}{\partial \psi_i})_{i=1, \dots, N}$ are computed using ```scipy.integrate.quad```.
# 
# We solve $(2)$ using ```scipy.optimize.minimize``` with the BFGS method.

# In[3]:


def hr(x, j, psi, eps, rho, b, Y_target):
    r = (x*Y_target - psi)/eps
    r -= np.max(r)
    pi = np.exp(r)/np.sum(np.exp(r))
    return (b - pi).flatten()[j]*rho(x)

def fr(x, psi, eps, rho, b, Y_target):
    r = x*Y_target - psi
    max_r = np.max(r)
    r -= max_r
    return (max_r + eps*np.log(np.sum(np.exp(r/eps))))*rho(x)

def get_cost_reg(psi, eps, rho, m, M, b, Y_target):
    A = quad(fr, m, M, args=(psi, eps, rho, b, Y_target), limit=100, epsabs=1e-18, epsrel=1e-18)[0]
    B = np.sum(b*psi)
    return A + B + eps

def gradient_reg(psi, eps, rho, m, M, b, Y_target):
    grad = np.zeros(len(Y_target))
    for j in range(len(Y_target)):
        grad[j] = quad(hr, m, M, args=(j, psi, eps, rho, b, Y_target), limit=100, epsabs=1e-18, epsrel=1e-18)[0]
    return grad

def solve_ot_reg(psi_init, eps, rho, m, M, b, Y_target, disp=False):
    res = minimize(get_cost_reg, psi_init, args=(eps, rho, m, M, b, Y_target),                    method='L-BFGS-B', jac=gradient_reg, options={'ftol': 1e-16, 'gtol': 1e-16,                                                                 'maxiter':1000000, 'maxfun': 1000000,                                                                 'maxls':50})
    if disp:
        print("Final gradient:", gradient_reg(res['x'], eps, rho, m, M, b, Y_target))
        print("Cost:", get_cost_reg(res['x'], eps, rho, m, M, b, Y_target))
        
    return res['x'] - (1/len(b))*np.sum(res['x'])


# # Computation of $\dot{\psi^\epsilon}$

# Compute the *true* derivatives using the formula
# $$ \dot{\psi^\epsilon} = - \left( \int_m^M M(\pi^\epsilon_x) \rho(x) dx \right)^{-1} \left( \int_m^M M(\pi^\epsilon_x) \log \pi^\epsilon_x \rho(x) dx \right),$$
# where $M(\pi^\epsilon_x) = \mathrm{diag}(\pi^\epsilon_x) - \pi^\epsilon_x (\pi^\epsilon_x)^\top$.

# In[4]:


def M_ij(x, i, j, psi, eps, rho, b, Y_target): 
    r = (x*Y_target - psi)/eps
    r -= np.max(r)
    pi = np.exp(r)/np.sum(np.exp(r))
    if i==j:
        return pi[i]*(1 - pi[i])*rho(x)
    else:
        return -pi[i]*pi[j]*rho(x)
    
def Mlog_i(x, i, psi, eps, rho, b, Y_target): 
    n_target = len(Y_target)
    r = (x*Y_target - psi)/eps
    r -= np.max(r)
    pi = np.exp(r)/np.sum(np.exp(r))
    M_i = np.zeros(n_target)
    for j in range(n_target):
        if j==i: M_i[j] = pi[i]*(1 - pi[i])
        else: M_i[j] = -pi[i]*pi[j]
    return np.sum(M_i*r)*rho(x)

def Hessian_K_eps(psi, eps, rho, m, M, b, Y_target):
    n_target = len(Y_target)
    mat = np.zeros((n_target, n_target))
    for i in range(n_target):
        mat[i, i] = quad(M_ij, m, M, args=(i, i, psi, eps, rho, b, Y_target), limit=100,                         epsabs=1e-12, epsrel=1e-12)[0]
        for j in range(i+1, n_target):
            mat[i, j] = quad(M_ij, m, M, args=(i, j, psi, eps, rho, b, Y_target), limit=100,                             epsabs=1e-12, epsrel=1e-12)[0]
            mat[j, i] = mat[i, j]
    return (1/eps)*mat

def der_epsilon_K_eps(psi, eps, rho, m, M, b, Y_target):
    n_target = len(Y_target)
    vect = np.zeros(n_target)
    for i in range(n_target):
        vect[i] = quad(Mlog_i, m, M, args=(i, psi, eps, rho, b, Y_target), limit=100,                       epsabs=1e-12, epsrel=1e-12)[0]
    return (1/eps)*vect

def get_dot_psi(psi, eps, rho, m, M, b, Y_target):
    mat = Hessian_K_eps(psi, eps, rho, m, M, b, Y_target)
    vect = der_epsilon_K_eps(psi, eps, rho, m, M, b, Y_target)
    dot_psi = np.linalg.lstsq(mat, -vect)[0] 
    return dot_psi - (1/n_target)*np.sum(dot_psi)


# # Lebesgue
# 
# * $\rho$ is a Lebesgue distribution
# * $\mu = \frac{1}{5} \sum_{i=1}^{5} \delta_{y_i}$ is a uniform discrete measure with support of size $5$ included in $[m, M]$. We denote $N=5$.

# In[5]:


m, M = 0, 0.35 

integrand = lambda u: np.ones_like(u)
int_integrand = quad(integrand, m, M)[0]

def rho(x):    
    return integrand(x)/int_integrand

plt.rcParams["figure.figsize"] = (8, 5)
xx = np.linspace(m, M, 100)
plt.plot(xx, rho(xx))
plt.xlabel("$x$")
plt.ylabel(r"$\rho(x)$")
plt.show()


# In[6]:


n_target = 5
b = np.ones(n_target)/n_target
rng = np.random.RandomState(42)
Y_target = np.sort(rng.rand(n_target))
Y_target = (M-m)*Y_target + m
print("Y =", Y_target)

plt.scatter(Y_target, np.zeros(n_target), label="Y")
plt.plot(Y_target, np.zeros(n_target), c="k", lw=0.5)
plt.legend()
plt.show()


# Solve unregularized OT:

# In[7]:


psi_init = np.zeros(len(b))
psi_0 = solve_ot_unreg(psi_init, rho, m, M, b, Y_target)

print("psi_0:", psi_0)


# Test resolution of regularized OT with $\epsilon$ small:

# In[8]:


psi_init = np.zeros(len(b))
eps = 1e-3

psi_eps = solve_ot_reg(psi_init, eps, rho, m, M, b, Y_target)

print("psi_eps:", psi_eps)


# Observe the functions $x \mapsto \pi^\epsilon_{i, x}$ for $i \in \{1, \dots, N\}$:

# In[9]:


def pi_eps(x, i, psi, eps, b, Y_target):
    r = (x*Y_target - psi)/eps
    r -= np.max(r)
    pi = np.exp(r)/np.sum(np.exp(r))
    return pi[i]

xx = np.linspace(m, M, 100)
for i in range(n_target):
    pi_i = np.array([pi_eps(x, i, psi_eps, eps, b, Y_target) for x in xx])
    title = r"$x \mapsto \pi^\epsilon_{" + str(i+1) + ", x}$"
    plt.plot(xx, pi_i, label=title)
    
plt.scatter(Y_target, np.zeros(n_target), label="Y", lw=4, c="k")
plt.legend()  
plt.show()


# ## Behavior for $\epsilon \to 0$:

# In[10]:


nb_eps = 30
epsilons = np.logspace(-3.5, 1.3, nb_eps) 
log_psi = np.zeros((n_target, nb_eps))

for i in range(nb_eps):
    eps = epsilons[nb_eps - 1 - i]
    print("i = {} - epsilon = {}".format(i, eps))
    
    # initialize potential
    if i==0: psi_init = np.zeros(n_target)
    else: psi_init = log_psi[:, nb_eps - i].copy()
    
    # compute potential 
    log_psi[:, nb_eps - 1 - i] = solve_ot_reg(psi_init, eps, rho, m, M, b, Y_target)


# ### Observe behavior w.r.t. $\epsilon$
# 
# We can first observe the rate of convergence of $\psi^\epsilon$ to $\psi^0$ as $\epsilon$ goes to $0$.

# In[11]:


differences_psi = np.linalg.norm(log_psi - psi_0.reshape((-1, 1)), axis=0, ord=np.inf)

comparison = np.array([8e1*eps**(2) for eps in epsilons])
plt.loglog(epsilons[:15], comparison[:15], "--", label=r"$C \epsilon^{2}$")

plt.loglog(epsilons, differences_psi, label=r"$||\psi^\epsilon - \psi^0||_\infty$")
plt.xlabel(r"$\epsilon$")
plt.title(r"Lebesgue")
plt.legend()
plt.savefig('outputs/psi-eps-psi-0-lebesgue.pdf', bbox_inches='tight')   
plt.show()


# Observe Theorem 3.2:

# In[12]:


v = np.random.rand(n_target)
var_v = np.var(v)

hessian_products = np.zeros(nb_eps)
for i in range(nb_eps):
    hessian_products[i] = v.dot(Hessian_K_eps(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target).dot(v))
    
plt.loglog(epsilons, 2e0*hessian_products, label=r'$C \langle v, \nabla^2 \mathcal{K}^\epsilon (\psi^\epsilon) v\rangle$')
plt.loglog(epsilons, var_v/(np.exp(1)+epsilons), label=r'$\frac{1}{e^1 + \epsilon} \mathrm{Var}_\mu(v)$')
plt.legend()
#plt.savefig('outputs/strong-convexity-lebesgue.pdf', bbox_inches='tight')  
plt.show()


# Observe Theorem 3.3:

# In[13]:


log_der_epsilon_K_eps = np.zeros((n_target, nb_eps))
for i in range(nb_eps):
    log_der_epsilon_K_eps[:, i] = der_epsilon_K_eps(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target) 
    
plt.loglog(epsilons[:10], 2e4*epsilons[:10]**(1), "--", label=r"$C \epsilon^{1}$")
plt.loglog(epsilons[3:], 1e-1/epsilons[3:], "--", label=r"$C'/\epsilon$")

norm_der_epsilon_K_eps = np.linalg.norm(log_der_epsilon_K_eps, axis=0, ord=np.inf)
plt.loglog(epsilons, norm_der_epsilon_K_eps, label=r"$||\frac{\partial}{\partial \epsilon}(\nabla \mathcal{K}^\epsilon)(\psi^\epsilon)||_\infty$")
plt.xlabel(r"$\epsilon$")
plt.legend()
#plt.savefig('outputs/bound-second-term-ode-lebesgue.pdf', bbox_inches='tight')  
plt.show()


# Observe Theorem 2.1:

# In[14]:


log_dot_psi = np.zeros((n_target, nb_eps))
for i in range(nb_eps):
    log_dot_psi[:, i] = get_dot_psi(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target)

plt.loglog(epsilons[:11], 2e2*epsilons[:11]**(1), "--", label=r"$C\epsilon^{1}$")

norm_derivatives_psi = np.linalg.norm(log_dot_psi, axis=0, ord=np.inf)
plt.loglog(epsilons, norm_derivatives_psi, label=r"$||\dot{\psi^{\epsilon}}||_\infty$")
plt.xlabel(r"$\epsilon$")
plt.title(r"Lebesgue")
plt.legend()
plt.savefig('outputs/dot-psi-lebesgue.pdf', bbox_inches='tight')  
plt.show()


# Observe exponential convergence of $\epsilon \mapsto \pi^\epsilon_{x,i}$ for all $i \in \{1, \dots, N\}$ and $\rho$-almost-every $x \in \mathcal{X}$.

# In[15]:


x = (M-m)*rng.rand() + m

print("x = {}".format(x))
log_pi_x = np.zeros((n_target, nb_eps))

for e in range(nb_eps):
    for i in range(n_target):
        log_pi_x[i, e] = pi_eps(x, i, log_psi[:, e], epsilons[e], b, Y_target)
        

plt.figure(figsize=(14, 7))
           
for i in range(n_target):
    plt.subplot(2, 3, i+1)
    plt.loglog(epsilons, log_pi_x[i])
    plt.title(r"$\epsilon \mapsto \pi^\epsilon_{" + str(i+1) + ", x} - x = $" + str(round(x, 3)))
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\pi^\epsilon_x$")

plt.subplot(2, 3, 6)
xx = np.linspace(m, M, 100)
for i in range(n_target):
    pi_i = np.array([pi_eps(x, i, log_psi[:, 0], epsilons[0], b, Y_target) for x in xx])
    title = r"$\pi^\epsilon_{" + str(i+1) + ", x}$" 
    plt.plot(xx, pi_i, label=title)
    
plt.title(r"$\epsilon = $" + str(epsilons[0]))

plt.tight_layout()
plt.legend()
#plt.savefig('outputs/pi-eps-x-vs-eps-0-lebesgue.pdf', bbox_inches='tight')  
plt.show()


# ## Behavior for $\epsilon \to \infty$ with $\mu$ non-uniform:

# In[16]:


b = np.random.rand(n_target)
b /= np.sum(b)
nb_eps = 30
epsilons = np.logspace(.5, 4, nb_eps)
log_psi = np.zeros((n_target, nb_eps))

for i in range(nb_eps):
    eps = epsilons[nb_eps - 1 - i]
    print("i = {} - epsilon = {}".format(i, eps))
    
    # initialize potential
    if i==0: psi_init = np.zeros(n_target)
    else: psi_init = log_psi[:, nb_eps - i].copy()
    
    # compute potential 
    log_psi[:, nb_eps - 1 - i] = solve_ot_reg(psi_init, eps, rho, m, M, b, Y_target)


# ### Observe behavior w.r.t. $\epsilon$

# Observe Theorem 3.2:

# In[17]:


v = np.random.rand(n_target)
var_v = np.var(v)

hessian_products = np.zeros(nb_eps)
for i in range(nb_eps):
    hessian_products[i] = v.dot(Hessian_K_eps(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target).dot(v))
    
plt.loglog(epsilons, 2e0*hessian_products, label=r'$C \langle v, \nabla^2 \mathcal{K}^\epsilon (\psi^\epsilon) v\rangle$')
plt.loglog(epsilons, var_v/(np.exp(1)+epsilons), label=r'$\frac{1}{e^1 + \epsilon} \mathrm{Var}_\mu(v)$')
plt.legend()
plt.show()


# Observe Theorem 3.3:

# In[18]:


log_der_epsilon_K_eps = np.zeros((n_target, nb_eps))
for i in range(nb_eps):
    log_der_epsilon_K_eps[:, i] = der_epsilon_K_eps(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target) 
    
plt.loglog(epsilons[3:], 1e-1/epsilons[3:], "--", label=r"$C/\epsilon$")

norm_der_epsilon_K_eps = np.linalg.norm(log_der_epsilon_K_eps, axis=0, ord=np.inf)
plt.loglog(epsilons, norm_der_epsilon_K_eps, label=r"$||\frac{\partial}{\partial \epsilon}(\nabla \mathcal{K}^\epsilon)(\psi^\epsilon)||_\infty$")
plt.xlabel(r"$\epsilon$")
plt.legend()
plt.show()


# Observe Theorem 2.1:

# In[19]:


log_dot_psi = np.zeros((n_target, nb_eps))
for i in range(nb_eps):
    log_dot_psi[:, i] = get_dot_psi(log_psi[:, i], epsilons[i], rho, m, M, b, Y_target)

norm_derivatives_psi = np.linalg.norm(log_dot_psi, axis=0, ord=np.inf)
plt.loglog(epsilons, norm_derivatives_psi, label=r"$||\dot{\psi^{\epsilon}}||_\infty$")
plt.xlabel(r"$\epsilon$")
plt.title(r"Lebesgue")
plt.legend()
plt.show()


# In[ ]:




