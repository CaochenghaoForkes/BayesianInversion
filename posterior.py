import numpy as np

from prior import Prior
from model import Model
from likelihood import Likelihood

class PosteriorCalculator:
    """对数后验计算器"""

    def __init__(self,prior:Prior,model:Model,likelihood:Likelihood):
        self.prior = prior
        self.model = model
        self.likelihood = likelihood

    def __call__(self,theta:dict[str,float],procedure_file_label:str) -> float:
        return log_posterior(theta,self.prior,self.model,self.likelihood,procedure_file_label)

def log_posterior(
    theta: dict[str,float],
    prior: Prior,
    model: Model,
    likelihood: Likelihood,
    procedure_file_label: str
) -> float:
    """后验"""

    # 计算先验
    log_prior = prior.log_prior(theta)
    if not np.isfinite(log_prior):
        return -np.inf
    # 计算预测值
    model_result = model.model_forward(theta,procedure_file_label)
    # 计算似然
    log_likelihood = likelihood.log_likelihood(model_result)
    if not np.isfinite(log_likelihood):
        return -np.inf

    return float(log_prior + log_likelihood)
