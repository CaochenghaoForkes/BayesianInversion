"""贝叶斯反演入口"""

from pathlib import Path

from config import JsonReader
from data import Data
from prior import Prior
from model import Model
from likelihood import Likelihood
from posterior import PosteriorCalculator
from sampler import Sampler
from output import Output

PROJECT_DIR = Path(__file__).resolve().parent

def main(input_path:str | Path) -> None:

    # 读取输入
    input_card_path = Path(input_path)
    if not input_card_path.is_absolute():
        input_card_path = PROJECT_DIR / input_card_path
    input_card = JsonReader.load(input_card_path)

    # 初始化
    data_obs = Data(input_card.data)
    prior = Prior(input_card.prior)
    model = Model(input_card.model)
    likelihood = Likelihood(input_card.likelihood,data_obs)
    sampler = Sampler(input_card.sampler)
    output = Output(input_card.output.output_dir)

    # 构造对数后验函数
    posterior_calculator = PosteriorCalculator(prior,model,likelihood)

    # 采样
    sampler_result = sampler.sample(input_card.parameters,posterior_calculator,input_card.random_seed)

    # 保存结果
    parameter_names = [config.name for config in input_card.parameters]
    output.save(parameter_names,sampler_result)

if __name__ == "__main__":
    main("input_card.json")
