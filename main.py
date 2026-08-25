"""Run the complete Bayesian-inversion workflow."""

import argparse
from pathlib import Path
from typing import Any

from config import Examine,InputCard,JsonReader
from data import Data
from likelihood import Likelihood
from model import Model
from output import Output
from postprocess import PostProcessor
from posterior import PosteriorCalculator
from prior import Prior
from sampler import Sampler


PROJECT_DIR = Path(__file__).resolve().parent


def load_input_card(input_path: str | Path) -> InputCard:
    """读取并解析JSON输入卡"""

    path = Path(input_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path

    return JsonReader.load(path)


def run_inversion(input_card: InputCard) -> dict[str,Path]:
    """执行一次完整的贝叶斯参数反演"""

    parameter_names = [config.name for config in input_card.parameters]
    observed_data = Data(input_card.data).load()
    Examine.examine_observed_data(input_card,observed_data.values)
    prior = Prior(input_card.prior,parameter_names)
    model = Model(input_card.model,parameter_names)
    likelihood = Likelihood(input_card.likelihood,observed_data)
    posterior = PosteriorCalculator(prior,model,likelihood)
    sampler = Sampler(input_card.sampler,input_card.study.name)

    sampler_result = sampler.sample(
        input_card.parameters,
        posterior,
        input_card.random_seed,
    )

    output = Output(input_card)
    return output.write(sampler_result)


def run_postprocess(input_card: InputCard) -> dict[str,Any]:
    """对输出目录中已有的贝叶斯结果绘图"""

    return PostProcessor(
        input_card.output.directory,
        separate_figures=input_card.postprocess.separate_figures,
    ).run()


def output_directory_available(input_card: InputCard) -> bool:
    """检查输出目录是否不存在或为空"""

    directory = Path(input_card.output.directory)
    if not directory.exists():
        return True
    if not directory.is_dir():
        raise NotADirectoryError(f"输出路径不是目录: {directory}")

    return not any(directory.iterdir())


def next_output_directory(directory: Path) -> Path:
    """为新一次反演生成未占用的目录名"""

    suffix = 2
    while True:
        candidate = directory.with_name(
            f"{directory.name}_{suffix}"
        )
        if not candidate.exists():
            return candidate
        if candidate.is_dir() and not any(candidate.iterdir()):
            return candidate
        suffix += 1


def prompt_choice(prompt: str,choices: set[str]) -> str:
    """读取一个合法的交互选项"""

    while True:
        try:
            choice = input(prompt).strip()
        except EOFError as error:
            raise RuntimeError(
                "当前环境不能进行交互，请先在JSON中 "
                "修改 output.directory"
            ) from error

        if choice in choices:
            return choice

        print(f"请输入: {', '.join(sorted(choices))}")


def prompt_new_output_directory(current_directory: Path) -> Path:
    """询问并返回新的输出目录"""

    suggestion = next_output_directory(current_directory)

    while True:
        try:
            value = input(
                "请输入新的输出目录\n"
                f"直接回车使用: {suggestion}\n> "
            ).strip()
        except EOFError as error:
            raise RuntimeError(
                "当前环境不能进行交互，请先在JSON中 "
                "修改 output.directory"
            ) from error

        candidate = suggestion if not value else Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_DIR / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            return candidate
        if candidate.is_dir() and not any(candidate.iterdir()):
            return candidate

        print(f"目录已被占用，请重新输入: {candidate}")


def resolve_workflow(input_card: InputCard) -> str:
    """确定重新反演、直接后处理或取消"""

    if output_directory_available(input_card):
        return "run"

    directory = Path(input_card.output.directory)
    print(f"\n输出目录已包含文件: {directory}")
    print("[1] 使用新目录重新运行贝叶斯反演并后处理")
    print("[2] 跳过反演，直接后处理已有结果")
    print("[3] 取消")
    choice = prompt_choice("请选择: ",{"1","2","3"})

    if choice == "2":
        return "postprocess"
    if choice == "3":
        return "cancel"

    new_directory = prompt_new_output_directory(directory)
    input_card.output.directory = str(new_directory)
    return "run"


def print_paths(paths: Any,prefix: str = "") -> None:
    """递归输出结果文件路径"""

    if isinstance(paths,dict):
        for name,value in paths.items():
            child_prefix = (
                f"{prefix}.{name}"
                if prefix
                else str(name)
            )
            print_paths(value,child_prefix)
        return

    print(f"{prefix}: {paths}")


def main() -> None:
    """读取命令行参数并执行贝叶斯反演"""

    parser = argparse.ArgumentParser(
        prog="bayesian-inversion",
        description="Run a Bayesian parameter inversion.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Bayesian-inversion JSON configuration.",
    )
    args = parser.parse_args()

    input_card = load_input_card(args.config)
    workflow = resolve_workflow(input_card)
    if workflow == "cancel":
        print("已取消。")
        return

    if workflow == "run":
        print("\n开始贝叶斯参数反演。")
        print(f"随机种子: {input_card.random_seed}")
        output_paths = run_inversion(input_card)
        print("\n贝叶斯参数反演完成。")
        print_paths(output_paths,"inversion")

    print("\n开始贝叶斯结果后处理。")
    figure_paths = run_postprocess(input_card)
    print("贝叶斯结果后处理完成。")
    print_paths(figure_paths,"postprocess")


if __name__ == "__main__":
    main()
