"""贝叶斯反演结果后处理"""

from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

def marginal_pdf(samples:np.ndarray,parameter_names:list[str],bins:int,output_dir:Path) -> None:
    """绘制各参数的一维边缘后验PDF"""

    for name in parameter_names:
        display_name = name.replace("_"," ")
        fig,ax = plt.subplots(figsize=(8,6))
        ax.hist(samples[name],bins=bins,density=True,color="steelblue",edgecolor="black",alpha=0.8)
        ax.set_xlabel(display_name)
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Marginal Posterior PDF of {display_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"marginal_{name}.png",dpi=300)
        plt.close(fig)

def joint_distribution(samples:np.ndarray,parameter_names:list[str],output_dir:Path) -> None:
    """绘制参数间联合分布散点图"""

    for name_i,name_j in combinations(parameter_names,2):
        display_name_i = name_i.replace("_"," ")
        display_name_j = name_j.replace("_"," ")
        fig,ax = plt.subplots(figsize=(8,6))
        ax.scatter(samples[name_i],samples[name_j],s=8,color="steelblue",alpha=0.35,edgecolors="none")
        ax.set_xlabel(display_name_i)
        ax.set_ylabel(display_name_j)
        ax.set_title(f"Joint Posterior Distribution of {display_name_i} and {display_name_j}")
        fig.tight_layout()
        fig.savefig(output_dir / f"joint_{name_i}_{name_j}.png",dpi=300)
        plt.close(fig)

def postprocess(samples_path:str | Path,bins:int) -> None:
    """生成后验分布图"""

    samples_path = Path(samples_path)
    if not samples_path.is_absolute():
        samples_path = PROJECT_DIR / samples_path

    samples = np.genfromtxt(samples_path,delimiter=",",names=True)
    parameter_names = [name for name in samples.dtype.names if name != "log_probability"]
    output_dir = samples_path.parent / "postprocess"
    output_dir.mkdir(parents=True,exist_ok=True)

    marginal_pdf(samples,parameter_names,bins,output_dir)
    joint_distribution(samples,parameter_names,output_dir)

if __name__ == "__main__":
    postprocess_input = {
        "samples_path":"results/triso_bayesian_learning/samples.csv",
        "bins":50,
    }
    postprocess(**postprocess_input)
