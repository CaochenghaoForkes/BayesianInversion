"""后处理"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm,Normalize,TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from config import JsonReader

class Plotter:
    """底层绘图"""

    @dataclass(frozen=True)
    class PlotStyle:
        """绘图风格"""

        font_family: str = "Times New Roman"

        font_size: int = 16
        title_size: int = 16
        label_size: int = 16
        tick_size: int = 14
        legend_size: int = 16

        line_width: float = 2.0
        axes_width: float = 1.0

        dpi: int = 300
        save_pdf: bool = True
        default_figsize: tuple[float,float] = (8.0,6.0)

        colors: tuple[str,...] = (
            "#4C72B0",
            "#DD8452",
            "#55A868",
            "#C44E52",
            "#8172B3",
            "#937860",
            "#DA8BC3",
            "#8C8C8C",
            "#CCB974",
            "#64B5CD",
        )

    class BasePlotter:
        """绘图器公共基类"""

        def __init__(
            self,
            figure_directory: str | Path,
            style: Plotter.PlotStyle | None = None,
            parameter_colors: dict[str,str] | None = None,
        ):
            self.figure_directory = Path(figure_directory).expanduser().resolve()
            self.figure_directory.mkdir(parents=True,exist_ok=True)

            self.style = style or Plotter.PlotStyle()
            self.parameter_colors = parameter_colors if parameter_colors is not None else {}

        def _style_context(self):
            """创建当前绘图器的Matplotlib风格上下文"""

            style = self.style
            rc_parameters = {
                "font.family": "serif",
                "font.serif": [style.font_family],
                "font.size": style.font_size,
                "mathtext.fontset": "stix",
                "axes.titlesize": style.title_size,
                "axes.labelsize": style.label_size,
                "axes.linewidth": style.axes_width,
                "axes.prop_cycle": matplotlib.cycler(color=style.colors),
                "xtick.labelsize": style.tick_size,
                "ytick.labelsize": style.tick_size,
                "xtick.major.width": style.axes_width,
                "ytick.major.width": style.axes_width,
                "legend.fontsize": style.legend_size,
                "legend.frameon": False,
                "lines.linewidth": style.line_width,
                "figure.titlesize": style.title_size,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": style.dpi,
                "pdf.fonttype": 42,
            }

            return matplotlib.rc_context(rc_parameters)

        def _create_figure(
            self,
            nrows: int = 1,
            ncols: int = 1,
            figsize: tuple[float,float] | None = None,
            **subplot_options: Any,
        ):
            """创建画布和坐标轴"""

            subplot_options.setdefault("constrained_layout",True)

            return plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize or self.style.default_figsize,
                **subplot_options,
            )

        def _apply_axis_style(self,axis: Axes) -> None:
            """统一坐标轴的基础风格"""

            axis.title.set_fontsize(self.style.title_size)
            axis.xaxis.label.set_fontsize(self.style.label_size)
            axis.yaxis.label.set_fontsize(self.style.label_size)
            axis.tick_params(
                axis="both",
                labelsize=self.style.tick_size,
                width=self.style.axes_width,
            )

            for spine in axis.spines.values():
                spine.set_linewidth(self.style.axes_width)

        def _register_parameter_colors(
            self,
            parameter_names: list[str] | tuple[str,...],
        ) -> dict[str,str]:
            """为参数登记固定颜色"""

            for parameter_name in parameter_names:
                if parameter_name in self.parameter_colors:
                    continue

                color_index = len(self.parameter_colors) % len(self.style.colors)
                self.parameter_colors[parameter_name] = self.style.colors[color_index]

            return {
                name: self.parameter_colors[name]
                for name in parameter_names
            }

        @staticmethod
        def _display_name(name: str) -> str:
            """将配置名称转换为图中名称"""

            return name.replace("_"," ")

        @staticmethod
        def _safe_filename(name: str) -> str:
            """将文件名转换为安全形式"""

            filename = re.sub(r"[^A-Za-z0-9_.-]+","_",name).strip("._")
            return filename or "figure"

        def _figure_path(self,filename: str,file_format: str) -> Path:
            """获取图片的完整路径"""

            filename_stem = Path(filename).stem
            safe_stem = self._safe_filename(filename_stem)
            extension = file_format.lower().lstrip(".")

            return self.figure_directory / f"{safe_stem}.{extension}"

        def _save_figure(self,figure: Figure,filename: str) -> dict[str,Path]:
            """保存PNG和可选PDF并关闭图片"""

            paths = {
                "png": self._figure_path(filename,"png"),
            }
            if self.style.save_pdf:
                paths["pdf"] = self._figure_path(filename,"pdf")

            try:
                figure.savefig(
                    paths["png"],
                    format="png",
                    dpi=self.style.dpi,
                    facecolor="white",
                )
                if self.style.save_pdf:
                    figure.savefig(
                        paths["pdf"],
                        format="pdf",
                        facecolor="white",
                    )
            finally:
                plt.close(figure)

            return paths

        @staticmethod
        def _figure_image(figure: Figure) -> Image.Image:
            """将Matplotlib画布转换为内存图像"""

            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba()).copy()

            return Image.fromarray(rgba,"RGBA").convert("RGB")

        def _save_animation(
            self,
            frames: list[Image.Image],
            filename: str,
            frame_duration: int = 350,
        ) -> Path:
            """将内存帧保存为GIF"""

            if not frames:
                raise ValueError("GIF至少需要一帧图像")
            if frame_duration <= 0:
                raise ValueError("GIF帧持续时间必须为正整数")

            path = self._figure_path(filename,"gif")
            try:
                frames[0].save(
                    path,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=frame_duration,
                    loop=0,
                    disposal=2,
                )
            finally:
                for frame in frames:
                    frame.close()

            return path

    class HeatmapPlotter(BasePlotter):
        """二维矩阵热力图"""

        @staticmethod
        def _color_norm(
            values: np.ndarray,
            center: float | None,
            vmin: float | None,
            vmax: float | None,
            color_scale: str,
        ) -> Normalize:
            """根据有限数值构造色标归一化"""

            finite_values = values[np.isfinite(values)]

            if color_scale not in {"linear","log"}:
                raise ValueError("热力图颜色映射只支持linear或log")

            if color_scale == "log":
                if center is not None:
                    raise ValueError("log颜色映射不支持center")
                if finite_values.size and np.any(finite_values <= 0.0):
                    raise ValueError("log颜色映射只支持正数")

                data_min = float(np.min(finite_values)) if finite_values.size else 1.0
                data_max = float(np.max(finite_values)) if finite_values.size else 10.0
                lower = data_min if vmin is None else vmin
                upper = data_max if vmax is None else vmax

                if lower <= 0.0 or upper <= 0.0:
                    raise ValueError("log颜色映射的vmin和vmax必须为正数")
                if lower > upper:
                    raise ValueError("热力图的vmin不能大于vmax")
                if lower == upper:
                    lower /= 10.0
                    upper *= 10.0

                return LogNorm(vmin=lower,vmax=upper)

            if finite_values.size:
                data_min = float(np.min(finite_values))
                data_max = float(np.max(finite_values))
            else:
                data_min = -1.0 if center is not None else 0.0
                data_max = 1.0

            lower = data_min if vmin is None else vmin
            upper = data_max if vmax is None else vmax

            if center is None:
                if lower == upper:
                    padding = abs(lower)*0.05 or 1.0
                    lower -= padding
                    upper += padding

                return Normalize(vmin=lower,vmax=upper)

            radius = max(
                abs(lower-center),
                abs(upper-center),
            ) or 1.0

            return TwoSlopeNorm(
                vmin=center-radius,
                vcenter=center,
                vmax=center+radius,
            )

        @staticmethod
        def _add_invalid_hatching(
            axis: Axes,
            invalid_cells: np.ndarray,
            hatch: str | None,
        ) -> None:
            """为无效数值单元格添加斜线"""

            if hatch is None:
                return

            invalid_rows,invalid_columns = np.nonzero(invalid_cells)
            for row,column in zip(invalid_rows,invalid_columns,strict=True):
                rectangle = Rectangle(
                    (column-0.5,row-0.5),
                    1.0,
                    1.0,
                    facecolor="#EEEEEE",
                    edgecolor="#A0A0A0",
                    linewidth=0.5,
                    hatch=hatch,
                )
                axis.add_patch(rectangle)

        @staticmethod
        def _annotation_color(
            value: float,
            colormap,
            norm: Normalize,
        ) -> str:
            """根据背景颜色选择标注文字颜色"""

            red,green,blue,_ = colormap(norm(value))
            luminance = 0.299*red + 0.587*green + 0.114*blue

            return "black" if luminance > 0.55 else "white"

        def plot(
            self,
            matrix: np.ndarray,
            row_labels: list[str],
            column_labels: list[str],
            filename: str,
            *,
            title: str = "",
            colorbar_label: str = "",
            cmap: str = "coolwarm",
            color_scale: str = "linear",
            center: float | None = None,
            vmin: float | None = None,
            vmax: float | None = None,
            mask: np.ndarray | None = None,
            invalid_hatch: str | None = "///",
            annotate: bool = True,
            annotation_format: str = ".3e",
            aspect: str = "equal",
            x_rotation: float = 30.0,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,Path]:
            """绘制并保存热力图"""

            values = np.asarray(matrix,dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("热力图数据必须是二维矩阵")

            row_count,column_count = values.shape
            if len(row_labels) != row_count:
                raise ValueError("行标签数量与热力图矩阵行数不一致")
            if len(column_labels) != column_count:
                raise ValueError("列标签数量与热力图矩阵列数不一致")

            masked_values = values.copy()
            if mask is not None:
                mask = np.asarray(mask,dtype=bool)
                if mask.shape != values.shape:
                    raise ValueError("热力图遮罩与数据矩阵形状不一致")
                masked_values[mask] = np.nan

            norm = self._color_norm(
                masked_values,center,vmin,vmax,color_scale
            )
            colormap = matplotlib.colormaps.get_cmap(cmap).copy()
            colormap.set_bad("#EEEEEE")
            invalid_cells = ~np.isfinite(masked_values)

            with self._style_context():
                figure,axis = self._create_figure(figsize=figsize)
                image = axis.imshow(
                    masked_values,
                    cmap=colormap,
                    norm=norm,
                    aspect=aspect,
                )
                self._add_invalid_hatching(
                    axis,invalid_cells,invalid_hatch
                )

                display_rows = [self._display_name(name) for name in row_labels]
                display_columns = [self._display_name(name) for name in column_labels]
                axis.set_xticks(
                    np.arange(column_count),
                    labels=display_columns,
                    rotation=x_rotation,
                    ha="right",
                )
                axis.set_yticks(
                    np.arange(row_count),
                    labels=display_rows,
                )
                axis.set_title(title)

                if annotate:
                    for row in range(row_count):
                        for column in range(column_count):
                            value = masked_values[row,column]
                            if not np.isfinite(value):
                                continue

                            axis.text(
                                column,
                                row,
                                format(value,annotation_format),
                                ha="center",
                                va="center",
                                color=self._annotation_color(value,colormap,norm),
                                fontsize=self.style.tick_size,
                            )

                colorbar = figure.colorbar(image,ax=axis,shrink=0.85)
                colorbar.set_label(
                    colorbar_label,
                    fontsize=self.style.label_size,
                )
                colorbar.ax.tick_params(
                    labelsize=self.style.tick_size,
                    width=self.style.axes_width,
                )

                self._apply_axis_style(axis)

                return self._save_figure(figure,filename)

        def animate(
            self,
            matrices: np.ndarray,
            row_labels: list[str],
            column_labels: list[str],
            frame_labels: list[str],
            filename: str,
            *,
            title: str = "",
            colorbar_label: str = "",
            cmap: str = "coolwarm",
            center: float | None = 0.0,
            vmin: float | None = None,
            vmax: float | None = None,
            annotate: bool = True,
            annotation_format: str = ".3e",
            invalid_hatch: str | None = "///",
            x_rotation: float = 30.0,
            frame_duration: int = 350,
            figsize: tuple[float,float] | None = None,
        ) -> Path:
            """绘制共用固定色标的热力图GIF"""

            values = np.asarray(matrices,dtype=np.float64)
            if values.ndim != 3:
                raise ValueError("热力图GIF数据必须为三维数组")

            frame_count,row_count,column_count = values.shape
            if len(frame_labels) != frame_count:
                raise ValueError("GIF帧标签数量与数据帧数不一致")
            if len(row_labels) != row_count:
                raise ValueError("行标签数量与热力图矩阵不一致")
            if len(column_labels) != column_count:
                raise ValueError("列标签数量与热力图矩阵不一致")

            norm = self._color_norm(values,center,vmin,vmax,"linear")
            colormap = matplotlib.colormaps.get_cmap(cmap).copy()
            colormap.set_bad("#EEEEEE")
            display_rows = [self._display_name(name) for name in row_labels]
            display_columns = [self._display_name(name) for name in column_labels]
            frames = []

            with self._style_context():
                for frame_index,frame_label in enumerate(frame_labels):
                    matrix = values[frame_index]
                    figure,axis = self._create_figure(figsize=figsize)
                    image = axis.imshow(
                        matrix,
                        cmap=colormap,
                        norm=norm,
                        aspect="equal",
                    )
                    self._add_invalid_hatching(
                        axis,~np.isfinite(matrix),invalid_hatch
                    )

                    axis.set_xticks(
                        np.arange(column_count),
                        labels=display_columns,
                        rotation=x_rotation,
                        ha="right",
                    )
                    axis.set_yticks(
                        np.arange(row_count),
                        labels=display_rows,
                    )
                    axis.set_title(
                        f"{title}\n{frame_label}" if title else frame_label
                    )

                    if annotate:
                        for row in range(row_count):
                            for column in range(column_count):
                                value = matrix[row,column]
                                if not np.isfinite(value):
                                    continue

                                axis.text(
                                    column,
                                    row,
                                    format(value,annotation_format),
                                    ha="center",
                                    va="center",
                                    color=self._annotation_color(
                                        value,colormap,norm
                                    ),
                                    fontsize=self.style.tick_size,
                                )

                    colorbar = figure.colorbar(image,ax=axis,shrink=0.85)
                    colorbar.set_label(
                        colorbar_label,
                        fontsize=self.style.label_size,
                    )
                    colorbar.ax.tick_params(
                        labelsize=self.style.tick_size,
                        width=self.style.axes_width,
                    )
                    self._apply_axis_style(axis)

                    frames.append(self._figure_image(figure))
                    plt.close(figure)

            return self._save_animation(
                frames,filename,frame_duration
            )

    class CurvePlotter(BasePlotter):
        """多参数曲线图"""

        LINE_STYLES = (
            "-",
            "--",
            "-.",
            ":",
            (0,(5,1)),
            (0,(3,1,1,1)),
            (0,(5,2,1,2)),
            (0,(1,1)),
        )

        def __init__(
            self,
            figure_directory: str | Path,
            style: Plotter.PlotStyle | None = None,
            parameter_colors: dict[str,str] | None = None,
            parameter_line_styles: dict[str,Any] | None = None,
        ):
            super().__init__(
                figure_directory,
                style,
                parameter_colors,
            )

            self.parameter_line_styles = (
                parameter_line_styles
                if parameter_line_styles is not None
                else {}
            )

        def _curve_line_styles(
            self,
            parameter_names: list[str],
            colors: dict[str,str],
        ) -> dict[str,Any]:
            """同一种颜色再次出现时才启用下一种线型"""

            color_counts = {}
            line_styles = {}
            for parameter_name in parameter_names:
                color = colors[parameter_name]
                style_index = color_counts.get(color,0)
                color_counts[color] = style_index+1
                line_styles[parameter_name] = self.LINE_STYLES[
                    style_index % len(self.LINE_STYLES)
                ]

            line_styles.update({
                name:style
                for name,style in self.parameter_line_styles.items()
                if name in line_styles
            })

            return line_styles

        def plot(
            self,
            x: np.ndarray,
            series: dict[str,np.ndarray],
            filename: str,
            *,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            lower_bounds: dict[str,np.ndarray] | None = None,
            upper_bounds: dict[str,np.ndarray] | None = None,
            colors: dict[str,str] | None = None,
            line_styles: dict[str,Any] | None = None,
            x_scale: str = "linear",
            y_scale: str = "linear",
            horizontal_lines: dict[str,float] | None = None,
            vertical_lines: dict[str,float] | None = None,
            legend_location: str = "best",
            legend_columns: int | None = None,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,Path]:
            """绘制并保存多参数曲线"""

            x_values = np.asarray(x,dtype=np.float64)
            if x_values.ndim != 1:
                raise ValueError("曲线图的x必须是一维数组")
            if not series:
                raise ValueError("曲线图至少需要一组数据")

            series_values = {}
            for parameter_name,values in series.items():
                values = np.asarray(values,dtype=np.float64)
                if values.shape != x_values.shape:
                    raise ValueError(
                        f"参数 {parameter_name} 的曲线长度与x不一致"
                    )
                series_values[parameter_name] = values

            has_lower_bounds = lower_bounds is not None
            has_upper_bounds = upper_bounds is not None
            if has_lower_bounds != has_upper_bounds:
                raise ValueError("曲线置信区间必须同时提供下界和上界")

            lower_values = {}
            upper_values = {}
            if has_lower_bounds:
                for parameter_name in series_values:
                    if parameter_name not in lower_bounds or parameter_name not in upper_bounds:
                        raise ValueError(
                            f"参数 {parameter_name} 缺少完整的置信区间"
                        )

                    lower = np.asarray(lower_bounds[parameter_name],dtype=np.float64)
                    upper = np.asarray(upper_bounds[parameter_name],dtype=np.float64)
                    if lower.shape != x_values.shape or upper.shape != x_values.shape:
                        raise ValueError(
                            f"参数 {parameter_name} 的置信区间长度与x不一致"
                        )

                    finite = np.isfinite(lower) & np.isfinite(upper)
                    if np.any(lower[finite] > upper[finite]):
                        raise ValueError(
                            f"参数 {parameter_name} 的置信区间下界大于上界"
                        )

                    lower_values[parameter_name] = lower
                    upper_values[parameter_name] = upper

            allowed_scales = {"linear","log","symlog"}
            if x_scale not in allowed_scales or y_scale not in allowed_scales:
                raise ValueError("曲线坐标尺度只支持linear、log或symlog")

            horizontal_lines = horizontal_lines or {}
            vertical_lines = vertical_lines or {}
            for line_name,line_value in horizontal_lines.items():
                if not np.isfinite(line_value):
                    raise ValueError(f"横向参考线 {line_name} 的位置必须为有限值")
                if y_scale == "log" and line_value <= 0.0:
                    raise ValueError(f"log纵坐标不能显示非正参考线 {line_name}")

            for line_name,line_value in vertical_lines.items():
                if not np.isfinite(line_value):
                    raise ValueError(f"纵向参考线 {line_name} 的位置必须为有限值")
                if x_scale == "log" and line_value <= 0.0:
                    raise ValueError(f"log横坐标不能显示非正参考线 {line_name}")

            parameter_names = list(series_values)
            curve_colors = self._register_parameter_colors(parameter_names)
            curve_line_styles = self._curve_line_styles(
                parameter_names,
                curve_colors,
            )

            if colors is not None:
                unknown_names = set(colors) - set(parameter_names)
                if unknown_names:
                    raise ValueError(
                        f"颜色配置中存在未知参数: {', '.join(sorted(unknown_names))}"
                    )
                curve_colors.update(colors)

            if line_styles is not None:
                unknown_names = set(line_styles) - set(parameter_names)
                if unknown_names:
                    raise ValueError(
                        f"线型配置中存在未知参数: {', '.join(sorted(unknown_names))}"
                    )
                curve_line_styles.update(line_styles)

            legend_item_count = (
                len(parameter_names)
                + len(horizontal_lines)
                + len(vertical_lines)
            )
            if legend_columns is None:
                legend_columns = 2 if legend_item_count > 4 else 1

            with self._style_context():
                figure,axis = self._create_figure(figsize=figsize)

                for parameter_name,values in series_values.items():
                    color = curve_colors[parameter_name]

                    if has_lower_bounds:
                        axis.fill_between(
                            x_values,
                            lower_values[parameter_name],
                            upper_values[parameter_name],
                            color=color,
                            alpha=0.18,
                            linewidth=0.0,
                        )

                    axis.plot(
                        x_values,
                        values,
                        color=color,
                        linestyle=curve_line_styles[parameter_name],
                        linewidth=self.style.line_width,
                        label=self._display_name(parameter_name),
                    )

                for line_name,line_value in horizontal_lines.items():
                    axis.axhline(
                        line_value,
                        color="#404040",
                        linewidth=self.style.axes_width,
                        linestyle="--",
                        label=self._display_name(line_name),
                        zorder=0,
                    )

                for line_name,line_value in vertical_lines.items():
                    axis.axvline(
                        line_value,
                        color="#707070",
                        linewidth=self.style.axes_width,
                        linestyle=":",
                        label=self._display_name(line_name),
                        zorder=0,
                    )

                axis.set_xscale(x_scale)
                axis.set_yscale(y_scale)
                axis.set_xlabel(xlabel)
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(
                    True,
                    color="#B0B0B0",
                    alpha=0.30,
                    linewidth=0.7,
                )
                axis.legend(
                    loc=legend_location,
                    ncols=legend_columns,
                )
                axis.margins(x=0.02)

                self._apply_axis_style(axis)

                return self._save_figure(figure,filename)

    class BarPlotter(BasePlotter):
        """多参数柱状图"""

        REFERENCE_LINE_STYLES = (
            "--",
            ":",
            "-.",
        )

        def plot(
            self,
            values: dict[str,float],
            filename: str,
            *,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            lower_bounds: dict[str,float] | None = None,
            upper_bounds: dict[str,float] | None = None,
            colors: dict[str,str] | None = None,
            hatches: dict[str,str] | None = None,
            y_scale: str = "linear",
            horizontal_lines: dict[str,float] | None = None,
            annotate: bool = True,
            annotation_format: str = ".3e",
            bar_width: float = 0.70,
            edge_width: float = 1.2,
            x_rotation: float = 25.0,
            y_limits: tuple[float,float] | None = None,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,Path]:
            """绘制并保存多参数柱状图"""

            if not values:
                raise ValueError("柱状图至少需要一个参数")
            if bar_width <= 0.0 or bar_width > 1.0:
                raise ValueError("柱宽度必须在(0,1]范围内")
            if not np.isfinite(edge_width) or edge_width < 0.0:
                raise ValueError("柱边缘线宽必须为非负有限值")

            if y_limits is not None:
                y_lower,y_upper = y_limits
                if not np.isfinite(y_lower) or not np.isfinite(y_upper):
                    raise ValueError("柱状图纵坐标范围必须为有限值")
                if y_lower >= y_upper:
                    raise ValueError("柱状图纵坐标下限必须小于上限")

            parameter_names = list(values)
            bar_values = np.asarray(
                [values[name] for name in parameter_names],
                dtype=np.float64,
            )
            finite_values = np.isfinite(bar_values)

            allowed_scales = {"linear","log","symlog"}
            if y_scale not in allowed_scales:
                raise ValueError("柱状图纵坐标只支持linear、log或symlog")
            if y_scale == "log" and np.any(bar_values[finite_values] <= 0.0):
                raise ValueError("log柱状图只支持正数")
            if y_scale == "log" and y_limits is not None and y_limits[0] <= 0.0:
                raise ValueError("log柱状图纵坐标下限必须为正数")

            has_lower_bounds = lower_bounds is not None
            has_upper_bounds = upper_bounds is not None
            if has_lower_bounds != has_upper_bounds:
                raise ValueError("柱状图置信区间必须同时提供下界和上界")

            lower_values = np.full(len(parameter_names),np.nan,dtype=np.float64)
            upper_values = np.full(len(parameter_names),np.nan,dtype=np.float64)
            if has_lower_bounds:
                for index,parameter_name in enumerate(parameter_names):
                    if parameter_name not in lower_bounds or parameter_name not in upper_bounds:
                        raise ValueError(
                            f"参数 {parameter_name} 缺少完整的置信区间"
                        )

                    lower_values[index] = lower_bounds[parameter_name]
                    upper_values[index] = upper_bounds[parameter_name]
                    lower = lower_values[index]
                    upper = upper_values[index]
                    if np.isfinite(lower) and np.isfinite(upper) and lower > upper:
                        raise ValueError(
                            f"参数 {parameter_name} 的置信区间下界大于上界"
                        )

            bar_colors = self._register_parameter_colors(parameter_names)
            if colors is not None:
                unknown_names = set(colors) - set(parameter_names)
                if unknown_names:
                    raise ValueError(
                        f"颜色配置中存在未知参数: {', '.join(sorted(unknown_names))}"
                    )
                bar_colors.update(colors)

            hatches = hatches or {}
            unknown_names = set(hatches) - set(parameter_names)
            if unknown_names:
                raise ValueError(
                    f"阴影配置中存在未知参数: {', '.join(sorted(unknown_names))}"
                )

            horizontal_lines = horizontal_lines or {}
            for line_name,line_value in horizontal_lines.items():
                if not np.isfinite(line_value):
                    raise ValueError(f"横向参考线 {line_name} 的位置必须为有限值")
                if y_scale == "log" and line_value <= 0.0:
                    raise ValueError(f"log纵坐标不能显示非正参考线 {line_name}")

            x_positions = np.arange(len(parameter_names))
            plot_values = np.where(finite_values,bar_values,0.0)
            display_labels = [self._display_name(name) for name in parameter_names]
            color_values = [bar_colors[name] for name in parameter_names]

            lower_limit_values = bar_values[finite_values]
            if has_lower_bounds:
                finite_lower_bounds = lower_values[np.isfinite(lower_values)]
                lower_limit_values = np.concatenate([
                    lower_limit_values,
                    finite_lower_bounds,
                ])
            lower_limit = (
                float(np.min(lower_limit_values))
                if lower_limit_values.size
                else None
            )

            with self._style_context():
                figure,axis = self._create_figure(figsize=figsize)
                bars = axis.bar(
                    x_positions,
                    plot_values,
                    width=bar_width,
                    color=color_values,
                    edgecolor="black",
                    linewidth=edge_width,
                )

                for parameter_name,hatch in hatches.items():
                    bar_index = parameter_names.index(parameter_name)
                    bars[bar_index].set_hatch(hatch)

                for index,is_finite in enumerate(finite_values):
                    if is_finite:
                        continue
                    bars[index].set_facecolor("#EEEEEE")
                    bars[index].set_edgecolor("#A0A0A0")
                    bars[index].set_hatch("///")

                if has_lower_bounds:
                    cap_half_width = bar_width*0.15
                    for index,(lower,upper) in enumerate(zip(
                        lower_values,upper_values,strict=True
                    )):
                        if not np.isfinite(lower) or not np.isfinite(upper):
                            continue

                        axis.vlines(
                            index,lower,upper,
                            color="black",
                            linewidth=self.style.axes_width,
                            zorder=3,
                        )
                        axis.hlines(
                            (lower,upper),
                            index-cap_half_width,
                            index+cap_half_width,
                            color="black",
                            linewidth=self.style.axes_width,
                            zorder=3,
                        )

                if annotate:
                    for index,(value,is_finite) in enumerate(zip(
                        bar_values,finite_values,strict=True
                    )):
                        if not is_finite:
                            label = "NaN"
                            label_position = 0.0
                            offset = 4.0
                            vertical_alignment = "bottom"
                        elif value >= 0.0:
                            label = format(value,annotation_format)
                            label_position = value
                            if np.isfinite(upper_values[index]):
                                label_position = max(label_position,upper_values[index])
                            offset = 4.0
                            vertical_alignment = "bottom"
                        else:
                            label = format(value,annotation_format)
                            label_position = value
                            if np.isfinite(lower_values[index]):
                                label_position = min(label_position,lower_values[index])
                            offset = -4.0
                            vertical_alignment = "top"

                        axis.annotate(
                            label,
                            xy=(index,label_position),
                            xytext=(0.0,offset),
                            textcoords="offset points",
                            ha="center",
                            va=vertical_alignment,
                            fontsize=self.style.tick_size,
                            annotation_clip=False,
                        )

                for line_index,(line_name,line_value) in enumerate(horizontal_lines.items()):
                    line_style = self.REFERENCE_LINE_STYLES[
                        line_index % len(self.REFERENCE_LINE_STYLES)
                    ]
                    axis.axhline(
                        line_value,
                        color="#404040",
                        linewidth=self.style.axes_width,
                        linestyle=line_style,
                        label=self._display_name(line_name),
                        zorder=0,
                    )

                axis.set_yscale(y_scale)
                axis.set_xticks(
                    x_positions,
                    labels=display_labels,
                    rotation=x_rotation,
                    ha="right" if x_rotation else "center",
                )
                axis.set_xlabel(xlabel)
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(
                    True,
                    axis="y",
                    color="#B0B0B0",
                    alpha=0.30,
                    linewidth=0.7,
                )
                if horizontal_lines:
                    axis.legend()
                axis.margins(y=0.15)
                if y_limits is not None:
                    axis.set_ylim(*y_limits)
                elif lower_limit is not None:
                    _,upper_limit = axis.get_ylim()
                    axis.set_ylim(lower_limit,upper_limit)

                self._apply_axis_style(axis)

                return self._save_figure(figure,filename)

    class DistributionPlotter(BasePlotter):
        """多参数统计分布图"""

        def _draw_distribution(
            self,
            axis: Axes,
            parameter_name: str,
            samples: np.ndarray,
            color: str,
            *,
            bins: int | str,
            density: bool,
            x_scale: str,
            xlabel: str,
            ylabel: str,
            point_estimate: float | None,
            secondary_estimate: float | None,
            confidence_interval: tuple[float,float] | None,
            histogram_alpha: float,
            edge_width: float,
            show_legend: bool,
            interval_label: str,
            point_estimate_label: str,
            secondary_estimate_label: str,
        ) -> bool:
            """在一个坐标轴上绘制单参数分布"""

            if samples.size:
                axis.hist(
                    samples,
                    bins=bins,
                    density=density,
                    color=color,
                    alpha=histogram_alpha,
                    edgecolor="#303030",
                    linewidth=edge_width,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No finite samples",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

            has_legend_item = False
            if confidence_interval is not None:
                lower,upper = confidence_interval
                if np.isfinite(lower) and np.isfinite(upper):
                    axis.axvspan(
                        lower,
                        upper,
                        color=color,
                        alpha=0.14,
                        label=interval_label if show_legend else None,
                        zorder=0,
                    )
                    axis.axvline(
                        lower,
                        color=color,
                        linestyle=":",
                        linewidth=self.style.axes_width,
                    )
                    axis.axvline(
                        upper,
                        color=color,
                        linestyle=":",
                        linewidth=self.style.axes_width,
                    )
                    has_legend_item = True

            if point_estimate is not None and np.isfinite(point_estimate):
                axis.axvline(
                    point_estimate,
                    color="black",
                    linestyle="--",
                    linewidth=self.style.line_width,
                    label=point_estimate_label if show_legend else None,
                )
                has_legend_item = True

            if secondary_estimate is not None and np.isfinite(secondary_estimate):
                axis.axvline(
                    secondary_estimate,
                    color="#505050",
                    linestyle="-",
                    linewidth=self.style.line_width,
                    label=secondary_estimate_label if show_legend else None,
                )
                has_legend_item = True

            axis.set_xscale(x_scale)
            axis.set_title(self._display_name(parameter_name))
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if density else "Count")
            axis.grid(
                True,
                axis="y",
                color="#B0B0B0",
                alpha=0.25,
                linewidth=0.7,
            )
            axis.margins(x=0.06)
            self._apply_axis_style(axis)

            if show_legend and has_legend_item:
                axis.legend()

            return has_legend_item

        def plot(
            self,
            distributions: dict[str,np.ndarray],
            filename: str,
            *,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "Density",
            point_estimates: dict[str,float] | None = None,
            secondary_estimates: dict[str,float] | None = None,
            confidence_intervals: dict[str,tuple[float,float]] | None = None,
            colors: dict[str,str] | None = None,
            bins: int | str = "auto",
            density: bool = True,
            x_scale: str = "linear",
            columns: int | None = None,
            share_x: bool = False,
            separate_figures: bool = False,
            histogram_alpha: float = 0.78,
            edge_width: float = 0.40,
            interval_label: str = "Confidence interval",
            point_estimate_label: str = "Point estimate",
            secondary_estimate_label: str = "Secondary estimate",
            single_figure_title: str | None = None,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,dict[str,Path]]:
            """绘制并保存多参数统计分布"""

            if not distributions:
                raise ValueError("分布图至少需要一组数据")
            if isinstance(bins,int) and bins <= 0:
                raise ValueError("直方图分箱数必须为正整数")
            if not 0.0 < histogram_alpha <= 1.0:
                raise ValueError("直方图透明度必须在(0,1]范围内")
            if not np.isfinite(edge_width) or edge_width < 0.0:
                raise ValueError("直方图边缘线宽必须为非负有限值")

            allowed_scales = {"linear","log","symlog"}
            if x_scale not in allowed_scales:
                raise ValueError("分布图横坐标只支持linear、log或symlog")

            parameter_names = list(distributions)
            finite_distributions = {}
            for parameter_name,samples in distributions.items():
                samples = np.asarray(samples,dtype=np.float64)
                if samples.ndim != 1:
                    raise ValueError(
                        f"参数 {parameter_name} 的分布数据必须是一维数组"
                    )

                samples = samples[np.isfinite(samples)]
                if x_scale == "log" and samples.size and np.any(samples <= 0.0):
                    raise ValueError(
                        f"参数 {parameter_name} 的log分布图中存在非正数"
                    )
                finite_distributions[parameter_name] = samples

            distribution_colors = self._register_parameter_colors(parameter_names)
            if colors is not None:
                unknown_names = set(colors) - set(parameter_names)
                if unknown_names:
                    raise ValueError(
                        f"颜色配置中存在未知参数: {', '.join(sorted(unknown_names))}"
                    )
                distribution_colors.update(colors)

            point_estimates = point_estimates or {}
            unknown_names = set(point_estimates) - set(parameter_names)
            if unknown_names:
                raise ValueError(
                    f"点估计中存在未知参数: {', '.join(sorted(unknown_names))}"
                )

            secondary_estimates = secondary_estimates or {}
            unknown_names = set(secondary_estimates) - set(parameter_names)
            if unknown_names:
                raise ValueError(
                    f"次要点估计中存在未知参数: "
                    f"{', '.join(sorted(unknown_names))}"
                )

            confidence_intervals = confidence_intervals or {}
            unknown_names = set(confidence_intervals) - set(parameter_names)
            if unknown_names:
                raise ValueError(
                    f"置信区间中存在未知参数: {', '.join(sorted(unknown_names))}"
                )

            for parameter_name,(lower,upper) in confidence_intervals.items():
                if np.isfinite(lower) and np.isfinite(upper) and lower > upper:
                    raise ValueError(
                        f"参数 {parameter_name} 的置信区间下界大于上界"
                    )
                if x_scale == "log":
                    finite_bounds = [value for value in (lower,upper) if np.isfinite(value)]
                    if any(value <= 0.0 for value in finite_bounds):
                        raise ValueError(
                            f"参数 {parameter_name} 的log置信区间中存在非正数"
                        )

            for parameter_name,estimate in point_estimates.items():
                if x_scale == "log" and np.isfinite(estimate) and estimate <= 0.0:
                    raise ValueError(
                        f"参数 {parameter_name} 的log点估计必须为正数"
                    )

            for parameter_name,estimate in secondary_estimates.items():
                if x_scale == "log" and np.isfinite(estimate) and estimate <= 0.0:
                    raise ValueError(
                        f"参数 {parameter_name} 的log次要点估计必须为正数"
                    )

            if separate_figures:
                output_paths = {}
                for parameter_name in parameter_names:
                    with self._style_context():
                        figure,axis = self._create_figure(figsize=figsize)
                        self._draw_distribution(
                            axis,
                            parameter_name,
                            finite_distributions[parameter_name],
                            distribution_colors[parameter_name],
                            bins=bins,
                            density=density,
                            x_scale=x_scale,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            point_estimate=point_estimates.get(parameter_name),
                            secondary_estimate=secondary_estimates.get(parameter_name),
                            confidence_interval=confidence_intervals.get(parameter_name),
                            histogram_alpha=histogram_alpha,
                            edge_width=edge_width,
                            show_legend=True,
                            interval_label=interval_label,
                            point_estimate_label=point_estimate_label,
                            secondary_estimate_label=secondary_estimate_label,
                        )
                        if title:
                            title_prefix = single_figure_title or title
                            axis.set_title(
                                f"{title_prefix}: "
                                f"{self._display_name(parameter_name)}"
                            )

                        parameter_filename = f"{filename}_{parameter_name}"
                        output_paths[parameter_name] = self._save_figure(
                            figure,parameter_filename
                        )

                return output_paths

            parameter_count = len(parameter_names)
            if columns is None:
                columns = min(2,parameter_count)
            if columns <= 0:
                raise ValueError("分布图子图列数必须为正整数")

            columns = min(columns,parameter_count)
            rows = int(np.ceil(parameter_count/columns))

            with self._style_context():
                figure,axes = self._create_figure(
                    nrows=rows,
                    ncols=columns,
                    figsize=figsize,
                    squeeze=False,
                    sharex=share_x,
                )

                legend_drawn = False
                for plot_index,parameter_name in enumerate(parameter_names):
                    axis = axes.flat[plot_index]
                    has_legend_item = self._draw_distribution(
                        axis,
                        parameter_name,
                        finite_distributions[parameter_name],
                        distribution_colors[parameter_name],
                        bins=bins,
                        density=density,
                        x_scale=x_scale,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        point_estimate=point_estimates.get(parameter_name),
                        secondary_estimate=secondary_estimates.get(parameter_name),
                        confidence_interval=confidence_intervals.get(parameter_name),
                        histogram_alpha=histogram_alpha,
                        edge_width=edge_width,
                        show_legend=not legend_drawn,
                        interval_label=interval_label,
                        point_estimate_label=point_estimate_label,
                        secondary_estimate_label=secondary_estimate_label,
                    )
                    if has_legend_item:
                        legend_drawn = True

                for axis in axes.flat[parameter_count:]:
                    axis.set_visible(False)

                if title:
                    figure.suptitle(title)

                return {
                    "combined": self._save_figure(figure,filename),
                }

        def animate(
            self,
            distributions: dict[str,np.ndarray],
            frame_labels: list[str],
            filename: str,
            *,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "Density",
            point_estimates: dict[str,np.ndarray] | None = None,
            confidence_intervals: (
                dict[str,tuple[np.ndarray,np.ndarray]] | None
            ) = None,
            bins: int | str = 30,
            density: bool = True,
            separate_figures: bool = False,
            histogram_alpha: float = 0.78,
            edge_width: float = 0.40,
            frame_duration: int = 350,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,dict[str,Path]]:
            """绘制共用固定分箱和坐标范围的分布GIF"""

            if not distributions:
                raise ValueError("分布GIF至少需要一组数据")

            parameter_names = list(distributions)
            distribution_values = {}
            frame_count = None

            for parameter_name,samples in distributions.items():
                samples = np.asarray(samples,dtype=np.float64)
                if samples.ndim != 2:
                    raise ValueError(
                        f"参数 {parameter_name} 的GIF分布数据必须为二维"
                    )
                if frame_count is None:
                    frame_count = samples.shape[0]
                elif samples.shape[0] != frame_count:
                    raise ValueError("各参数的GIF帧数不一致")

                distribution_values[parameter_name] = samples

            if len(frame_labels) != frame_count:
                raise ValueError("GIF帧标签数量与分布帧数不一致")

            point_estimates = point_estimates or {}
            confidence_intervals = confidence_intervals or {}
            for parameter_name in parameter_names:
                if parameter_name in point_estimates:
                    estimates = np.asarray(
                        point_estimates[parameter_name],dtype=np.float64
                    )
                    if estimates.shape != (frame_count,):
                        raise ValueError(
                            f"参数 {parameter_name} 的点估计帧数不一致"
                        )
                    point_estimates[parameter_name] = estimates

                if parameter_name in confidence_intervals:
                    lower,upper = confidence_intervals[parameter_name]
                    lower = np.asarray(lower,dtype=np.float64)
                    upper = np.asarray(upper,dtype=np.float64)
                    if lower.shape != (frame_count,) or upper.shape != (frame_count,):
                        raise ValueError(
                            f"参数 {parameter_name} 的置信区间帧数不一致"
                        )
                    confidence_intervals[parameter_name] = (lower,upper)

            finite_groups = [
                values[np.isfinite(values)]
                for values in distribution_values.values()
                if np.any(np.isfinite(values))
            ]
            if finite_groups:
                finite_values = np.concatenate(finite_groups)
                value_min = float(np.min(finite_values))
                value_max = float(np.max(finite_values))
            else:
                finite_values = np.asarray([-1.0,1.0])
                value_min,value_max = -1.0,1.0

            if value_min == value_max:
                padding = abs(value_min)*0.05 or 1.0
                value_min -= padding
                value_max += padding

            bin_edges = np.histogram_bin_edges(
                finite_values,
                bins=bins,
                range=(value_min,value_max),
            )
            y_limits = {}
            for parameter_name,values in distribution_values.items():
                maximum = 0.0
                for samples in values:
                    samples = samples[np.isfinite(samples)]
                    if not samples.size:
                        continue
                    heights,_ = np.histogram(
                        samples,bins=bin_edges,density=density
                    )
                    finite_heights = heights[np.isfinite(heights)]
                    if finite_heights.size:
                        maximum = max(maximum,float(np.max(finite_heights)))

                y_limits[parameter_name] = maximum*1.12 if maximum else 1.0

            colors = self._register_parameter_colors(parameter_names)

            def draw_frame(
                selected_names: list[str],
                frame_index: int,
            ) -> Figure:
                column_count = min(2,len(selected_names))
                row_count = int(np.ceil(len(selected_names)/column_count))
                figure,axes = self._create_figure(
                    nrows=row_count,
                    ncols=column_count,
                    figsize=figsize,
                    squeeze=False,
                )

                legend_drawn = False
                for plot_index,parameter_name in enumerate(selected_names):
                    axis = axes.flat[plot_index]
                    point_estimate = None
                    interval = None
                    if parameter_name in point_estimates:
                        point_estimate = point_estimates[parameter_name][frame_index]
                    if parameter_name in confidence_intervals:
                        lower,upper = confidence_intervals[parameter_name]
                        interval = (lower[frame_index],upper[frame_index])

                    samples = distribution_values[
                        parameter_name
                    ][frame_index]
                    samples = samples[np.isfinite(samples)]

                    has_legend = self._draw_distribution(
                        axis,
                        parameter_name,
                        samples,
                        colors[parameter_name],
                        bins=bin_edges,
                        density=density,
                        x_scale="linear",
                        xlabel=xlabel,
                        ylabel=ylabel,
                        point_estimate=point_estimate,
                        secondary_estimate=None,
                        confidence_interval=interval,
                        histogram_alpha=histogram_alpha,
                        edge_width=edge_width,
                        show_legend=not legend_drawn,
                        interval_label="Confidence interval",
                        point_estimate_label="Point estimate",
                        secondary_estimate_label="Secondary estimate",
                    )
                    axis.set_xlim(value_min,value_max)
                    axis.set_ylim(0.0,y_limits[parameter_name])
                    if has_legend:
                        legend_drawn = True

                for axis in axes.flat[len(selected_names):]:
                    axis.set_visible(False)

                frame_title = frame_labels[frame_index]
                figure.suptitle(
                    f"{title}\n{frame_title}" if title else frame_title
                )

                return figure

            output_paths = {}
            groups = (
                [[name] for name in parameter_names]
                if separate_figures
                else [parameter_names]
            )

            with self._style_context():
                for selected_names in groups:
                    frames = []
                    for frame_index in range(frame_count):
                        figure = draw_frame(selected_names,frame_index)
                        frames.append(self._figure_image(figure))
                        plt.close(figure)

                    if separate_figures:
                        group_name = selected_names[0]
                        group_filename = f"{filename}_{group_name}"
                    else:
                        group_name = "combined"
                        group_filename = filename

                    output_paths[group_name] = {
                        "gif": self._save_animation(
                            frames,group_filename,frame_duration
                        ),
                    }

            return output_paths

    class TracePlotter(BasePlotter):
        """MCMC walker轨迹图"""

        def plot(
            self,
            steps: np.ndarray,
            chains: np.ndarray,
            series_names: list[str],
            filename: str,
            *,
            title: str = "",
            xlabel: str = "MCMC step",
            separate_figures: bool = False,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,dict[str,Path]]:
            """绘制 step×walker×series 轨迹"""

            x = np.asarray(steps,dtype=np.float64)
            values = np.asarray(chains,dtype=np.float64)
            if values.ndim == 2:
                values = values[...,None]
            if values.ndim != 3:
                raise ValueError("轨迹数据必须为 step×walker×series")
            if x.ndim != 1 or values.shape[0] != x.size:
                raise ValueError("轨迹步数与steps长度不一致")
            if values.shape[2] != len(series_names):
                raise ValueError("轨迹序列数量与名称数量不一致")

            if separate_figures:
                return {
                    name:self.plot(
                        x,
                        values[:,:,index:index+1],
                        [name],
                        f"{filename}_{name}",
                        title=title,
                        xlabel=xlabel,
                        separate_figures=False,
                        figsize=figsize,
                    )["combined"]
                    for index,name in enumerate(series_names)
                }

            colors = self._register_parameter_colors(series_names)
            row_count = len(series_names)
            figure_size = figsize or (9.0,max(3.2,2.45*row_count))

            with self._style_context():
                figure,axes = self._create_figure(
                    nrows=row_count,
                    ncols=1,
                    figsize=figure_size,
                    squeeze=False,
                    sharex=True,
                )

                for index,name in enumerate(series_names):
                    axis = axes[index,0]
                    color = colors[name]
                    for walker in range(values.shape[1]):
                        axis.plot(
                            x,
                            values[:,walker,index],
                            color=color,
                            alpha=0.32,
                            linewidth=max(0.65,self.style.line_width*0.45),
                        )

                    median = np.nanmedian(values[:,:,index],axis=1)
                    axis.plot(
                        x,
                        median,
                        color="black",
                        linestyle="--",
                        linewidth=self.style.axes_width,
                        label="Walker median",
                    )
                    axis.set_ylabel(self._display_name(name))
                    axis.grid(
                        True,
                        color="#B0B0B0",
                        alpha=0.25,
                        linewidth=0.7,
                    )
                    axis.margins(x=0.01)
                    self._apply_axis_style(axis)

                axes[0,0].legend(loc="best")
                axes[-1,0].set_xlabel(xlabel)
                if title:
                    figure.suptitle(title)

                return {
                    "combined":self._save_figure(figure,filename)
                }

    class PairScatterPlotter(BasePlotter):
        """后验参数两两散点图"""

        SCATTER_COLOR = "#707070"
        SCATTER_ALPHA = 0.35

        def plot(
            self,
            samples: np.ndarray,
            parameter_names: list[str],
            filename_prefix: str,
            *,
            maximum_points: int = 5000,
            figsize: tuple[float,float] | None = None,
        ) -> dict[str,dict[str,Path]]:
            """为每对参数单独绘图并标注Pearson相关系数"""

            values = np.asarray(samples,dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("参数散点图样本必须是二维数组")
            if values.shape[1] != len(parameter_names):
                raise ValueError("参数数量与散点图样本列数不一致")
            if len(parameter_names) < 2:
                raise ValueError("参数散点图至少需要两个参数")
            if maximum_points <= 0:
                raise ValueError("参数散点图最大点数必须为正整数")

            finite_rows = np.all(np.isfinite(values),axis=1)
            values = values[finite_rows]
            if not values.size:
                raise ValueError("参数散点图没有可用的有限样本")

            plot_values = values
            if len(values) > maximum_points:
                indices = np.linspace(
                    0,len(values)-1,maximum_points,dtype=int
                )
                plot_values = values[indices]

            correlation = np.corrcoef(values,rowvar=False)
            figure_size = figsize or self.style.default_figsize
            output_paths = {}

            with self._style_context():
                for first in range(len(parameter_names)-1):
                    for second in range(first+1,len(parameter_names)):
                        first_name = parameter_names[first]
                        second_name = parameter_names[second]
                        coefficient = correlation[first,second]
                        coefficient_label = (
                            rf"Pearson $\rho={coefficient:.3f}$"
                            if np.isfinite(coefficient)
                            else r"Pearson $\rho=\mathrm{NaN}$"
                        )

                        figure,axis = self._create_figure(
                            figsize=figure_size
                        )
                        axis.scatter(
                            plot_values[:,first],
                            plot_values[:,second],
                            s=10.0,
                            color=self.SCATTER_COLOR,
                            alpha=self.SCATTER_ALPHA,
                            edgecolors="none",
                            rasterized=True,
                        )
                        axis.text(
                            0.04,0.95,
                            coefficient_label,
                            ha="left",
                            va="top",
                            fontsize=self.style.label_size,
                            transform=axis.transAxes,
                            bbox={
                                "facecolor":"white",
                                "edgecolor":"none",
                                "alpha":0.88,
                                "boxstyle":"round,pad=0.30",
                            },
                        )
                        axis.set_xlabel(self._display_name(first_name))
                        axis.set_ylabel(self._display_name(second_name))
                        axis.set_title(
                            f"{self._display_name(first_name)} vs "
                            f"{self._display_name(second_name)}"
                        )
                        axis.ticklabel_format(
                            axis="both",
                            style="sci",
                            scilimits=(-3,4),
                            useMathText=True,
                        )
                        axis.grid(
                            True,
                            color="#B0B0B0",
                            alpha=0.25,
                            linewidth=0.7,
                        )
                        axis.margins(x=0.05,y=0.05)
                        self._apply_axis_style(axis)

                        pair_name = f"{first_name}__{second_name}"
                        output_paths[pair_name] = self._save_figure(
                            figure,
                            f"{filename_prefix}_{first_name}_{second_name}",
                        )

            return output_paths


class Postprocess:
    """贝叶斯反演后处理"""

    @dataclass
    class SamplerOutput:
        """从标准NPZ读取的采样结果"""

        parameter_names: list[str]
        samples: np.ndarray
        log_probability: np.ndarray
        chain: np.ndarray
        log_probability_chain: np.ndarray
        steps: np.ndarray
        acceptance_fraction: np.ndarray
        autocorrelation_time: np.ndarray
        effective_sample_size: np.ndarray
        chain_long_enough: np.ndarray
        converged: bool
        metadata: dict[str,Any]

    class ResultRepository:
        """定位、读取并验证贝叶斯标准输出"""

        def __init__(self,result_directory: str | Path):
            self.result_directory = Path(
                result_directory
            ).expanduser().resolve()
            if not self.result_directory.is_dir():
                raise FileNotFoundError(
                    f"贝叶斯结果目录不存在: {self.result_directory}"
                )

        def result_path(self) -> Path:
            """返回目录中最新的标准NPZ结果"""

            paths = list(
                self.result_directory.glob("sampler_result_*.npz")
            )
            if not paths:
                raise FileNotFoundError(
                    f"目录中没有 sampler_result_*.npz: "
                    f"{self.result_directory}"
                )

            return max(paths,key=lambda path:path.stat().st_mtime)

        @staticmethod
        def _scalar(value: np.ndarray):
            """将NPZ标量数组转换为Python标量"""

            return value.item() if value.ndim == 0 else value.copy()

        def load(self) -> "Postprocess.SamplerOutput":
            """读取并验证标准采样结果"""

            path = self.result_path()
            with np.load(path,allow_pickle=False) as file:
                required = {
                    "parameter_names","samples","log_probability",
                    "chain","log_probability_chain","steps",
                    "acceptance_fraction","autocorrelation_time",
                    "effective_sample_size","chain_long_enough",
                    "converged",
                }
                missing = required-set(file.files)
                if missing:
                    raise ValueError(
                        f"标准采样结果缺少字段: {sorted(missing)}"
                    )

                metadata = {
                    name.removeprefix("metadata__"):
                    self._scalar(file[name])
                    for name in file.files
                    if name.startswith("metadata__")
                }
                result = Postprocess.SamplerOutput(
                    parameter_names=[str(name) for name in file["parameter_names"]],
                    samples=np.asarray(file["samples"],dtype=np.float64),
                    log_probability=np.asarray(file["log_probability"],dtype=np.float64),
                    chain=np.asarray(file["chain"],dtype=np.float64),
                    log_probability_chain=np.asarray(file["log_probability_chain"],dtype=np.float64),
                    steps=np.asarray(file["steps"],dtype=np.int64),
                    acceptance_fraction=np.asarray(file["acceptance_fraction"],dtype=np.float64),
                    autocorrelation_time=np.asarray(file["autocorrelation_time"],dtype=np.float64),
                    effective_sample_size=np.asarray(file["effective_sample_size"],dtype=np.float64),
                    chain_long_enough=np.asarray(file["chain_long_enough"],dtype=bool),
                    converged=bool(self._scalar(file["converged"])),
                    metadata=metadata,
                )

            self._validate(result,path)
            return result

        @staticmethod
        def _validate(
            result: "Postprocess.SamplerOutput",
            path: Path,
        ) -> None:
            """验证后处理依赖的数组形状"""

            if result.chain.ndim != 3:
                raise ValueError(f"采样链必须为三维数组: {path}")

            steps,walkers,dimension = result.chain.shape
            if dimension != len(result.parameter_names):
                raise ValueError("采样链参数维度与参数名数量不一致")
            if result.log_probability_chain.shape != (steps,walkers):
                raise ValueError("对数后验链形状与采样链不一致")
            if result.samples.shape != (steps*walkers,dimension):
                raise ValueError("展平后验样本形状与采样链不一致")
            if result.log_probability.shape != (steps*walkers,):
                raise ValueError("展平对数后验形状与采样链不一致")
            if result.steps.shape != (steps,):
                raise ValueError("保存步数与采样链不一致")
            if result.acceptance_fraction.shape != (walkers,):
                raise ValueError("接受率数量与walker数量不一致")

            diagnostic_shape = (dimension,)
            diagnostics = (
                result.autocorrelation_time,
                result.effective_sample_size,
                result.chain_long_enough,
            )
            if any(values.shape != diagnostic_shape for values in diagnostics):
                raise ValueError("参数诊断数组形状不一致")

    class TracePostprocess:
        """采样轨迹与对数后验轨迹"""

        def __init__(
            self,
            plotter: Plotter.TracePlotter,
            separate_figures: bool,
        ):
            self.plotter = plotter
            self.separate_figures = separate_figures

        def run(
            self,
            result: "Postprocess.SamplerOutput",
        ) -> dict[str,dict[str,Path]]:
            """绘制全部轨迹图"""

            return {
                "parameter_trace":self.plotter.plot(
                    result.steps,
                    result.chain,
                    result.parameter_names,
                    "parameter_trace",
                    title="Posterior parameter traces",
                    separate_figures=self.separate_figures,
                ),
                "log_probability_trace":self.plotter.plot(
                    result.steps,
                    result.log_probability_chain,
                    ["log_probability"],
                    "log_probability_trace",
                    title="Log-posterior trace",
                ),
            }

    class PosteriorDistributionPostprocess:
        """参数边缘后验分布"""

        def __init__(
            self,
            plotter: Plotter.DistributionPlotter,
            separate_figures: bool,
        ):
            self.plotter = plotter
            self.separate_figures = separate_figures

        def run(
            self,
            result: "Postprocess.SamplerOutput",
        ) -> dict[str,dict[str,Path]]:
            """绘制后验分布、95%可信区间和MAP样本"""

            distributions = {
                name:result.samples[:,index]
                for index,name in enumerate(result.parameter_names)
            }
            lower,upper = np.nanquantile(
                result.samples,[0.025,0.975],axis=0
            )
            intervals = {
                name:(lower[index],upper[index])
                for index,name in enumerate(result.parameter_names)
            }
            median_values = np.nanmedian(result.samples,axis=0)
            medians = dict(zip(
                result.parameter_names,median_values,strict=True
            ))

            finite = np.isfinite(result.log_probability)
            point_estimates = {}
            if np.any(finite):
                finite_indices = np.flatnonzero(finite)
                local_index = np.argmax(result.log_probability[finite])
                map_values = result.samples[finite_indices[local_index]]
                point_estimates = dict(zip(
                    result.parameter_names,map_values,strict=True
                ))

            return self.plotter.plot(
                distributions,
                "posterior_distributions",
                title="Marginal posterior distributions",
                ylabel="Density",
                point_estimates=point_estimates,
                secondary_estimates=medians,
                confidence_intervals=intervals,
                bins=30,
                density=True,
                columns=2,
                separate_figures=self.separate_figures,
                interval_label="95% credible interval",
                point_estimate_label="MAP sample",
                secondary_estimate_label="Posterior median",
                single_figure_title="Marginal posterior distribution",
                figsize=(10.0,8.0),
            )

    class PosteriorPairPostprocess:
        """后验参数两两散点图"""

        def __init__(self,plotter: Plotter.PairScatterPlotter):
            self.plotter = plotter

        def run(
            self,
            result: "Postprocess.SamplerOutput",
        ) -> dict[str,dict[str,Path]]:
            """将每一对参数单独绘制为散点图"""

            return self.plotter.plot(
                result.samples,
                result.parameter_names,
                "posterior_pair",
            )

    class PosteriorCorrelationPostprocess:
        """后验参数相关矩阵"""

        def __init__(self,plotter: Plotter.HeatmapPlotter):
            self.plotter = plotter

        def run(
            self,
            result: "Postprocess.SamplerOutput",
        ) -> dict[str,Path]:
            """绘制Pearson相关系数热力图"""

            finite_rows = np.all(np.isfinite(result.samples),axis=1)
            if np.count_nonzero(finite_rows) < 2:
                raise ValueError("后验相关矩阵至少需要两个有限样本")

            correlation = np.corrcoef(
                result.samples[finite_rows],rowvar=False
            )
            return self.plotter.plot(
                correlation,
                result.parameter_names,
                result.parameter_names,
                "posterior_correlation",
                title="Posterior parameter correlation",
                colorbar_label="Pearson correlation",
                cmap="coolwarm",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                annotation_format=".3f",
                x_rotation=30.0,
                figsize=(8.0,6.5),
            )

    class SamplingDiagnosticPostprocess:
        """接受率、自相关时间、ESS和链长充分性"""

        def __init__(self,plotter: Plotter.BarPlotter):
            self.plotter = plotter

        @staticmethod
        def _parameter_values(
            names: list[str],
            values: np.ndarray,
        ) -> dict[str,float]:
            """组装参数名到诊断数值的映射"""

            return {
                name:float(value)
                for name,value in zip(names,values,strict=True)
            }

        @staticmethod
        def _upper_limit(
            values: np.ndarray,
            minimum: float = 1.0,
        ) -> float:
            """为诊断柱状图生成稳定的纵坐标上限"""

            finite = np.asarray(values,dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if not finite.size:
                return minimum

            return max(minimum,1.25*float(np.max(finite)))

        def run(
            self,
            result: "Postprocess.SamplerOutput",
        ) -> dict[str,dict[str,Path]]:
            """绘制全部采样诊断图"""

            acceptance = {
                f"walker_{index}":float(value)
                for index,value in enumerate(result.acceptance_fraction)
            }
            acceptance_colors = {
                name:"#4C72B0" for name in acceptance
            }
            parameter_colors = self.plotter._register_parameter_colors(
                result.parameter_names
            )

            tau = self._parameter_values(
                result.parameter_names,result.autocorrelation_time
            )
            ess = self._parameter_values(
                result.parameter_names,result.effective_sample_size
            )

            production_steps = float(
                result.metadata.get(
                    "production_steps",
                    result.chain.shape[0]
                    * int(result.metadata.get("thin",1)),
                )
            )
            with np.errstate(divide="ignore",invalid="ignore"):
                sufficiency_values = production_steps/(
                    100.0*result.autocorrelation_time
                )
            sufficiency = self._parameter_values(
                result.parameter_names,sufficiency_values
            )

            return {
                "acceptance_fraction":self.plotter.plot(
                    acceptance,
                    "acceptance_fraction",
                    title="Walker acceptance fraction",
                    ylabel="Acceptance fraction",
                    colors=acceptance_colors,
                    horizontal_lines={
                        "mean":float(np.mean(result.acceptance_fraction))
                    },
                    annotation_format=".3f",
                    x_rotation=45.0,
                    y_limits=(0.0,1.0),
                    figsize=(10.0,6.0),
                ),
                "autocorrelation_time":self.plotter.plot(
                    tau,
                    "autocorrelation_time",
                    title="Integrated autocorrelation time",
                    ylabel="MCMC steps",
                    colors=parameter_colors,
                    annotation_format=".2f",
                    x_rotation=25.0,
                    y_limits=(
                        0.0,self._upper_limit(result.autocorrelation_time)
                    ),
                ),
                "effective_sample_size":self.plotter.plot(
                    ess,
                    "effective_sample_size",
                    title="Effective sample size",
                    ylabel="Effective samples",
                    colors=parameter_colors,
                    annotation_format=".1f",
                    x_rotation=25.0,
                    y_limits=(
                        0.0,self._upper_limit(result.effective_sample_size)
                    ),
                ),
                "chain_length_sufficiency":self.plotter.plot(
                    sufficiency,
                    "chain_length_sufficiency",
                    title="Chain-length sufficiency",
                    ylabel=r"$N/(100\tau)$",
                    colors=parameter_colors,
                    horizontal_lines={"criterion":1.0},
                    annotation_format=".3f",
                    x_rotation=25.0,
                    y_limits=(
                        0.0,self._upper_limit(
                            sufficiency_values,minimum=1.15
                        )
                    ),
                ),
            }


class PostProcessor:
    """贝叶斯反演后处理统一调度器"""

    def __init__(
        self,
        result_directory: str | Path,
        figure_directory: str | Path | None = None,
        style: Plotter.PlotStyle | None = None,
        separate_figures: bool = False,
    ):
        self.repository = Postprocess.ResultRepository(result_directory)
        self.figure_directory = (
            Path(figure_directory).expanduser().resolve()
            if figure_directory is not None
            else self.repository.result_directory/"figures"
        )
        self.style = style or Plotter.PlotStyle()
        self.separate_figures = separate_figures

        parameter_colors: dict[str,str] = {}
        self.trace = Postprocess.TracePostprocess(
            Plotter.TracePlotter(
                self.figure_directory,self.style,parameter_colors
            ),
            separate_figures,
        )
        self.posterior_distribution = (
            Postprocess.PosteriorDistributionPostprocess(
                Plotter.DistributionPlotter(
                    self.figure_directory,self.style,parameter_colors
                ),
                separate_figures,
            )
        )
        self.posterior_pair = Postprocess.PosteriorPairPostprocess(
            Plotter.PairScatterPlotter(
                self.figure_directory,self.style,parameter_colors
            )
        )
        self.posterior_correlation = (
            Postprocess.PosteriorCorrelationPostprocess(
                Plotter.HeatmapPlotter(
                    self.figure_directory,self.style,parameter_colors
                )
            )
        )
        self.sampling_diagnostic = Postprocess.SamplingDiagnosticPostprocess(
            Plotter.BarPlotter(
                self.figure_directory,self.style,parameter_colors
            )
        )

    def _remove_stale_layout_files(
        self,
        parameter_names: list[str],
    ) -> None:
        """删除与当前子图输出模式相反的旧图"""

        if self.separate_figures:
            stale_stems = [
                "parameter_trace",
                "posterior_distributions",
            ]
        else:
            stale_stems = [
                f"parameter_trace_{name}"
                for name in parameter_names
            ]
            stale_stems.extend(
                f"posterior_distributions_{name}"
                for name in parameter_names
            )

        for stem in stale_stems:
            for extension in ("png","pdf"):
                path = self.figure_directory/f"{stem}.{extension}"
                path.unlink(missing_ok=True)

    def run(self) -> dict[str,Any]:
        """读取结果并生成第一、第二层全部图形"""

        result = self.repository.load()
        self._remove_stale_layout_files(result.parameter_names)
        return {
            "trace":self.trace.run(result),
            "posterior_distribution":self.posterior_distribution.run(result),
            "posterior_pairs":self.posterior_pair.run(result),
            "posterior_correlation":self.posterior_correlation.run(result),
            "sampling_diagnostic":self.sampling_diagnostic.run(result),
        }


def _print_paths(paths: Any,prefix: str = "") -> None:
    """递归输出后处理文件路径"""

    if isinstance(paths,dict):
        for name,value in paths.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            _print_paths(value,child_prefix)
        return

    print(f"{prefix}: {paths}")


def main() -> None:
    """从标准贝叶斯输出目录生成后处理图形"""

    parser = argparse.ArgumentParser(
        prog="bayesian-postprocess",
        description="Plot Bayesian-inversion samples and diagnostics.",
    )
    parser.add_argument(
        "result_directory",
        type=Path,
        nargs="?",
        default=None,
        help="Optional directory containing sampler_result_*.npz.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("input_card.json"),
        help="Input card providing output and postprocess settings.",
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=None,
        help="Optional figure output directory.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Save PNG only.",
    )
    args = parser.parse_args()

    input_card = JsonReader.load(args.config)
    result_directory = (
        args.result_directory
        if args.result_directory is not None
        else Path(input_card.output.directory)
    )
    style = Plotter.PlotStyle(save_pdf=not args.no_pdf)
    paths = PostProcessor(
        result_directory,
        args.figure_directory,
        style,
        input_card.postprocess.separate_figures,
    ).run()
    _print_paths(paths,"postprocess")


if __name__ == "__main__":
    main()
