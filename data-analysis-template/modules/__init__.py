"""数据分析模块包

包含数据加载、清洗、探索性分析、可视化和机器学习等模块。
"""

from .loader import load_data, get_data_info, preview_data
from .cleaner import (
    get_missing_info,
    handle_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    remove_duplicates,
    convert_dtype
)
from .eda import (
    descriptive_stats,
    correlation_analysis,
    group_analysis,
    distribution_analysis
)
from .visualizer import (
    plot_line,
    plot_bar,
    plot_pie,
    plot_heatmap,
    plot_box,
    plot_scatter,
    plot_histogram
)
from .modeler import (
    train_logistic_regression,
    train_random_forest,
    train_kmeans,
    train_linear_regression
)

__all__ = [
    # loader
    "load_data",
    "get_data_info",
    "preview_data",
    # cleaner
    "get_missing_info",
    "handle_missing_values",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "remove_duplicates",
    "convert_dtype",
    # eda
    "descriptive_stats",
    "correlation_analysis",
    "group_analysis",
    "distribution_analysis",
    # visualizer
    "plot_line",
    "plot_bar",
    "plot_pie",
    "plot_heatmap",
    "plot_box",
    "plot_scatter",
    "plot_histogram",
    # modeler
    "train_logistic_regression",
    "train_random_forest",
    "train_kmeans",
    "train_linear_regression",
]
