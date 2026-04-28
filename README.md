# EC_proj

本仓库包含基于 BrainPy 的 V1 / VCD 相关实验代码与运行脚本。

## 主要运行脚本

当前项目的主要运行脚本是：

```bash
run_vcd_4rec.py
```

该脚本是运行 Temporal-LGN + 4-receptor-background VCD 实验的主要入口。

## Conda 环境配置

本项目提供了 `environment.yml` 文件，可以用于创建 conda 环境。

创建环境：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate <env_name>
```

其中 `<env_name>` 可以在 `environment.yml` 文件的第一行查看，例如：

```yaml
name: your_env_name
```

如果希望手动指定环境名，也可以使用：

```bash
conda env create -f environment.yml -n ec_proj
conda activate ec_proj
```

## 数据准备

项目所需的数据文件需要放在以下路径：

```bash
./data/GLIF_V1_network
```

推荐的项目目录结构如下：

```text
EC_proj/
├── run_vcd_4rec.py
├── environment.yml
├── brainpy_impl/
├── common/
├── tensorflow_impl/
└── data/
    └── GLIF_V1_network/
```

由于数据文件通常较大，本仓库不包含完整数据集。请手动将所需的 V1 / GLIF network 数据放到：

```bash
./data/GLIF_V1_network
```

## 运行前需要修改的数据路径

在运行项目之前，需要确认代码中的数据路径已经正确设置。

### 1. 修改 `brainpy_impl/load_sparse.py`

在以下文件中：

```bash
brainpy_impl/load_sparse.py
```

需要将其中的 `h5path` 和 `path` 都改为：

```python
"./data/GLIF_V1_network"
```

例如：

```python
h5path = "./data/GLIF_V1_network"
path = "./data/GLIF_V1_network"
```

### 2. 修改 `run_vcd_4rec.py` 中的数据路径

在以下文件中：

```bash
run_vcd_4rec.py
```

也需要将数据路径改为：

```python
"./data/GLIF_V1_network"
```

例如，如果脚本中包含类似：

```python
DEFAULT_DATA_DIR = ...
```

或者配置项中包含：

```python
"data_dir": ...
```

请将其改为：

```python
"./data/GLIF_V1_network"
```

## 运行方式

完成环境配置和数据路径修改后，可以运行：

```bash
python run_vcd_4rec.py
```

如果 `run_vcd_4rec.py` 中提供了额外的配置项或命令行参数，可以根据实验需求进一步修改配置。

## 注意事项

- `run_vcd_4rec.py` 是当前项目的主要运行脚本。
- `environment.yml` 用于通过 conda 复现运行环境。
- 数据目录需要统一设置为：

```bash
./data/GLIF_V1_network
```