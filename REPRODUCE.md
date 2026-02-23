# Dyn-HaMR 复现流程

基于 [Dyn-HaMR (CVPR 2025)](https://arxiv.org/abs/2412.12861) 的 4D 手部动作重建流程，使用 VIPE 替代 DROID-SLAM 做相机估计。

## 环境要求

两个 conda 环境：

| 环境 | 用途 | 关键依赖 |
|------|------|----------|
| `dynhamr` | HaMeR 手部检测 + Dyn-HaMR 优化/可视化 | PyTorch 2.7+, MANO |
| `vipe` | VIPE 相机估计 | PyTorch, vipe (pip install -e) |

## 目录结构

```
Dyn-HaMR/
├── dyn-hamr/                          # 主运行目录（所有 python 命令在此执行）
│   ├── run_opt.py                     # 主入口
│   ├── confs/
│   │   ├── config.yaml
│   │   └── data/video_vipe.yaml       # VIPE 数据配置
│   ├── data/vidproc.py                # 预处理（帧提取、HaMeR、VIPE 相机加载）
│   ├── optim/                         # 优化器
│   └── vis/                           # 可视化
├── third-party/
│   ├── hamer/                         # HaMeR 手部估计（修改版）
│   └── vipe/                          # VIPE 相机估计（修改版）
│       └── vipe_results/              # VIPE 输出目录
│           ├── pose/<seq>.npz         #   相机 pose (N, 4, 4)
│           └── intrinsics/<seq>.npz   #   相机内参 (N, 4) [fx, fy, cx, cy]
├── test/                              # 输入数据根目录
│   ├── videos/<seq>.mp4               #   输入视频
│   ├── images/<seq>/                  #   提取的帧（自动生成）
│   └── dynhamr/
│       ├── track_preds/<seq>/         #   HaMeR 检测结果（自动生成）
│       ├── cameras/<seq>/shot-0/      #   相机参数（自动生成）
│       └── shot_idcs/<seq>.json       #   镜头分段索引（自动生成）
├── outputs/logs/video-custom/<date>/  # 输出目录
│   └── <seq>-all-shot-0-0--1/
│       ├── *_smooth_fit_grid.mp4      #   主要结果（四宫格视频）
│       ├── *_src_cam.mp4              #   原视角叠加渲染
│       ├── *_{above,side,front}.mp4   #   3D 视角渲染
│       ├── root_fit/                  #   root 阶段优化参数
│       └── smooth_fit/                #   smooth 阶段优化参数
└── _DATA/                             # 模型权重
    ├── data/mano/                     #   MANO 模型
    └── hmp_model/                     #   HMP motion prior（可选）
```

## 完整流程

### 0. 准备输入

将视频放到 `test/videos/<seq>.mp4`。例如：

```bash
cp my_video.mp4 test/videos/my-video.mp4
```

> **长视频处理**：VIPE 对长视频（>400帧）可能 OOM。建议用 ffmpeg 分段：
> ```bash
> # 按帧数切割（每段 341 帧，25fps）
> ffmpeg -i test/videos/long-video.mp4 -vframes 341 -c copy test/videos/long-video-p1.mp4
> ffmpeg -i test/videos/long-video.mp4 -ss 13.64 -vframes 341 -c copy test/videos/long-video-p2.mp4
> ```
> 每段作为独立的 `<seq>` 分别跑完整流程。

---

### Stage 1: 帧提取 + HaMeR 手部检测

```bash
conda activate dynhamr
cd Dyn-HaMR/dyn-hamr

python run_opt.py data=video_vipe data.seq=<seq> run_opt=False run_vis=False
```

自动执行：
1. 从 `test/videos/<seq>.mp4` 提取帧到 `test/images/<seq>/`
2. YOLO 手部检测 + HaMeR 手部 mesh 估计
3. 结果保存到 `test/dynhamr/track_preds/<seq>/`
4. 如果已有 VIPE 结果，还会自动生成相机参数

---

### Stage 2: VIPE 相机估计

```bash
conda activate vipe
cd Dyn-HaMR/third-party/vipe

# VIPE_LITE=1 跳过深度模型（防止高分辨率 OOM），只保存 pose + intrinsics
VIPE_LITE=1 vipe infer test/videos/<seq>.mp4 -o vipe_results/
```

输出：
- `vipe_results/pose/<seq>.npz` — 相机外参 cam2world (N, 4, 4)
- `vipe_results/intrinsics/<seq>.npz` — 相机内参 [fx, fy, cx, cy] (N, 4)

> **注意**：`-o vipe_results/` 中的视频路径是**绝对路径或相对于当前目录**。
> VIPE 会自动用视频文件名（不含扩展名）作为序列名。

---

### Stage 3: 优化（root_fit + smooth_fit）

```bash
conda activate dynhamr
cd Dyn-HaMR/dyn-hamr

python run_opt.py \
    data=video_vipe \
    data.seq=<seq> \
    run_opt=True \
    run_vis=False \
    is_static=False \
    run_prior=False
```

- `root_fit`：50 iterations，初始化全局位移和尺度
- `smooth_fit`：300 iterations，联合优化手部 pose + 平滑约束

输出保存到 `outputs/logs/video-custom/<date>/<seq>-all-shot-0-0--1/`

---

### Stage 4: 可视化

```bash
python run_opt.py \
    data=video_vipe \
    data.seq=<seq> \
    run_opt=False \
    run_vis=True \
    is_static=False \
    run_prior=False
```

生成视频：
- `*_smooth_fit_grid.mp4` — 四宫格：输入 + above/side/front 3D 视角
- `*_smooth_fit_final_000300_src_cam.mp4` — 原视角手部叠加
- `*_smooth_fit_final_000300_{above,side,front}.mp4` — 单独 3D 视角

---

### 一步到位（优化 + 可视化）

```bash
conda activate dynhamr
cd Dyn-HaMR/dyn-hamr

python run_opt.py \
    data=video_vipe \
    data.seq=<seq> \
    run_opt=True \
    run_vis=True \
    is_static=False \
    run_prior=False
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data=video_vipe` | — | 使用 VIPE 相机数据配置 |
| `data.seq=<seq>` | — | 序列名（对应视频文件名，不含扩展名） |
| `run_opt` | True | 是否运行优化 |
| `run_vis` | True | 是否运行可视化 |
| `run_prior` | False | 是否运行 HMP motion prior（Stage III） |
| `is_static` | False | 相机是否静止（动态相机设 False） |

## 修改说明（相对上游）

### Dyn-HaMR (`dyn-hamr/`)
- `run_opt.py`: torch.load monkey-patch（PyTorch 2.7 兼容）
- `data/vidproc.py`: 添加 VIPE 相机加载 (`load_vipe_cameras`)
- `confs/data/video_vipe.yaml`: VIPE 数据配置
- `optim/optimizers.py`: 修复 checkpoint 恢复时 requires_grad 丢失
- `optim/output.py`: 修复 dtype 不匹配
- `HMP/fitting.py`: 修复多 chunk 拟合、移除 StepLR verbose

### VIPE (`third-party/vipe/`)
- `vipe/pipeline/default.py`: `VIPE_LITE=1` 跳过深度后处理
- `vipe/utils/io.py`: `VIPE_LITE=1` 跳过 RGB/depth/mask 保存

### HaMeR (`third-party/hamer/`)
- `run.py`: torch.load monkey-patch（PyTorch 2.7 兼容）

## Fork 仓库

| Repo | Branch |
|------|--------|
| [Dyn-HaMR](https://github.com/zekai-chen-deepreach/Dyn-HaMR/tree/dynhamr-compat) | `dynhamr-compat` |
| [vipe](https://github.com/zekai-chen-deepreach/vipe/tree/dynhamr-compat) | `dynhamr-compat` |
| [hamer](https://github.com/zekai-chen-deepreach/hamer/tree/dynhamr-compat) | `dynhamr-compat` |
