# 本地虚拟穿搭模型

开发机已完成 FASHN VTON v1.5 的本地端到端评测：RTX 5060 Laptop 8GB、BF16、576×864、20 步生成成功。主模型在 GPU 运行；当前 DWPose 使用 CPU ONNX Provider，只会增加预处理时间，不会外发图片。

权重和运行时均不提交到源码仓库。开发环境可在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-tryon-evaluation.ps1
```

脚本创建 `runtime/tryon/.venv`，复用电脑已有的 CUDA PyTorch，固定源码和三个模型仓库的提交版本，下载后按 `manifest.json` 校验大小与 SHA-256。桌面端生成时强制 `HF_HUB_OFFLINE=1`，缓存缺失时会明确失败，不会偷偷联网。

当前状态仍是开发评测。FASHN VTON 与 DWPose 主体为 Apache-2.0；`fashn-human-parser` 指向的 NVIDIA SegFormer 许可必须在软件商店发布前完成法律与分发范围审查。未完成审查前，不把这些权重打进安装包，也不向正式用户提供自动安装。
