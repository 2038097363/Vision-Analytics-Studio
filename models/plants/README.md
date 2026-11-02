# BioCLIP 植物候选检索

本目录的清单固定了首版实验功能使用的 BioCLIP ViT-B/16 权重版本。大型权重位于 `weights/`，被 `.gitignore` 排除，应通过 `scripts/setup-plant-evaluation.ps1` 下载。

当前候选集只有 20 种常见植物。界面中的分数是集合内的相似排序，不是概率或准确率；模型不会自动确认物种，也不用于食用、毒性、药用价值或病因判断。

模型来源：https://huggingface.co/imageomics/bioclip

固定提交：`ce901ab3c6a913f9e9ef94ce6d27761069f4f01c`
