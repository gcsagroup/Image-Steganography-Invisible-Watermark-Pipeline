# VAE Invisible 在 Wispio 中的分发权限

**现有 VAE 模型包暂未取得完整的商店分发依据。** 2026-11-16，GCSA 组织管理员 `GCSA-ToolsTeam` 就 GCSA 拥有或有权许可的贡献，书面同意 Global Cybersecurity Alliance Limited 为 Wispio 非独占地复制、修改、转换提取器并分发指定的移动端产物。确认范围包括 `extraction/service.py`、`models/vae.py` 的提取路径及 `FineTurnEncoder.pth` 中的 GCSA 贡献。确认绑定提交 `58c5ba0ad0479be379c7a2dc76fbbe1da28c2a4c` 和下表的资产摘要。[管理员书面确认](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline/pull/1#pullrequestreview-5918019726)

Wispio 1.0.0 的测试分发、TestFlight、App Store 和 Apple 专供该 App 下载的托管资源均在这项许可的渠道范围内。发行方须保留适用的版权、许可、署名及修改声明。1.1.0 计划提供的 GitHub 独立权重下载需要另行取得相应分发依据。此次确认也未授权公开训练数据、分发第三方基础模型或供其他产品使用。仓库内第三方材料继续适用各自条款。

## 核定的文件

本仓库的 [VAE 发布](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline/releases/tag/VAE)和 Wispio 使用的 [VAE_v1.0 发布](https://github.com/gcsagroup/watermark/releases/tag/VAE_v1.0)对以下两个文件报告相同的大小和 SHA-256。表中后续产物来自 Wispio 的固定清单。单文件使用 SHA-256，Core ML 目录使用 Wispio 的 `sha256-tree-v1`。

| 资产 | SHA-256 | 用途 |
| --- | --- | --- |
| `StableDiffusion3VAE.safetensors` | `8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc` | 基础 VAE，167,666,902 字节 |
| `FineTurnEncoder.pth` | `609a6724f0e13ef11d1af6e71251dd02fdcf84755139ab1c2b561519088e8611` | 微调后的编码器，137,137,394 字节 |
| `modified_encoder_bf16.onnx` | `7e9acc61c303c268d4a992f3810f497a5b21878dcc4b6cdaafabbd51c9c417ac` | Wispio 采用的转换输入，来自 VAE_v1.0 |
| `Weights/stability_1024x1024.npz` | `b7f4552926a91944eb8c96aeb73d89e9ba5c109d7aedc41339d6baacd3e34aa6` | 源稳定图，与 Wispio 固定的输入一致 |
| `vae-invisible-v1-1024-fp16.mlpackage` | `7bdb50051c3bc969fb4c58941837bca3849cca78a5eb1cc0f8235f06613e6847` | iOS 提取器 |
| `vae-invisible-v1-1024-fp16.tflite` | `2f566805563e1ae6291f60f187efbc68eaa0957d93decb5ad07558e2cd788a92` | Android 提取器 |
| `vae-invisible-v1-1024-stability.json` | `ec275a9d47572f0d4683111f06435d90b3f9e26aa521b1975ccf97910cfc38cc` | 移动端随包的位置配置 |

移动端使用提取编码器和位置配置。原始 VAE 编解码器、safetensors 及 PyTorch 训练权重仅用于追溯。[上游管线源码](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline/tree/9f126633893651140b0821f23dda1b848d6bdcfd) · [VAE_v1.0 源码](https://github.com/gcsagroup/watermark/tree/979b1e96a7c217c56c383ec1b4ee4d9cd9da7c6b/packages/model-assets/tools/invisible_watermark) · [Wispio 移动端清单](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/mobile/research/research-native-models.json)

## 发行前需要补齐的依据

- **基础权重。** 上表基础 VAE 的大小和 SHA-256 与 Stability AI 发布的 SD3.5 Large、Medium 及 Large Turbo 中的 VAE 一致。相同文件出现在多个发行版本中，仍需由获取方确认当时的来源、接受的具体许可版本及适用资格。如适用所引 Large 版本的社区许可，商用须注册并核实收入条件，分发须提供许可副本、保留 Notice 并按条款展示 “Powered by Stability AI”。微调和格式转换的修改应在声明中说明。另有商业授权时，应确认其覆盖移动端衍生权重的分发。[Large 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f/vae/diffusion_pytorch_model.safetensors) · [Medium 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/b940f670f0eda2d07fbb75229e779da1ad11eb80/vae/diffusion_pytorch_model.safetensors) · [Large Turbo 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/ec07796fc06b096cc56de9762974a28f4c632eda/vae/diffusion_pytorch_model.safetensors) · [该版本随附的 Stability AI 许可](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f/LICENSE.md)
- **代码与微调部分。** GCSA 的书面确认只覆盖其有权许可的贡献。微调训练所用基础权重、数据来源及其他贡献者的授权仍需记录到对应转换件。训练时使用的第三方库和引用代码应保留各自声明。
- **稳定图。** 上游 README 记载 `Weights/stability_1024x1024.npz` 由 ALASKA2 数据计算。数据发布方标注 CC BY-NC-ND，并说明商用及分发基于其材料制作的内容须事先获作者许可。现有记录未提供覆盖这张图及 `vae-invisible-v1-1024-stability.json` 的例外许可。发行前应取得覆盖这两项产物及用途的书面依据，或用权利明确的数据重新生成、验证并登记新摘要。[稳定图来源](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline/blob/9f126633893651140b0821f23dda1b848d6bdcfd/README.md#L235) · [ALASKA2 发布方条款](https://alaska.utt.fr/)
