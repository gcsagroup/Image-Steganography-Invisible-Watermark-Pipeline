# VAE Invisible 在 Wispio 中的分发许可提案

**状态：待权利人书面确认。** 本文件供这个模型的权利人与 Wispio 发行方核对权利范围。当前没有针对整个仓库的开源许可。以下提案仅适用于 Wispio 使用的提取器。

## 提请权利人批准的范围

对签署人拥有或有权许可的部分，拟授予 Global Cybersecurity Alliance Limited 一项非独占许可，使其可以为 Wispio 复制、修改、转换提取器推理代码、微调编码器和稳定图，将下表中的移动端产物随 Wispio 分发，并允许应用商店为交付该 App 而托管和复制这些产物。适用渠道为 Wispio 的测试分发及正式应用商店分发。独立下载权重、公开发布训练数据、发布第三方基础模型以及其他产品的使用均不在本提案范围内。

提案生效需要相应权利人书面确认其身份、所控制的源文件或资产、适用渠道、许可生效日期和需随包保留的声明，并引用本文件的提交与资产摘要。任何第三方权利继续适用其原条款。

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

## 需要确认的权利

- **基础权重。** 上表基础 VAE 的大小和 SHA-256 与 Stability AI 发布的 SD3.5 Large、Medium 及 Large Turbo 中的 VAE 一致。相同文件出现在多个发行版本中，仍需由获取方确认当时的来源、接受的具体许可版本及适用资格。若适用 Stability AI Community License，应按相应版本落实商用条件、许可副本、署名和修改说明。另有商业授权时，应确认其覆盖移动端衍生权重的分发。[Large 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f/vae/diffusion_pytorch_model.safetensors) · [Medium 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/b940f670f0eda2d07fbb75229e779da1ad11eb80/vae/diffusion_pytorch_model.safetensors) · [Large Turbo 权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/ec07796fc06b096cc56de9762974a28f4c632eda/vae/diffusion_pytorch_model.safetensors) · [Stability AI 许可条款](https://stability.ai/community-license-agreement)
- **代码与微调部分。** 请维护者列明原创推理代码的文件、作者或获授权代表，说明微调训练所用基础权重与数据来源，并确认有权许可相应的微调贡献和移动端转换。训练时使用的第三方库和引用代码应保留各自声明。
- **稳定图。** 上游 README 将该图描述为基于 ALASKA2 计算。请记录适用的数据使用条件，以及生成的统计图和由其转换的移动端位置配置能否随 Wispio 分发。移动端分发范围不含原始数据集图片。[稳定图说明](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline/blob/9f126633893651140b0821f23dda1b848d6bdcfd/README.md)
