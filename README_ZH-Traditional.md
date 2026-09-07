# 通用的魯棒圖像隱寫術

[簡體中文](README.md) | [繁體中文](README_ZH-Traditional.md) | [English](README_EN.md)

本項目是一套基於 VAE 在圖像潛空間 latent 上進行秘密信息隱藏的圖像隱寫管線，覆蓋載體圖像預處理、秘密消息成幀與糾錯、VAE 編解碼、穩定嵌入位置選擇、秘密提取、JPEG 魯棒性評估、圖像質量評估和批量測試。所有運行參數集中在 [`Config/config.yaml`](Config/config.yaml) 中管理。


## 核心特性

- 使用 SD3 系列 LDM 模型原生 VAE 將載體圖像編碼為 `latent`，並將嵌入秘密信息後的 `latent` 解碼為隱寫圖像。
- 支持直接輸入 `[16,H,W]` 或 `[1,16,H,W]` 的 `.pt` 格式 `latent`，便於繞過圖像編碼階段做研究和調試。
- 支持 UTF-8、GBK、ASCII 和 UTF-16 文本編碼，也支持原始字節或字節對齊 bit 序列。
- 使用緊湊的固定長度頭部幀記錄編碼方式、ECC 方案、Payload 長度和校驗信息，不依賴容易受損的結束標記。
- 可選 Hamming(7,4) 糾錯，並使用 CRC8/CRC32 分別校驗頭部和 Payload。
- 支持嚴格穩定圖、歸一化插值和解析式頻帶三種嵌入位置策略。
- 輸入圖像尺寸不規整時，可選擇直接拉伸或保持比例縮放後中心裁剪。
- 容量隨實際載體像素數動態計算，默認預算為 `0.005 bit/pixel`。
- 內置 PNG、JPEG 90/70/50 提取測試，以及 PSNR、SSIM、LPIPS 圖像質量評估。

## 工作流程


**嵌入流程：**
```text
載體圖像
   │
   ├─ 尺寸檢查與規范化 ──> RGB 圖像
   │
   └─ 原生 VAE（FP32）編碼 ──> cover latent
                                      │
秘密文本/字節/bit ──> 編碼 ──> 頭部幀 ──> ECC ──> 最終 bit 流
                                      │
穩定性得分圖/解析頻帶 ──> DCT 位置選擇 ──> alpha 計算 ──> latent 嵌入
                                                              │
                                                        原生 VAE 解碼
                                                               │
                                                            隱寫圖像
```

**提取流程：**
```text
PNG/JPEG 隱寫圖像 ──> 微調編碼器 ──> latent ──> 相同位置與參數提取
                     ──> 頭部多數表決 ──> ECC 糾錯 ──> CRC 校驗 ──> 文本/字節
```

嵌入端和提取端必須使用一致的模型權重、穩定性策略、通道分組和隱寫參數，否則雙方選擇的位置或判決邊界可能不同。

## 項目結構

```text
PipeLine/
├── Config/
│   └── config.yaml                 # 唯一配置文件，中英文注釋
├── Weights/
│   ├── vae_config.json              # 原生 VAE 結構配置
│   ├── diffusion_pytorch_model.safetensors # 原生 VAE 權重
│   ├── encoder_final.pth            # 微調後的獨立提取編碼器
│   ├── stability_*.npz              # 壓縮穩定性得分圖
├── Stego/                           # 默認隱寫 PNG 輸出目錄
├── Test/
│   ├── Cover/                       # 批量測試載體圖像
│   ├── Stego/                       # 批量測試隱寫圖像
│   └── Reports/report.md            # 最新批量測試報告
├── configuration/                   # YAML 加載、校驗與配置對象
├── preprocessing/                   # 圖像規范化、Payload、幀頭、ECC
├── models/                          # 原生 VAE 與改造編碼器的延遲加載
├── stability/                       # 穩定圖生成、匹配、插值與加載
├── steganography/                   # DCT、位置選擇、alpha 和 bit 編解碼
├── embedding/                       # 完整嵌入服務
├── extraction/                      # 完整提取服務
├── evaluation/                      # JPEG 攻擊與質量/準確率指標
├── testing/                         # 批量測試和 Markdown 報告生成
├── tests/                           # 單元測試和偽模型集成測試
├── pipeline.py                      # Python 公共門面 StegoPipeline
├── cli.py                           # 命令行入口
├── position_strategy_debug.py       # 兩種無嚴格穩定圖策略的調試入口
├── tools/export_onnx.py              # 可重復執行的 ONNX 導出工具
├── profiles.py                      # 圖像規格與動態容量 profile
├── schemas.py                       # 輸入輸出數據結構
└── requirements.txt                 # Python 依賴
```

## 模型權重與作用

正式 Pipeline 的模型文件位於 `PipeLine/Weights/`，配置默認直接從該目錄加載：

```text
PipeLine/Weights/vae_config.json
PipeLine/Weights/diffusion_pytorch_model.safetensors
PipeLine/Weights/encoder_final.pth
```

路徑可在 `Config/config.yaml` 的 `model.original_vae` 和 `model.modified_encoder` 中修改；相對路徑以 `PipeLine/` 為基準解析。

兩個模型在鏈路中的職責不同：

| 模型 | 文件 | 作用 |
| --- | --- | --- |
| 原生 VAE | `vae_config.json` + `diffusion_pytorch_model.safetensors` | 嵌入前將載體 RGB 圖像編碼為 latent；DCT 嵌入完成後再將 stego latent 解碼為 PNG 隱寫圖像 |
| 微調後的獨立編碼器 | `encoder_final.pth` | 提取時將 PNG/JPEG 隱寫圖像重新編碼為 latent，使後續 DCT 判決、頭部恢復和 ECC 解碼能夠提取秘密 |

### 權重存儲精度與 FP32 upcast

經實際逐張量檢查，`diffusion_pytorch_model.safetensors` 的 244 個張量均以 **BF16** 保存。BF16 只是減小 checkpoint 體積的存儲形式；加載時由 Pipeline 構建 FP32 模型並把權重上轉換到 FP32，後續 VAE 推理也保持 FP32，以取得當前項目的最佳效果。

當前倉庫中的 `encoder_final.pth` 經檢查包含 106 個 **FP32** 張量，本身已是 FP32，並非 BF16。Pipeline 仍顯式按 `model.dtype: float32` 加載它。若未來將微調編碼器另存為 BF16，也必須在加載後上轉換為 FP32，再引用本項目的 FP32 評測結論。這裡以文件實際 dtype 為準，不將當前編碼器錯誤標注為 BF16。

兩個復制後的權重已與原始 `Weights/` 文件進行 SHA-256 一致性校驗，內容保持不變。

## 環境安裝

推薦使用支持 CUDA 的 PyTorch 環境。項目已按本機 Conda `base` 環境進行測試：

```powershell
conda activate base
python -m pip install -r PipeLine/requirements.txt
```

主要依賴包括 PyTorch、Diffusers、Safetensors、SciPy、Pillow、PyYAML、scikit-image 和 LPIPS。若使用 GPU，請先安裝與本機 CUDA 驅動匹配的 PyTorch 版本。

運行前建議確認：

1. 原生 VAE 配置和權重存在；
2. 改造編碼器權重存在；
3. `model.dtype` 保持為 `float32`；
4. 所選穩定性策略具備對應穩定圖，或已配置可用的回退策略；
5. GPU/系統內存足以處理目標分辨率。

## 配置文件

全部配置均位於 [`Config/config.yaml`](Config/config.yaml)，並按以下大類組織：

| 配置段 | 用途 |
| --- | --- |
| `model` | 設備、FP32 精度、VAE/編碼器權重、latent 約定 |
| `preprocessing` | 文本編碼、ECC、頭部重復、圖像尺寸調整、動態容量 |
| `stability` | 穩定圖生成、保存和三種匹配策略 |
| `steganography` | 通道分組、位置數、閾值、alpha 與自適應參數 |
| `extraction` | 輸入歸一化、投票平局規則、文本解碼策略 |
| `evaluation` | PSNR、SSIM、LPIPS 與 JPEG 攻擊等級 |
| `testing` | 測試目錄、隨機種子、容量利用率和報告輸出 |

指定自定義配置文件時，將全局參數放在子命令之前：

```powershell
python -m PipeLine.cli --config path/to/config.yaml embed --image cover.png --text "秘密信息"
```

## 圖像預處理與尺寸規范化

VAE 需要圖像寬高與下采樣結構兼容。`preprocessing.image_adjustment.multiple` 定義輸出寬高必須滿足的倍數，默認值為 `8`，並且必須兼容 `model.latent.downsample_factor`。

當輸入寬高不是該倍數時，Pipeline 會發出 warning，明確顯示原始尺寸和調整後尺寸。可選方法如下：

- `stretch`：分別將寬和高直接縮放到最近的合法倍數。速度直接，但可能輕微改變原始寬高比。
- `scale_crop`：保持寬高比進行縮放，再從中心裁剪到最近的合法尺寸。默認使用該方法，避免幾何拉伸，但可能裁去少量邊緣內容。

輸出采用配置中的 `nearest`、`bilinear`、`bicubic` 或 `lanczos` 重采樣算法。默認配置為：

```yaml
preprocessing:
  image_adjustment:
    multiple: 8
    method: scale_crop
    resample: lanczos
```

`1024x1024`、`1536x1024`、`1024x1536` 是預設隱寫參數 profile。其他合法尺寸會根據實際寬高動態創建 profile，並復用幾何上最接近的預設隱寫參數。

## Payload、頭部幀與 ECC

輸入秘密可以是以下三類之一：

- `text`：按配置的 UTF-8、GBK、ASCII 或 UTF-16 編碼；
- `data`：直接輸入任意字節；
- `bits`：輸入只包含 0/1 且長度能按字節對齊的序列。

字節以 MSB-first 順序轉換為 bit。默認最終碼流為：

```text
固定頭部 × 3 + Hamming74(payload_bits)
```

每份邏輯頭部為 67 bit：

| 字段 | 長度 | 說明 |
| --- | ---: | --- |
| 文本編碼 | 2 bit | `00=UTF-8`、`01=GBK`、`10=ASCII`、`11=UTF-16` |
| ECC 方案 | 1 bit | `0=none`、`1=hamming74` |
| 原始 Payload 長度 | 24 bit | 記錄糾錯編碼前的有效 bit 數 |
| Payload CRC32 | 32 bit | 驗證最終恢復的數據 |
| 頭部 CRC8 | 8 bit | 驗證邏輯頭部 |

頭部默認重復 3 次並逐 bit 多數表決，因此頭部總開銷為 201 bit。提取端從頭部獲得實際 Payload 長度和 ECC 參數，不使用結束位標記，也不依賴掃描某個終止序列。

Hamming(7,4) 可糾正每個 7-bit 碼字內的單 bit 錯誤，但不能保證識別或恢復所有多 bit 錯誤；CRC32 是 Payload 完整性是否通過的最終依據。ECC 只提升容錯能力，隱寫本身也不提供加密，敏感秘密應在嵌入前單獨加密。

## 容量計算

圖像的像素預算上限按實際規范化後尺寸動態計算：

```text
pixel_budget = floor(width × height × capacity_bits_per_pixel)
```

默認 `capacity_bits_per_pixel = 0.005`。例如，1024×1024 圖像的像素預算為 5242 bit。該數值是**最終成幀碼流**的預算，包含重復頭部和 ECC 冗余，並不等於用戶可輸入的純 Payload 長度。

### 高容量模式

在以 **PNG 提取準確率保持 100%、LPIPS 維持在約 0.05** 為目標的場景中，可通過同時降低基礎嵌入強度 `base_alpha`、針對實際穩定圖重新選擇 `stability_threshold`，並提高每個 latent 8×8 塊的 `positions_per_block`，將更多 bit 分散到更多穩定位置。在完成針對性標定和逐圖驗證後，可將 `capacity_bits_per_pixel` 提高到 `0.02`：

```text
floor(1024 × 1024 × 0.02) = 20971 bit
```

即 1024×1024 圖像的最終成幀嵌入容量約為 **2 萬 bit**。這裡的 20971 bit 包含重復頭部和 ECC 冗余；啟用默認 Hamming(7,4) 後，用戶可用的原始 Payload bit 數會更少。

該高容量結論屬於需要重新標定的調參目標，不是當前默認 `positions_per_block=3`、`stability_threshold=0.8`、`base_alpha=0.22` 配置的自動保證。實際 DCT 可選位置必須不少於像素預算，同時應使用與目標圖片風格、分辨率和比例匹配的穩定性得分圖。降低 `base_alpha` 有助於控制模糊和 LPIPS，提高 `positions_per_block` 用於補充容量，而 `stability_threshold` 需要在位置數量與可靠性之間重新選擇；最終必須在獨立驗證集上確認 PNG 提取達到 100%，並逐圖檢查 LPIPS 是否維持在約 0.05。

實際可嵌入量還受到 DCT 穩定位置數量限制：

```text
effective_capacity = min(pixel_budget, selected_DCT_capacity)
```

因此，即使圖像像素預算足夠，當穩定性閾值過高、每塊位置數過少或所選穩定圖可用位置不足時，也可能報告容量不足。啟用 Hamming(7,4) 後，每 4 bit Payload 會擴展為 7 bit，且還需扣除 201 bit 頭部開銷，所以有效文本容量明顯小於 `pixel_budget`。

## 穩定性得分圖與位置策略

穩定性得分描述 latent DCT 位置經過 VAE 解碼和重新編碼後保持數值/判決穩定的程度。嵌入和提取復用相同的得分與排序邏輯。

> **當前 1024×1024 穩定圖的數據域限制（重要）**
>
> 當前隨項目提供的 `PipeLine/Weights/stability_1024x1024.npz` 基於 **Alaska2 數據集**計算得到。該數據集樣本的原始分辨率和圖像質量相對有限，其內容風格、紋理分布、壓縮特征與真實業務圖片可能存在明顯差異。因此，這張穩定圖只能作為已有實驗先驗或缺少目標數據時的回退基線，不能視為適用於所有圖片的通用穩定性指標。
>
> 當它被直接用於其他數據域，或經 `normalized_interpolation` 擴展到不同分辨率和寬高比時，選出的 DCT 位置可能並非目標圖片中真正穩定的位置。這既可能降低 PNG/JPEG 下的秘密提取精度、增加頭部或 Payload CRC 錯誤，也可能把較強修改施加到視覺敏感區域，從而降低 PSNR、SSIM、LPIPS 等圖像質量表現。

為了取得最佳效果，正式部署前應使用實際業務的一批代表性圖片重新運行穩定性得分圖計算。樣本應覆蓋預計輸入中的不同內容風格、紋理復雜度、清晰度、來源/壓縮質量、分辨率和寬高比。建議按“風格或數據來源 + 分辨率/比例檔位”劃分數據組，並為各組分別生成、驗證和保存匹配的穩定性得分圖，而不是只依賴 Alaska2 的單一 1024×1024 先驗。

推薦流程如下：

1. 從真實業務數據中抽取具有代表性的樣本，並保留獨立驗證集。
2. 按攝影/插畫/低紋理/高紋理等風格，以及 1:1、3:2、2:3、16:9 等常見比例和目標分辨率分組。
3. 對每組圖片使用正式部署所采用的模型權重、FP32 精度、預處理和 latent 規范生成原始/重建 latent 對。
4. 分組計算穩定性得分圖，並在對應圖片組上重新標定 `stability_threshold`、`positions_per_block` 和 `base_alpha`。
5. 使用未參與計算的驗證圖片檢查 PNG/JPEG 提取率、CRC 成功情況和圖像質量，推薦繼續以單圖 `LPIPS < 0.05` 為質量目標。
6. 部署時根據輸入圖片所屬的數據域、分辨率和比例選擇最匹配的穩定圖；沒有嚴格匹配時才使用插值或解析頻帶策略回退。

如果同一分辨率包含差異很大的圖片風格，建議維護不同業務場景的 `Weights` 集合或獨立配置，避免同名 profile 的穩定圖相互覆蓋。穩定圖生成數據、模型權重、精度和關鍵預處理發生變化後，都應重新驗證，不能沿用原有成績。

通過 `stability.map_matching.strategy` 選擇策略：

### `strict`

只接受與目標圖像 profile 嚴格對應的 `stability_<寬>x<高>.npz`。該模式可控性最好，適合正式評測和對固定分辨率做過充分預計算的環境；缺少對應文件時直接報錯。

### `normalized_interpolation`

優先加載嚴格對應的穩定圖。若不存在，則綜合寬高比差異和面積差異，從 `PipeLine/Weights/` 中選擇最接近的穩定圖，並插值到目標 latent 尺寸。該模式適合分辨率和比例不固定的輸入，也是當前默認策略。

插值只是工程回退方案，不能等價於針對目標分辨率預計算的穩定圖。對於高可靠性需求，應逐步補充常用分辨率的原生穩定圖並復測。

### `analytic_band`

不依賴任何先驗穩定圖，按 DCT 歸一化徑向頻率、頻帶中心、帶寬和通道先驗解析生成得分。它適合未知尺寸、冷啟動或對照實驗，但穩定性來自頻帶假設而非真實重建統計，必須獨立評估提取率與圖像質量。

## 隱寫控制參數

每個圖像 profile 都有獨立的 `positions_per_block`、`stability_threshold` 和 `base_alpha`。這裡的“8×8 塊”是 **latent 空間的 8×8 DCT 分塊**，不是直接在原圖像像素上分塊。

| 參數 | 直接作用 | 增大後的典型影響 |
| --- | --- | --- |
| `positions_per_block` | 每個 latent 8×8 塊、每個通道最多選擇多少個 DCT 位置 | 候選容量上升、修改密度增大、圖像質量更易下降 |
| `stability_threshold` | 只允許得分不低於閾值的位置參與嵌入 | 可用位置減少、容量下降；通常位置更可信，但過高可能直接容量不足 |
| `base_alpha` | 每個 latent 通道的基礎嵌入強度/判決裕量 | 魯棒性通常上升，但失真和細微模糊更明顯 |

三者的關系如下：

- `positions_per_block` 與 `stability_threshold` **共同決定穩定位置的實際容量上限**。前者給每塊設置數量上限，後者再過濾不夠穩定的位置。
- `base_alpha` 不直接增加名義 bit 容量，主要決定嵌入後的判決裕量與魯棒性。
- 增加 `positions_per_block` 或降低 `stability_threshold` 通常可以容納更多 bit，但會使用更多或較不穩定的位置。
- 提高 `base_alpha` 通常有利於抵抗 VAE 重編碼和 JPEG 壓縮，但會增大 latent 修改幅度。
- 更高嵌入容量和更強 `base_alpha` 都會更明顯地損害圖像質量；本系統常見視覺副作用是隱寫圖像出現細微模糊或局部紋理軟化。

`base_alpha` 是 16 個 latent 通道各自的數組。啟用 `content_adaptive` 和 `frequency_adaptive` 後，實際位置使用的 alpha 還會根據內容能量和頻率權重調整，並受 `minimum_alpha`、`maximum_alpha` 限制。因此配置中的 `base_alpha` 是基礎值，不一定是每個位置最終使用的絕對值。

默認 profile 示例：

```yaml
steganography:
  profiles:
    1024x1024:
      positions_per_block: 3
      stability_threshold: 0.8
      base_alpha: [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
                   0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]
```

## 圖像質量與調參建議

本項目建議將 **LPIPS 保持在 `0.05` 左右** 作為視覺不可察覺性的工程目標。LPIPS 越低通常表示感知差異越小；該閾值是推薦控制線，不是對所有圖像、Payload 和攻擊場景的絕對保證。實際驗收應同時查看原圖/隱寫圖對比、局部紋理、單圖 LPIPS 以及業務所需的壓縮魯棒性。

推薦調參順序：

1. 保持 FP32，固定模型、穩定圖策略和一組代表性載體圖像。
2. 從默認的 `positions_per_block=3`、`stability_threshold=0.8` 和 `base_alpha=0.22` 起步。
3. 先確認最終碼流長度低於像素預算和實際 DCT 容量。
4. 每次只改變一個參數，並同時記錄 PNG/JPEG 提取結果與 PSNR、SSIM、LPIPS。
5. 若 LPIPS 超過 0.05 或模糊明顯，優先減少 Payload/位置使用量，或適當降低 `base_alpha`；也可提高 `stability_threshold`，僅使用更高分位置。
6. 若圖像質量合格但 JPEG 提取不穩，可小步提高 `base_alpha`，或改善穩定圖、位置篩選和 ECC；不要只用大幅增加強度解決問題。
7. 不僅觀察平均 LPIPS，還應關注最差樣本和高分位數，避免少數紋理敏感圖像出現明顯失真。

容量、魯棒性和圖像質量不可同時無限提高。建議先根據業務確定實際 Payload 長度和目標攻擊等級，再尋找滿足 `LPIPS < 0.05` 的最低嵌入強度。

## Python API

### 圖像嵌入、提取與評估

```python
from PipeLine import StegoPipeline

pipeline = StegoPipeline.from_config()

embedded = pipeline.embed(
    image="cover.png",
    text="秘密信息",
    output_name="example",
)

extracted = pipeline.extract(image=embedded.stego_path)
print(extracted.status)
print(extracted.decoded_text)

evaluation = pipeline.evaluate("cover.png", embedded)
print(evaluation.psnr, evaluation.ssim, evaluation.lpips)
for attack in evaluation.jpeg_results:
    print(attack.quality, attack.raw_bit_accuracy, attack.post_ecc_bit_accuracy)
```

默認隱寫圖像保存到 `PipeLine/Stego/`，格式始終為 PNG，避免保存階段引入額外有損壓縮。

### 分階段預處理

```python
prepared = pipeline.preprocess(image="cover.png", text="秘密信息")
embedded = pipeline.embed(prepared=prepared, output_name="example")
```

### 直接使用 latent

```python
embedded = pipeline.embed(latent_path="cover.pt", text="秘密信息")
extracted = pipeline.extract(latent_path="stego_latent.pt")
```

`.pt` 文件必須只包含一個 `[16,H,W]` 或 `[1,16,H,W]` Tensor。系統根據 `model.latent.downsample_factor` 從 latent 空間尺寸反推圖像 profile。輸入 latent 必須符合配置中的 `shifted_scaled` 約定；舊實驗中未應用相同 shift/scale 的 latent 不能直接混用。

### 評估已有圖像對

```python
evaluation = pipeline.evaluate_existing(
    cover_image="cover.png",
    stego_image="stego.png",
    text="秘密信息",
)
```

載體圖和隱寫圖經規范化後必須屬於同一 profile，參考秘密必須與嵌入時一致。

## 命令行使用

以下命令均在工作區根目錄運行。

### 預處理並保存 latent

```powershell
python -m PipeLine.cli preprocess `
  --image cover.png `
  --text "秘密信息" `
  --save-latent cover.pt
```

### 嵌入

```powershell
python -m PipeLine.cli embed `
  --image cover.png `
  --text "秘密信息" `
  --output-name example
```

也可以使用 `--text-file`、`--data-hex` 或 `--bits`，它們與 `--text` 互斥：

```powershell
python -m PipeLine.cli embed --image cover.png --data-hex "48656c6c6f"
python -m PipeLine.cli embed --image cover.png --bits "0100100001101001"
```

### 提取

```powershell
python -m PipeLine.cli extract --image PipeLine/Stego/example.png
```

### 評估已有載體/隱寫圖像

```powershell
python -m PipeLine.cli evaluate `
  --cover cover.png `
  --stego PipeLine/Stego/example.png `
  --text "秘密信息"
```

### 批量測試

```powershell
python -m PipeLine.cli test
```

批量測試讀取 `PipeLine/Test/Cover/`，生成的隱寫 PNG 保存到 `PipeLine/Test/Stego/`，逐圖指標和匯總結果寫入 [`Test/Reports/report.md`](Test/Reports/report.md)。隨機 Payload 的種子、容量利用率、最大圖像數量等由 `testing` 配置段控制。

當前測試配置使用：

```yaml
testing:
  payload_utilization: 0.5  # 使用有效最大容量的 50%
  max_images: null          # 掃描 Test/Cover 中全部圖像
```

這裡的 50% 先作用於 `min(pixel_budget, selected_DCT_capacity)`，再扣除 201-bit 頭部並反推 ECC 前可生成的整字節 Payload；報告中的 `Payload bits` 因此是原始秘密 bit 數，不是包含頭部和 Hamming 冗余的最終碼流長度。

## 生成穩定性得分圖

準備兩個文件夾，其中相同 base id 的 `.pt` 文件分別表示原始 latent 和經過目標鏈路重建的 latent：

```powershell
python -m PipeLine.cli stability `
  --original-dir path/to/original_latents `
  --reconstructed-dir path/to/reconstructed_latents
```

生成器根據匹配 latent 的數值差異計算穩定性得分，並按對應圖像規格壓縮保存，例如：

```text
PipeLine/Weights/stability_1024x1024.npz
PipeLine/Weights/stability_1536x1024.npz
PipeLine/Weights/stability_1024x1536.npz
```

可用 `--max-samples` 限制參與統計的樣本數，或用 `--output` 指定輸出文件。轉換舊的方形 `.pt` 穩定性緩存：

```powershell
python -m PipeLine.cli convert-stability `
  --input Weights/Alaska_Dct_stability.pt `
  --profile 1024x1024
```

穩定圖與模型精度、權重、編解碼路徑和數據分布相關。改變 VAE 權重、提取編碼器、數值精度或關鍵預處理後，應重新生成/驗證穩定圖，而不是沿用舊評測結論。

## 評估指標說明

| 指標 | 含義 | 方向 |
| --- | --- | --- |
| `raw_bit_accuracy` | 從判決位置直接提取的原始 bit 準確率 | 越高越好 |
| `post_ecc_bit_accuracy` | 頭部恢復與 ECC 糾錯後的 Payload bit 準確率 | 越高越好 |
| `text_match` | 原始文本與恢復文本是否完全一致 | `true` 為完全一致，`false` 為不一致 |
| `PSNR` | 原圖與隱寫圖的像素信噪比 | 通常越高越好 |
| `SSIM` | 原圖與隱寫圖的結構相似度 | 越接近 1 越好 |
| `LPIPS` | 感知特征空間差異 | 越低越好，建議 `< 0.05` |

報告中的 `N/A` 不等於準確率為 0。它表示該項無法可靠計算，例如頭部/CRC 恢復失敗、沒有可比較的有效 Payload，或隨機字節 Payload 本身不適用文本對比。判斷一次提取是否完整成功時，應聯合檢查 `status`、`error`、CRC、ECC 後準確率和 `text_match`，不能只看單個字段。

> **精度與評測聲明**
>
> 本項目以 **Float32（FP32）** 作為 VAE 編碼、解碼和隱寫評測的標準精度。在當前模型、改造編碼器和算法鏈路中，VAE 使用 FP32 運行可取得最優、最穩定的隱寫與提取效果。配置文件默認設置為 `model.dtype: float32`。原生 VAE checkpoint 雖以 BF16 存儲，但加載後必須上轉換（upcast）到 FP32 再推理；不能把“權重文件的存儲 dtype”誤當成“受保證的計算 dtype”。
>
> README、測試報告及項目當前給出的全部質量與魯棒性成績，**僅對 FP32 精度下的運行結果負責**。FP16、BF16、FP8 等低精度模式可能改變 latent 數值、DCT 系數和提取判決邊界，從而降低提取準確率、CRC 成功率或圖像質量；這些精度目前不屬於已保證范圍。如需部署低精度模型，必須針對目標硬件、模型精度和推理引擎重新生成或校準穩定性得分圖，並重新完成全套評測。

## 模型評測與參考結果

倉庫內現有 [`Test/Reports/report.md`](Test/Reports/report.md) 使用當前 `Test/Cover` 全部 20 張圖片、50% 有效最大容量、Hamming(7,4)、歸一化插值穩定圖策略和 **PyTorch FP32** 模型鏈路。20 張全部完成，無測試流程失敗。

| 匯總項 | 本輪結果 |
| --- | ---: |
| 成功 / 失敗圖片 | 20 / 0 |
| 平均 LPIPS | 0.0433 |
| PNG 原始 bit 平均準確率 | 0.9972 |
| PNG ECC 後平均準確率 | 0.9996 |
| JPEG 90 原始 / ECC 後平均準確率 | 0.9933 / 0.9984 |
| JPEG 70 原始 / ECC 後平均準確率 | 0.9644 / 0.9968 |
| JPEG 50 原始 / ECC 後平均準確率 | 0.9431 / 0.9936 |

JPEG ECC 後平均值會忽略頭部損壞而無法得到 Payload 的 `N/A` 樣本：JPEG 90、70、50 分別有 19、14、13 張具備可比較的 ECC 後結果。因此這些均值不能解釋為對應檔位的全樣本 CRC 成功率。完整的逐圖準確率和 `N/A` 分布應以報告表格為準。

這些結果只代表報告中列出的樣本、隨機 Payload、權重、配置和運行環境，不構成其他數據集、ONNX runtime 或低精度模式下的性能保證。雖然平均 LPIPS 0.0433 低於推薦控制線，但只有 12/20 張單圖低於 0.05，仍有 8 張需要按載體內容繼續降低容量或調整 alpha；正式應用不能只看平均值。

### Cover / Stego 參考示例

以下示例取自 `Test/Cover` 按文件名升序排列後的最後五張圖片。點擊圖片可查看原尺寸。文件名只顯示便於識別的簡稱，完整路徑保留在圖片鏈接中。

| Cover | Stego | 分辨率 | 容量 | 精度（Origin / ECC） | 視覺指標 |
| :---: | :---: | :---: | --- | --- | --- |
| <a href="Test/Cover/131106338932881396_p14.png"><img src="Test/Cover/131106338932881396_p14.png" width="320" alt="p14 Cover"></a><br><sub>p14</sub> | <a href="Test/Stego/131106338932881396_p14_stego.png"><img src="Test/Stego/131106338932881396_p14_stego.png" width="320" alt="p14 Stego"></a><br><sub>p14 stego</sub> | `1536×1024` | Payload <br>`2128 bit`<br>碼流 <br>`3925 bit`<br>上限<br> `10190 bit` | PNG<br> `0.9995 / 1.0000`<br>J90 <br>`0.9929 / 0.9977`<br>J70<br> `0.9740 / —`<br>J50 <br>`0.9592 / —` | PSNR<br> `31.41`<br>SSIM<br> `0.9414`<br>LPIPS <br>`0.0251` |
| <a href="Test/Cover/20260818140720_1563_31.jpg"><img src="Test/Cover/20260818140720_1563_31.jpg" width="320" alt="1563_31 Cover"></a><br><sub>1563_31</sub> | <a href="Test/Stego/20260818140720_1563_31_stego.png"><img src="Test/Stego/20260818140720_1563_31_stego.png" width="320" alt="1563_31 Stego"></a><br><sub>1563_31 stego</sub> | `1440×1920` | Payload <br>`3832 bit`<br>碼流 <br>`6907 bit`<br>上限 <br>`18157 bit` | PNG <br>`1.0000 / 1.0000`<br>J90<br> `0.9933 / 0.9997`<br>J70 <br>`0.8455 / —`<br>J50 <br>`0.7963 / —` | PSNR <br>`37.94`<br>SSIM<br> `0.9728`<br>LPIPS<br> `0.0207` |
| <a href="Test/Cover/v2-1a5f6ca2d7303de409237de4a6e70707_r.jpg"><img src="Test/Cover/v2-1a5f6ca2d7303de409237de4a6e70707_r.jpg" width="320" alt="v2-1a5f Cover"></a><br><sub>v2-1a5f…</sub> | <a href="Test/Stego/v2-1a5f6ca2d7303de409237de4a6e70707_r_stego.png"><img src="Test/Stego/v2-1a5f6ca2d7303de409237de4a6e70707_r_stego.png" width="320" alt="v2-1a5f Stego"></a><br><sub>v2-1a5f… stego</sub> | `512×512` | Payload <br>`256 bit`<br>碼流<br> `649 bit`<br>上限 <br>`1823 bit` | PNG<br> `0.9553 / 0.9922`<br>J90 <br>`0.9399 / 0.9805`<br>J70<br> `0.8814 / —`<br>J50 <br>`0.8505 / —` | PSNR<br> `27.53`<br>SSIM<br> `0.9544`<br>LPIPS<br> `0.0165` |
| <a href="Test/Cover/v2-33489711df97a738a22182a331d0cece_r.jpg"><img src="Test/Cover/v2-33489711df97a738a22182a331d0cece_r.jpg" width="320" alt="v2-3348 Cover"></a><br><sub>v2-3348…</sub> | <a href="Test/Stego/v2-33489711df97a738a22182a331d0cece_r_stego.png"><img src="Test/Stego/v2-33489711df97a738a22182a331d0cece_r_stego.png" width="320" alt="v2-3348 Stego"></a><br><sub>v2-3348… stego</sub> | `2240×1264` | Payload <br>`3928 bit`<br>碼流<br> `7075 bit`<br>上限<br> `18393 bit` | PNG <br>`1.0000 / 1.0000`<br>J90<br> `0.9945 / 0.9997`<br>J70<br> `0.8167 / —`<br>J50 <br>`0.7405 / —` | PSNR<br> `37.74`<br>SSIM<br> `0.9824`<br>LPIPS<br> `0.0214` |
| <a href="Test/Cover/v2-de6bae99061d02d212a5a22931ebf730_r.jpg"><img src="Test/Cover/v2-de6bae99061d02d212a5a22931ebf730_r.jpg" width="320" alt="v2-de6b Cover"></a><br><sub>v2-de6b…</sub> | <a href="Test/Stego/v2-de6bae99061d02d212a5a22931ebf730_r_stego.png"><img src="Test/Stego/v2-de6bae99061d02d212a5a22931ebf730_r_stego.png" width="320" alt="v2-de6b Stego"></a><br><sub>v2-de6b… stego</sub> | `1776×1152` | Payload <br>`2800 bit`<br>碼流<br> `5101 bit`<br>上限 <br>`13455 bit` | PNG <br>`0.9996 / 1.0000`<br>J90 <br>`0.9994 / 1.0000`<br>J70 <br>`0.9978 / 1.0000`<br>J50 <br>`0.9941 / 0.9989` | PSNR<br> `30.02`<br>SSIM `0.9339`<br>LPIPS <br>`0.0593` |

`Origin/ECC` 表示原始 bit 準確率 / ECC 後 Payload bit 準確率；實際碼流包含 201-bit 重復頭部和 Hamming(7,4) 冗余；最後一張 LPIPS 為 0.0593，高於約 0.05 的推薦目標，因此仍需以實際情況結果判斷不可察覺性。

## 運行測試

單元測試和偽模型集成測試：

```powershell
conda run -n base python -m pytest PipeLine/tests -q
```

純算法測試不會加載真實 VAE 權重。當前單元測試結果為 **41 passed**。完整批量測試會加載原生 VAE、改造編碼器和 LPIPS 網絡，並需要對應權重、穩定性策略以及足夠的內存/顯存：

```powershell
conda run -n base python -m PipeLine.cli test
```

建議在提交新的參數組合前保存以下信息：FP32 精度、模型權重版本、配置文件、穩定圖版本、測試圖片清單、隨機種子、逐圖容量、PNG/JPEG 提取結果和三項質量指標。

## 常見問題

### 提示圖像尺寸被調整

這是預處理的預期行為。warning 會給出調整前後尺寸。若不希望改變寬高比，使用 `scale_crop`；若不希望裁剪邊緣，使用 `stretch`，並接受輕微幾何拉伸。

### `strict` 模式提示缺少穩定圖

為目標 profile 生成嚴格對應的 `stability_<寬>x<高>.npz`，或者將策略切換為 `normalized_interpolation` / `analytic_band`。切換策略後必須重新評測。

### 提示 Payload 超過容量

最終容量同時受 `0.005 bit/pixel` 預算、重復頭部、ECC 開銷和實際 DCT 位置數約束。可縮短秘密、關閉 ECC（會降低容錯能力）、提高位置容量，或使用更大載體。不要僅為了容量盲目提高嵌入強度。

### PNG 可提取，但 JPEG 提取失敗

先確認圖像沒有被縮放或裁剪，再檢查 JPEG 等級和色度子采樣。可在圖像質量允許范圍內小步提高 `base_alpha`，使用更可靠的穩定圖/閾值，並保持 ECC 開啟。每次改動都需重新檢查 LPIPS 指標以確保圖像質量。

### LPIPS 超過 0.05 或出現細微模糊

減少 Payload 使用率或 `positions_per_block`，適當提高 `stability_threshold`，或降低 `base_alpha`。不同圖像對失真的敏感程度不同，應以逐圖結果為準。

### 更換 FP16、BF16 或 FP8 後準確率下降

低精度量化誤差會影響 VAE 的圖像到 latent 之間的雙向轉換和 DCT 判決鏈路。恢復 FP32 是當前唯一受項目評測保證的做法；若必須低精度部署，需要針對目標精度重新校準並完整測試，不應引用本項目 FP32 成績作為依據。

## 使用邊界

- 隱寫並不等於加密，秘密內容的機密性應由獨立密碼學方案保證。
- 有損壓縮、二次 VAE 編解碼、縮放、裁剪、旋轉、濾鏡和平臺轉碼都可能破壞 Payload。
- `normalized_interpolation` 和 `analytic_band` 是缺少嚴格穩定圖時的可行策略，不代表與真實統計穩定圖具有相同性能。
- 任何模型、權重、精度、配置或預處理變化，都可能使既有穩定圖和評測結果失效。
- 正式部署應使用自有數據分布進行 FP32 基準測試，並以 CRC 成功、文本完全匹配和逐圖 LPIPS 為驗收依據。
