# MeetTranslate

會議室即時多語翻譯。一台專用電腦以旁聽身分加入 Teams 會議，把發言即時轉錄、翻譯，投到會議室電視上；會後產出依發言者分段的多語對照逐字稿。

預設語言組合為繁體中文（台灣）、越南語、英語，可改為任意 2 至 3 種。

## 運作方式

這台電腦**不外放、不開麥克風**，純粹旁聽。所有與會者的聲音從同一條音訊進來，因此語者分離不只是為了標示「誰在講」——它同時決定每句話要用哪種語言辨識、翻成哪些語言。

```
虛擬音效裝置 → VAD 切句 → 語者嵌入 → 聚類 → 該語者的主導語言
   │                                              │
   │                              Whisper 辨識（指定語言，不自動偵測）
   │                                              │
   │                                    翻譯成其餘各語言
   │                                              │
   │                        暫定字幕（秒出）──→ 下一句翻譯時順帶修飾
   │                                              │
   │                                   字幕頁（新增 / 就地改寫）→ 電視
   └→ session_{時間}.wav（會後重跑的唯一來源）
```

順序不可對調。Whisper 的語言在建立辨識器時就固定，判錯不會優雅降級而是崩塌成重複填充詞，所以必須先確定語者才能決定語言。語者嵌入只看聲學特徵、不需要先轉錄，放在前面不增加延遲。

設計取捨與被推翻過的方案記在 [plan.md](plan.md)。

## 需求

- Python 3.12+
- Node.js 20+
- 虛擬音效裝置：Windows 用 [VB-Cable](https://vb-audio.com/Cable/)，macOS 用 [BlackHole](https://existential.audio/blackhole/)
- 約 2 GB 磁碟空間放模型
- 翻譯需要 Anthropic API 金鑰（沒有也能跑，只轉錄不翻譯）

## 安裝

### 1. 音訊路徑（最容易出錯的一步）

把 Teams 的**音訊輸出**指向虛擬音效裝置，程式再從該裝置收音。

不要改用系統靜音來達成「不外放」——Windows 的播放裝置一旦靜音，擷取到的就是一片無聲，而且沒有任何錯誤訊息，只會得到空白的逐字稿。收音頁的峰值指示就是為了讓你當場發現這件事。

### 2. 下載模型

模型不在版控裡。在專案根目錄執行（Windows 用 Git Bash；`curl` 與 `tar` 在 Windows 10 以上已內建）：

```bash
mkdir -p models && cd models
curl -L -o silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -L -o speaker_embedding.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
curl -L -o whisper-small.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2
tar -xjf whisper-small.tar.bz2 && rm whisper-small.tar.bz2
```

`models/` 最終應包含 `silero_vad.onnx`、`speaker_embedding.onnx`、`sherpa-onnx-whisper-small/`。

需要更高準確度可另外下載 `sherpa-onnx-whisper-medium` 或 `large-v3`，程式會自動偵測；會後處理一律使用磁碟上最大的那個。

### 3. 啟動

```bash
start.bat
```

macOS 用 `./start.command`。腳本會清掉佔用 port 的舊程序、建立虛擬環境、安裝套件、建置前端，然後開啟瀏覽器。首次執行需要幾分鐘。

## 使用

啟動後瀏覽器停在 <http://127.0.0.1:8000>。

| 頁面 | 用途 |
|---|---|
| 收音 | 選擇輸入裝置、開始／停止、即時音量與狀態 |
| 會議紀錄 | 歷次逐字稿；在此把 `S1`／`S2` 對應成真實姓名 |
| 詞彙表 | 專有名詞處理方式 |
| 設定 → 字幕顯示 | 語言組合與電視排版 |
| 設定 → LLM | 翻譯模型與供應商 |
| 設定 → LLM 金鑰 | 多組金鑰輪替 |

**字幕投影**：在設定→字幕顯示點「開啟字幕頁」，把該視窗拖到電視那個顯示器，按 F11 全螢幕。

**開會流程**：Teams 加入會議 → 收音頁按「開始」→ 確認峰值有跳動 → 開會 → 結束後按「停止」。

### 詞彙表的三種處理方式

| 方式 | 用途 |
|---|---|
| 指定譯詞 | 強制使用你填的翻譯。公司、產品、部門名稱 |
| 保留原文 | 完全不翻譯。跨國團隊共通的英文詞如 `schedule`、`delay`，硬翻反而更難讀 |
| 僅作辨識提示 | 只提高語音辨識準確度，不介入翻譯 |

### 語者與語言

程式以聲紋分辨發言者，但取不到 Teams 的參與者名單，所以先標 `S1`、`S2`。在會議紀錄頁對應一次姓名即可。

每位語者的語言由累積統計決定，需要連續數句不符才會切換。中文與英語之間的門檻特別高——台灣的中文經常夾雜英文詞（「這個 schedule 要 delay 一週」），一句話裡幾個英文字不該讓程式判定他改講英語了。若某位固定講某種語言，可在設定中釘死。

中文輸出一律轉為台灣繁體。Whisper 對中文一律輸出簡體，不轉換的話電視上會跳出簡體字。

## 設定與資料

執行期產生的檔案都不進版控：

| 檔案 | 內容 |
|---|---|
| `config.json` | 語言、輸入裝置、模型、顯示格式 |
| `llm.json` | LLM 供應商設定 |
| `llm_keys.json` | 輪替用的 API 金鑰 |
| `meettranslate.db` | 詞彙表、會議紀錄、逐字稿 |
| `recordings/` | 原始錄音 |

環境變數可覆寫：`ANTHROPIC_API_KEY`、`MEETTRANSLATE_INPUT_DEVICE`、`MEETTRANSLATE_LANGUAGES`、`MEETTRANSLATE_WHISPER_MODEL`。

## 開發

```bash
.venv\Scripts\python.exe -m server.test_audio      # 裝置解析與設定
.venv\Scripts\python.exe -m server.test_pipeline   # 辨識與語者判斷邏輯
.venv\Scripts\python.exe -m server.test_e2e        # HTTP API 與完整管線
```

前端另外開一個 dev server（會透過 `VITE_API_URL` 連到後端）：

```bash
cd dashboard && npm run dev
```

## 效能

Whisper small、int8、4 執行緒，20 邏輯核心：

| 指標 | 數值 |
|---|---|
| Realtime factor | 0.20 |
| 一般短句上字幕 | 約 2 秒 |
| 16 秒長句上字幕 | 4.8 秒（含模型首次載入） |

CPU 足夠，不需要顯示卡。有 NVIDIA 顯卡或 Apple Silicon 會自動使用。

## 已知限制

- **越南語與台灣國語的辨識率尚未驗證**。開發過程只有英語測試音，這是目前最大的未知
- **僅在 Windows 實測過**。macOS 路徑已寫好但未在實機驗證
- 遠端與會者看不到會議室電視。字幕只服務現場
- 兩人同時講話時，切出的片段混有兩人聲音，語者分離會不穩
- 會議室發言經 Teams 壓縮與降噪後才進來，辨識率低於直接收音
