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

用既有錄影檔測辨識率，不必接虛擬音效裝置：

```bash
ffmpeg -i meeting.mp4 -ac 1 -ar 16000 -c:a pcm_s16le recordings/test01.wav
.venv\Scripts\python.exe -m scripts.bench_wav recordings/test01.wav --ref ref.txt
```

輸出分段逐字稿、realtime factor、每位語者的主導語言，以及詞彙表裡哪些詞被辨識出來。`--ref` 給一份人工聽打的參考逐字稿就會算 CER（中文沒有詞邊界，用字錯誤率而非 WER）。`--model medium` 可比較不同模型層級。

### 同音字修正

公司與產品名稱常常音對字錯。兩層處理：

1. **解碼後修正**（`server/correct.py`，一律啟用）— 把解碼結果和詞彙表逐詞比對去聲調拼音，發音完全相同就換成詞彙表的寫法。`公單 → 工單`、`微剛科技 → 威剛科技`、`生館 → 生管`。英文與越南語詞容許 25% 編輯距離（`incent → Vincent`），中文不容許：中文音節太密集，容許一個編輯距離會把「知道」改成「製造」
2. **解碼時偏置** — GPU 路徑用 faster-whisper 原生 hotwords 把詞彙表傳進模型。sherpa-onnx 的 Whisper 沒有這個能力（contextual biasing 只支援 transducer），CPU 路徑改用下面的同音字替換器

同音字替換器只在 CPU 路徑有意義，且僅適用中文。

```bash
mkdir -p models/hr && cd models/hr
curl -L -o dict.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar -xjf dict.tar.bz2 && rm dict.tar.bz2
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
cd ../.. && .venv\Scripts\python.exe -m scripts.build_hr
```

`models/hr/` 三個檔（`dict/`、`lexicon.txt`、`replace.fst`）齊全後，即時管線與會後重跑都會自動套用；缺任一個就整組略過，不會半套。詞彙表改了要重跑 `build_hr.py`。

只對中文辨識器套用。替換器會把文字轉成拼音再轉回來，拉丁字母的空白會在這趟來回中消失——實測 `You gotta take those from them` 會變成一個字。自動偵測的辨識器同理排除。

產生 `replace.fst` 需要 pynini，而 pynini 沒有 Windows wheel。在 Windows 上 `build_hr.py` 只會寫出 `replace.txt`（拼音對照表，可人工檢查），`.fst` 用 Docker 產生：

```bash
docker run --rm -v "D:/Works/MeetTranslate/models/hr:/w" -w /w python:3.12-slim bash -c \
  "pip install -q --only-binary :all: pynini && python -c \"
import pynini
from pynini.lib import utf8
rules = [l.rstrip().split('\t') for l in open('replace.txt', encoding='utf-8') if l.strip()]
c = None
for p, t in rules:
    o = pynini.cross(p, t); c = o if c is None else (c | o)
pynini.cdrewrite(c.optimize(), '', '', utf8.VALID_UTF8_CHAR.star).write('replace.fst')\""
```

前端另外開一個 dev server（會透過 `VITE_API_URL` 連到後端）：

```bash
cd dashboard && npm run dev
```

## 效能

量測環境：20 邏輯核心 + RTX 5060 Ti 16 GB。

| 路徑 | 模型 | Realtime factor | 機器可用性 |
|---|---|---|---|
| GPU（有顯卡時自動使用） | large-v3 float16 | 0.15 – 0.30 | CPU 約 12%，可正常工作 |
| CPU | small int8、4 執行緒 | 0.20 | 尚可 |
| CPU | small float32、全部核心 | 0.57 | 整台機器卡死 |

會後重跑七場訪談共 9.5 小時音訊，GPU 上 1 小時 30 分完成。同樣的量在 CPU 上約需 8.5 小時，且期間無法使用電腦——這是 `--threads` 預設只取一半核心的原因。

即時字幕延遲：一般短句約 2 秒，16 秒長句 4.8 秒（含模型首次載入）。

沒有顯卡也能跑，退回 sherpa-onnx CPU 路徑；設 `MEETTRANSLATE_NO_GPU=1` 可強制不用 GPU。

### GPU 安裝

```bash
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

CUDA runtime 來自 pip wheel，不需要另外裝 CUDA Toolkit。兩個踩過的坑：

- **Blackwell（RTX 50 系列，sm_120）需要 CTranslate2 4.8 以上**。多數教學釘的 CUDA 12.1 版本不涵蓋這代顯卡
- CTranslate2 透過 **PATH** 尋找 cuBLAS 與 cuDNN，`os.add_dll_directory()` 無效——模型會載入成功，第一次 encode 才報 `cublas64_12.dll is not found`。`server/asr_gpu.py` 在 import 時處理這件事

## 已知限制

- **越南語的幻覺率明顯高於中文**。large-v3 的越南語訓練資料多來自 YouTube 字幕，遇到聽不清的片段會吐出「請訂閱頻道」這類台詞。七場訪談實測佔越南語句數 14.8%（中文 0.2%、英語 0.1%），已用片語過濾器擋掉，但這代表越南語段落的可信度本來就較低
- **辨識率仍無 CER 數字**。七場訪談已產出逐字稿，但沒有人工聽打的參考稿可比對，只能定性判斷「可讀」
- 同音字修正只對中文有效，且只接受發音完全相同。越南語的專有名詞沒有對應機制
- 詞彙表放常見雙字詞有風險。`採購` 會把「才夠」改掉，`料號` 會把「料耗」改掉——同音本身就是歧義。適合放的是公司、產品、模組名稱這類獨特詞
- **僅在 Windows 實測過**。macOS 路徑已寫好但未在實機驗證
- 遠端與會者看不到會議室電視。字幕只服務現場
- 兩人同時講話時，切出的片段混有兩人聲音，語者分離會不穩
- 會議室發言經 Teams 壓縮與降噪後才進來，辨識率低於直接收音

## 授權與致謝

MIT，見 [LICENSE](LICENSE)。第三方授權聲明見 [NOTICE](NOTICE)。

管理介面衍生自 [OpenWA-Lab](https://github.com/JaplinChen/OpenWA-Lab)（MIT）。整體架構參考 [meetily](https://github.com/Zackriya-Solutions/meetily)，但改為本機服務加瀏覽器，未採用其 Tauri 外殼。

語音處理使用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：Silero VAD、Whisper、3D-Speaker 語者嵌入。簡繁轉換使用 [OpenCC](https://github.com/BYVoid/OpenCC)。
