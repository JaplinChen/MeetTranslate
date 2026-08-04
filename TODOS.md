# TODOS

由 `/autoplan`（2026-08-04, commit bf2f461）產出的延後項目。

## T4 — ASR 後端可插拔 + 雲端 fallback

抽 `Transcriber` protocol（`transcribe` 必要 / `transcribe_many` 選配），
新增 OpenAI 相容的 `server/asr_cloud.py` 作為無 GPU 時的 fallback。

**延後理由**：目前是固定機房 PC + RTX 5060 Ti，「GPU 缺席」情境不存在；
`asr_gpu.maybe()` 已在 GPU 不可用時退回 `asr.Transcriber`
（`server/asr_gpu.py:160-175`）。`Transcriber` 介面實際上已經隱性一致
（`server/asr_gpu.py:68` 註解自陳「Same surface as asr.Transcriber」）。

**觸發條件**（滿足任一再做）：
- 更換或增購不含 NVIDIA GPU 的機器
- 需要在分公司部署第二台
- 本地模型品質不再足夠，需要雲端模型

**先決條件**：評估把公司內部會議音訊送上第三方 ASR 的合規問題。
本輪審查未做此評估。

## E5 — hotwords 依 category / 近期使用排序

`asr_gpu.hotwords_from()`（`server/asr_gpu.py:178`）目前無上限串接整份
glossary。本輪只加長度上限（截斷）。若日後出現「glossary 長到截斷會切掉
重要術語」的實例，再加排序啟發式。

**觸發條件**：實際觀察到熱詞被截斷且影響辨識品質。

## 已知但未排程

- `server/main.py` 完全沒有驗證。單機 localhost 情境下可接受，
  若日後暴露到內網需重新評估（審查發現 E6）。
- forced-language 架構：`is_degenerate` / `dominant_languages` /
  `MIN_LANGUAGE_EVIDENCE` 都是「強制錯語言會塌成復讀」的補丁。
  替代方案是一律 auto-detect + 事後多數決，live 端語言先驗只用於顯示
  而非解碼約束。本輪未評估，屬獨立議題。
