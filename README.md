# Jstris Replay Decoder & Full Match State Reconstructor (`jstris-replay-tool`)

一套零外部相依性（純 Python 標準函式庫）、100% 對齊 Jstris 官方 `Replayer` / `GameCore` 實作的 **Jstris Replay Code 完整解碼與逐次落子對局還原工具**。

本工具將 Jstris Replay Code（`N4Ig...` URI-safe LZ-String 或 JSON）完整解碼並透過精確物理引擎模擬，輸出可供 Tetris AI / Coding Agent 直接讀取的結構化資料（`metadata.json`、`events.json`、`placements.json`、`states.json`、`summary.json`）。

---

## 一、核心技術發現與對 `rrbutani/jstris-replay` 的修正

在逆向與對照 Jstris 官方前端程式碼（`replayer.js` 與 `game.js`）後，我們發現參考專案 `rrbutani/jstris-replay` 有多處假設錯誤或未實作細節，若直接使用會導致解碼崩潰與狀態錯誤：

1. **V3+ 非定長 Bitstream 與 32-bit Word 端序反轉 (`_base64ToArrayBuffer` + `pullBits`)**：
   - `rrbutani/jstris-replay` 誤將事件視為固定 16-bit（2 bytes）陣列。
   - 實際上 Jstris V3+ 將 Base64 解碼後的每 4 bytes 進行反轉後建立 `Uint32Array`（等效於以 **Big-Endian `uint32`** 讀取每 4 bytes），並透過串流 `pullBits(n)` 逐位元讀取：
     - 每個事件開頭為 **12-bit 時間戳**（若讀到 `4095` / `0xFFF` 代表串流結束 Padding）+ **4-bit 動作碼 (`a`)**。
     - 若 `a == 11 (GARBAGE_ADD)`，緊接著讀取 **5-bit `lines`** 與 **4-bit `hole_column`**（共 25 bits，非 16 bits！）。
     - 若 `a == 13 (REDBAR_SET)`，緊接著讀取 **5-bit**。
     - 若 `a == 14 (ARR_MOVE)`，緊接著讀取 **1-bit**。
     - 若 `a == 15 (AUX)`，緊接著讀取 **4-bit `aux` 子動作碼** 及對應長度的 payload（`AFK`: 16b、`BLOCK_SET`: 1+4b、`MOVE_TO`: 4+5b、`RANDOMIZER`: 1+5b、`MATRIX_MOD`: 4+5b、`WIDE_GARBAGE_ADD`: 5+4+3+1b）。
2. **時間戳溢位週期是 `4094` ms 而非 `4096` (`0x1000`) ms**：
   - `rrbutani/jstris-replay` 在 12-bit 時間戳回繞時加上 `0x1000` (4096)。
   - 實際上 Jstris 編碼端設定週期常數為 `4094`（保留 `4095` 作為結束標記），解碼公式為：當 `raw_ts < prev_ts` 時 `wrap_count += 1`，`t = 4094 * wrap_count + raw_ts`。
   - 以本場 100L Cheese Race Replay 為例，共發生 11 次回繞，最後一個事件 `11 * 4094 + 2661 = 47695 ms`，與 `gameEnd - gameStart = 47695 ms` **完全吻合（誤差 0 ms）**；若用 4096 會產生 +22 ms 漂移。
3. **複合遊戲模式 `m = 196611` 解碼**：
   - Jstris 以 `pmode = (m & 0xFFFF0000) >> 16`、`subMode = m & 0xFFFF` 編碼模式。
   - `196611 = 0x00030003` -> `pmode = 3` (**Cheese Race**)，`subMode = 3` (**100L**)，初始目標 `linesRemaining = 100`。
4. **首袋起手規則版本差異 (`v >= 3.3` vs `v < 3.3`)**：
   - `rrbutani/jstris-replay` 對所有模式強制置換首袋開頭的 `S/Z`。
   - 實際上 Jstris `GameCore.prototype.generateQueue` 規定：
     - `pmode == 1` (Sprint)：若首塊為 `S` 或 `Z` (`id >= 5`)，與第 2 或第 3 塊交換。
     - `pmode != 1`（如 Cheese Race `pmode == 3`）：僅在 **`v < 3.3`** 且前兩塊皆為 `S/Z` 時才將第 1 塊與第 3 塊交換；在 **`v >= 3.3`**（本 Replay 為 `v = 3.3`）時**不進行交換**（但線上產生練習局 seed 時 `isBannedStartSequence` 會重抽直到前兩塊不全為 `S/Z`，本局首塊即為 `Z`、第二塊為 `I`）。
5. **Cheese Race 垃圾行上升與進度計算機制**：
   - 開局 `t = 0` 時，Jstris 先執行 `getNextBlock()` 取出第 1 塊 (`Z`)，再執行 9 個 `GARBAGE_ADD(1, col)` 推入初始 9 行起司垃圾行。
   - 每次落子鎖定後，若未消行或消行後垃圾高度低於下限（`minCheeseHeight = 3` / 未消行補至 `maxCheeseHeight = 9`），會在 Replay 中緊接著寫入相同時間戳的 `GARBAGE_ADD` 事件，且垃圾行推升時若壓到當前剛生成的方塊，`garbage_add_adv` 會自動將當前方塊往上推 (`activeBlock.pos.y--`) 並更新幽靈方塊位置。

---

## 二、安裝與使用方式

### 1. 直接以 CLI 執行

```bash
cd /home/antigravity/jstris-replay-tool

# 從檔案讀取 Replay Code 並輸出結構化 JSON 至 ./output
python3 -m jstris_replay -i data/sample_100l_cheese_replay.txt -o output

# 執行第二份獨立驗證樣本 (seed='cktcp')
python3 -m jstris_replay -i data/sample_100l_cheese_replay_2.txt -o output_2

# 或直接傳入 Replay Code 字串
python3 -m jstris_replay -c "N4IgxiBcoG5QzAOngGhAZw..." -o output
```

### 2. 以 Python API 呼叫

```python
from jstris_replay import decode_replay_code, JstrisSimulator, export_all_json

decoded = decode_replay_code(replay_code_str)
sim = JstrisSimulator(decoded)
sim.run()

# 匯出 metadata.json, events.json, placements.json, states.json, summary.json
export_all_json(decoded, sim, "./output")
```

### 3. 執行自動化測試

```bash
python3 -m unittest discover -v -s tests
```

---

## 三、輸出檔案結構說明 (`output/`)

| 檔案名稱 | 用途與內容 |
| :--- | :--- |
| `metadata.json` | 完整解析後的 Metadata（版本 `3.3`、模式 `196611` -> `Cheese Race 100L`、Seed、起訖時間、Soft Drop `Instant (steps=20)`、DAS `75ms`、Ruleset 等）。 |
| `events.json` | 全部二進位操作事件的時間序列（包含 `index`、`timestamp_ms`、`raw_12bit_ts`、`epoch_wrap_count`、`action_code`、`action_name`、`data`、`bit_offset`、`bit_length`、`hard_drop_id`）。 |
| `placements.json` | 全部方塊鎖定（落子）的完整紀錄：包含落子序號、當前方塊、Hold 狀態、可見 5 預覽、落子前盤面、最終鎖定 `(x, y, rot)` 與佔用座標、消行前/消行後/垃圾補充後三階段盤面、消除行數、消除垃圾行數、剩餘目標行數、攻擊量、Finesse、該手完整按鍵序列。 |
| `states.json` | 專為 Tetris AI 決策分析設計的合法決策觀測快照（`observable_state` 嚴格只包含該決策當下合法可見的 20x10 盤面、Active 方塊、Hold 方塊、5 個可見 Previews 與進度資訊，絕不洩漏未來牌序或未知垃圾孔位）以及該手實際採取的 `chosen_action`。 |
| `summary.json` | 對局完整統計、最終盤面、多維度正確性驗證報告（`VERIFIED` 檢查合法落點、`0xFFF` 終止符、事件完整處理、時間戳單調性與對局完成條件），以及嚴格區分的「合法可觀測狀態 vs. Replay 事後重建完整序列 vs. 無法僅由 Replay 還原之資訊」。 |

---

## 四、內建雙樣本驗證基準 (`data/`)

| 樣本檔案 | 遊戲模式 | Seed | 遊戲時間 | 事件總數 | 放置方塊數 | 總消行 / 垃圾行消除 | 分數 / 攻擊 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/sample_100l_cheese_replay.txt` | Cheese Race 100L (`m=196611`) | `zglrnd` | `47.695 s` (`47695 ms`) | `864` | `162` | `154` 行 / `100` 垃圾行 | `34004` pts / `106` atk |
| `data/sample_100l_cheese_replay_2.txt` | Cheese Race 100L (`m=196611`) | `cktcp` | `48.838 s` (`48838 ms`) | `875` | `167` | `156` 行 / `100` 垃圾行 | `33834` pts / `104` atk |
