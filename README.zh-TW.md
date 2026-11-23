# Meto 繁體中文

[English](README.md) · **繁體中文**

在自己的 Mac 上學習 Apple GPU 計算。Meto 依上游編號順序收錄 LeetGPU 的 **147 道題目**：每道題目都提供樸素實作（baseline）、完整解答（solution）與可編輯的練習複本，用題目自帶的 CPU 參考實作驗證 GPU 結果，並在本機量出可重現的耗時。全程離線執行，不需要雲端、帳號或網路。

ID 97、100、134–139 是上游編號空缺；07–155 當中有 61 課附手寫 shader。

## 快速開始

```sh
git clone --recurse-submodules <倉庫位址>
cd meto
./meto setup     # 拉取題目子模組、建立本機 Python 環境、自我檢查
```

已經在本目錄內，就先 `git submodule update --init`，再 `./meto setup`。

用 VS Code 開啟本目錄，按 **⌘⇧B**，輸入 `01` 後按 Enter；或直接在終端機執行 `./meto check 01`。這一條命令會建置 Swift 宿主、在執行期編譯 shader、把 kernel 派發到 GPU，並用 5 種輸入長度對照 CPU 計算。長度 257 會讓最後一個 threadgroup 只用到大半，正好暴露漏掉的尾端處理。

動手改一遍最能建立手感：`./meto practice 01` 產生 `lessons/01-vector-add/practice.metal`，把加法改成減法，再跑 `./meto check 01 practice`——第一個不相符的索引就是執行器替你定位到的錯誤。改回加法後用 `./meto bench 01` 與內建版本比較。`practice.metal` 與 `practice.py` 是你的檔案，可以放心提交；重複執行 `practice` 不會覆蓋改動。

## 環境需求

| 需要什麼 | 用途 | 準備方式 |
|---|---|---|
| 帶 Metal 裝置的 Apple silicon Mac | 所有驗證都跑在本機 GPU 上 | `./meto doctor` 會印出裝置名稱 |
| Apple 命令列工具 | Swift 宿主與執行期 shader 編譯 | `xcode-select --install`（當 `xcrun --find swiftc` 失敗時） |
| VS Code | 把命令包裝成 Run Task | 開啟本目錄，Terminal → Run Task |
| `uv` 或 Python 3.13 | 07–155 軌道的 PyTorch MPS 後端 | `brew install uv`；沒有 uv 時改用 `PYTHON=/path/to/python3.13 ./meto setup` |

`setup` 把依賴裝進專案內部的 `.build/catalog-env`（PyTorch 2.15 + NumPy），VS Code 已設定為自動使用這個直譯器；裝好之後，做題、驗證與計時全程離線。

## 日常工作流

| 想做的事 | 命令 | 得到什麼 |
|---|---|---|
| 瀏覽課程 | `./meto list` | 全部 ID 與標題 |
| 讀題 | `./meto show 04` | 題目敘述，以及範例與實作檔案的位置 |
| 開始動手 | `./meto practice 04` | 課目錄旁的 `practice.metal` / `practice.py` |
| 驗證自己的版本 | `./meto check 04 practice` | 第一個不相符的索引，含期望值與實際值 |
| 驗證內建解答 | `./meto check 04` | 範例加上全部功能性測試案例 |
| 計時 | `./meto bench 04` | 先驗證、再暖機，接著印出中位數 |
| 全課快速檢查 | `./meto smoke all` | 原生測試案例 + 每課一個範例 |
| 全課正確性 | `./meto check all` | 原生軌道 + catalog 軌道 |
| 全課計時 | `./meto bench all` | 每課一行結果 |
| 上游效能規模 | `./meto bench 17 --performance` | 題目宣告的效能形狀（部分 transformer 輸入達 GB 級，先看題目） |

## 兩條學習軌道

**01–06 · 原生 Metal。** 每課自行定義形狀、CPU 參考演算法與派發幾何；共用的 Swift 執行期負責編譯 shader、綁定 buffer、派發 threadgroup、等待完成並比對輸出。先把一條派發從 Swift 追到 GPU，再往後走。[Metal 執行期](Sources/MetalRuntime.swift) · [前六課詳解](docs/first-six.md)

**07–155 · catalog。** 每課是一個可獨立編輯的 `solution.py`，收到的張量已經在 GPU 上。61 課透過 `torch.mps.compile_shader` 派發手寫 Metal，其餘用 PyTorch 的 MPS 運算子組出 GEMM、FFT、attention。想看執行緒對應、記憶體存取與同步，打開同目錄的 `.metal` 檔案。[catalog 執行器](Tools/catalog.py) · [shader 載入助手](catalog/metal_shader.py)

## 驗證與計時

- 輸入由鎖定的上游測試案例產生；參考實作在 CPU 上執行，你的實作在互不干擾的 GPU buffer 上執行。
- 輸出 buffer 在派發前先填入毒值（浮點 `NaN`、整數 `11`），漏寫不會被零初始化矇混過去。
- 題目宣告的每個輸出都會比對，包括被就地修改的輸入；整數要求完全相等，浮點使用題目自己的 `atol` / `rtol`。
- catalog 軌道遇到 `PYTORCH_ENABLE_MPS_FALLBACK=1` 會直接結束，避免悄悄退回 CPU 卻假裝通過。
- 原生軌道回報同一形狀暖機後 15 次取樣的 **GPU command buffer 時間**中位數；catalog 軌道回報**同步後的牆鐘耗時**中位數，包含 Python 派發與 solution 內部的記憶體配置。
- 預設挑選張量參數不超過 32 MiB 的最大測試案例，全部超過上限時取最小的一個；每次執行都會印出案例名稱與張量形狀。
- CPU 驗證、資料搬移、shader 編譯都發生在計時區間之外，就地修改的輸入每輪之間還原。
- 晶片世代、散熱狀態、矩陣尺寸、派發開銷都會改變勝負；用文件裡的數字挑選實驗，再於自己的負載上重新量測。

## 倉庫結構

```text
Meto/
├── meto            統一入口，所有命令都從這裡走
├── Makefile         Swift 宿主的建置規則
├── Sources/         GPU 裝置、buffer、pipeline 快取、派發與比對
├── lessons/         01–06：baseline.metal / solution.metal + Swift 派發
├── catalog/         07–155：solution.py，61 課另有 solution.metal
├── Tools/           課程索引與 CPU/GPU 驗證執行器
├── docs/            課程表、前六課詳解、三份基準報告
└── reference/       上游題目子模組（鎖定版本）
```

## 延伸文件

| 內容 | 位置 |
|---|---|
| 147 課的 ID、難度、實作連結與 M3 實測耗時 | [curriculum.md](docs/curriculum.md) |
| 一條派發從 Swift 到 GPU 的完整追蹤，逐課提問 | [first-six.md](docs/first-six.md) |
| 實測資料與調校說明 | [07–50](docs/benchmarks-07-50.md) · [51–101](docs/benchmarks-51-101.md) · [102–155](docs/benchmarks-102-155.md) |
| 全部命令與參數 | `./meto help` |

## 上游題目與授權條款

題目敘述、範例與 CPU 參考實作來自未修改的 Git 子模組 `reference/leetgpu-challenges`，鎖定在 commit `3b2ec31`，授權條款隨子模組一起取得。執行器、說明與解答都在本倉庫中。[AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) · [上游授權條款](reference/leetgpu-challenges/LICENSE)
