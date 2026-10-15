# DIC-2 Kaggle 50 Startups & Boston Housing 多元線性迴歸分析平台

> **國立中興大學 (NCHU) · 物聯網數據分析與應用 (IoT Data Analysis and Applications)**  
> **作業項目**：DIC-2 kaggle 50 startup and boston housing  
> **作業重點**：多元線性迴歸 (Multiple Linear Regression, MLR)、最小平方法 (OLS)、特徵篩選、殘差診斷與 IoT 邊緣端即時推論。

---

## 🌟 作業要求與加分項完整實作對照 (Checklist)

| 評分項目 | 項目說明 | 達成狀況 | 平台實作位置 |
| :--- | :--- | :---: | :--- |
| **基本要求** | 支援 50 Startups 與 Boston Housing 兩種問題分析 | 💯 **100% 達成** | 頂端選單及預設按鈕，可即時切換 |
| **加積分 1 分** | 寫下拉式選單換兩種 problem | 🏆 **完成 (+1分)** | 頂部導航欄中央大型「選擇問題」下拉選單 |
| **加積分 2 分** | 可任意輸入 Dataset (支援自訂 URL 與本地 CSV 上傳) | 🏆 **完成 (+2分)** | 次導航欄「自訂資料集 URL」與「本地上傳 CSV」 |
| **GitHub Pages** | 支援自動化部署工作流程 `.github/workflows/pages.yml` | 🚀 **完成** | 隨附標準 GitHub Actions 官方自動部署檔 |

---

## 🎯 核心功能特色

### 1. 雙經典 Kaggle 資料集無縫切換
* **🏢 50 Startups Dataset**:
  * 預測標的：新創公司獲利 (`Profit`)
  * 自變數：研發支出 (`R&D Spend`)、行政支出 (`Administration`)、行銷支出 (`Marketing Spend`)、所屬州別 (`State` 類別型變數：New York、California、Florida，自動進行 One-Hot Encoding)。
* **🏡 Boston Housing Dataset**:
  * 預測標的：波士頓房屋價值中位數 (`MEDV` in $1000's)
  * 自變數：13 項環境與人口特徵 (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`)。

### 2. 即時特徵篩選 (Feature Selection & Preliminary Analysis)
* 每一項特徵均以動態 Chip 呈現，附帶與目標變數之皮爾森相關係數 $r$。
* 點擊任意特徵 Chip 即可動態納入或排除，整個 OLS 迴歸模型於毫秒內即時重新擬合，指標與圖表即時更新。

### 3. 作業指定雙評估曲線 (Stepwise Dual Curves)
精準重現作業截圖所要求的兩項重要模型診斷圖表：
* **📉 RMSE by Number of Features**：呈現隨著特徵數量增加，模型均方根誤差的收斂趨勢。
* **📈 Adjusted R² by Number of Features**：呈現懲罰多餘特徵項後的調整決定係數變化，協助找出最佳特徵子集。
* **額外提供**：實際值 vs. 預測值對角散佈圖 (Actual vs Predicted) 與殘差常態分佈長條圖 (Residuals Diagnostic)。

### 4. 完整統計假設檢定 (Model Coefficients & Statistical Tests)
* 最小平方法 (OLS) 估計係數 $\beta$
* 係數標準誤差 (Std. Error)
* 雙尾 $t$ 檢定值與精確 $p$-value 計算 (含顯著性星星標記 `***`, `**`, `*`, `·`)
* 95% 信賴區間 $[\beta - 1.96 \cdot SE, \beta + 1.96 \cdot SE]$
* 整體模型 $F$-Statistic 與總體 $p$-value

### 5. 互動式即時推論器 (Live Real-Time Predictor)
* 針對每一個啟用的特徵自動生成互動式數值滑桿與輸入框。
* 類別變數自動轉化為下拉式選單。
* 拖曳滑桿時，預測值即時跳動計算，並提供 95% 預測區間。

### 6. 測試集與異常值診斷 (Test Set Outlier Diagnostics)
* 自動依設定切分訓練集與獨立測試集 (預設 80% / 20%，可透過滑桿自訂 50% ~ 95%)。
* 測試集表格逐筆比對 真實值、預測值、殘差與誤差百分比。
* 自動標記殘差大於 2 倍標準差之樣本為 **異常值 Outlier (>2σ)**。

---

## 🧮 數學與演算法原理

### 1. 多元線性迴歸正規方程式 (OLS Normal Equation)
模型方程式假設：
$$Y = X\beta + \epsilon$$

其中最小化殘差平方和 (RSS) 的解析解為：
$$\beta = (X^T X)^{-1} X^T Y$$

為防止特徵間出現高度多重共線性 (Multicollinearity) 導致矩陣奇異無法求逆，演算法內建高斯-約當 (Gauss-Jordan) 消去法，並加入 Tikhonov 脊微正則化 ($\lambda = 10^{-7}$)，保證任何特徵組合下反矩陣求解的數值穩定性。

### 2. 模型評估指標
* **判定係數**：
  $$R^2 = 1 - \frac{RSS}{TSS}$$
* **調整後判定係數 (Adjusted R²)**：
  $$R_{adj}^2 = 1 - (1 - R^2) \frac{n - 1}{n - p}$$
* **均方根誤差**：
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
* **平均絕對誤差**：
  $$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

---

## 🚀 部署至 GitHub Pages 步驟指南

本專案根目錄已建立 official GitHub Pages 自動部署工作流 `.github/workflows/pages.yml`。

### 步驟 1: 建立並推播到您的 GitHub Repository
```bash
# 進入專案目錄
cd IoT-DIC2-MultipleLR

# 初始化 git
git init
git add .
git commit -m "feat: complete DIC-2 multiple linear regression project"

# 建立 main 分支並設定遠端儲存庫
git branch -M main
git remote add origin https://github.com/liejefang-source/IoT-DIC2-MultipleLR.git

# 推播至 GitHub
git push -u origin main
```

### 步驟 2: 前往 GitHub 啟用 Pages
1. 開啟您的 GitHub 專案頁面：`https://github.com/liejefang-source/IoT-DIC2-MultipleLR`
2. 點擊頂部 **Settings** &rarr; 左側側邊欄 **Pages**。
3. **方式一 (官方推薦 · 速度最快)：**
   * 在 `Build and deployment` 的 `Source` 下拉選單中選擇 **GitHub Actions**。
   * GitHub 將自動偵測 `.github/workflows/pages.yml`，約 30 秒內自動完成部署上線！
4. **方式二 (傳統分支部署)：**
   * 在 `Source` 選擇 **Deploy from a branch**。
   * Branch 選擇 `main`，資料夾選擇 `/ (root)`，點擊 **Save**。

### 步驟 3: 複製公開網址繳交至 iLearning 3.0
* 部署完成後，公開網址格式如下：
  ```
  https://liejefang-source.github.io/IoT-DIC2-MultipleLR/
  ```
* 將此網址填入作業系統中即可獲得滿分與全部加分！

---

## 📁 檔案結構說明

```
IoT-DIC2-MultipleLR/
├── .github/
│   └── workflows/
│       └── pages.yml          # GitHub Pages 官方自動部署工作流程
├── data/
│   ├── 50_Startups.csv        # 50 Startups 原始 CSV 資料集
│   └── BostonHousing.csv      # Boston Housing 原始 CSV 資料集
├── index.html                 # 主介面 HTML5 架構 (語意化與無障礙設計)
├── style.css                  # 深色毛玻璃現代設計系統 (Vanilla CSS)
├── data.js                    # 內嵌預載資料集 (支援離線直接執行)
├── app.js                     # 線性代數引擎、OLS 擬合、統計檢定與圖表渲染
└── README.md                  # 專案詳細說明文件
```

---
© 國立中興大學 (NCHU) 物聯網數據分析與應用
