"use client";
import { useState, useEffect, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TIER_CONFIG = {
  Gold: { color: "#f59e0b", bg: "rgba(245,158,11,0.15)", icon: "👑" },
  Silver: { color: "#94a3b8", bg: "rgba(148,163,184,0.15)", icon: "🥈" },
  Bronze: { color: "#cd7c3c", bg: "rgba(205,124,60,0.15)", icon: "🥉" },
};
const RISK_CONFIG = {
  HIGH: { color: "#ef4444", label: "Cao" },
  MEDIUM: { color: "#f59e0b", label: "Trung bình" },
  LOW: { color: "#22c55e", label: "Thấp" },
};
const STATE_LABELS = ["Ngủ đông", "Thụ động", "Đang hoạt động", "VIP"];

export default function Home() {
  const [tab, setTab] = useState("predict");

  // ── Predict tab state ──────────────────────────────────────────────
  const [customerId, setCustomerId] = useState("");
  const [customerName, setCustomerName] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [predError, setPredError] = useState("");

  // ── Dataset tab state ──────────────────────────────────────────────
  const [dsInfo, setDsInfo] = useState(null);
  const [dsRows, setDsRows] = useState([]);
  const [dsLoading, setDsLoading] = useState(false);
  const [dsError, setDsError] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const PAGE_SIZE = 15;

  // ── Load dataset ───────────────────────────────────────────────────
  const loadDataset = useCallback(async (p = 1) => {
    setDsLoading(true);
    setDsError("");
    try {
      const [infoRes, rowsRes] = await Promise.all([
        fetch(`${API_URL}/api/dataset/info`),
        fetch(`${API_URL}/api/dataset/rows?page=${p}&page_size=${PAGE_SIZE}`),
      ]);
      const info = await infoRes.json();
      const rows = await rowsRes.json();
      setDsInfo(info);
      setDsRows(rows.data || []);
      setTotal(rows.total || 0);
      setPage(p);
    } catch (e) {
      setDsError("Không thể kết nối BE. Kiểm tra Docker đang chạy.");
    } finally {
      setDsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (tab === "dataset" && dsRows.length === 0) loadDataset(1);
  }, [tab, dsRows.length, loadDataset]);

  // ── Predict ────────────────────────────────────────────────────────
  useEffect(() => {
    const fetchName = async () => {
      if (customerId.length >= 3) {
        try {
          const res = await fetch(`${API_URL}/api/raw-data?search=${customerId}&page_size=1`);
          const data = await res.json();
          if (data.data && data.data.length > 0 && data.data[0].customer_id === customerId) {
            setCustomerName(data.data[0].customer_name);
          }
        } catch (e) {
          console.error("Name lookup failed", e);
        }
      }
    };
    const timer = setTimeout(fetchName, 500);
    return () => clearTimeout(timer);
  }, [customerId]);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!customerId.trim()) return;
    setPredicting(true);
    setPredError("");
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customer_id: customerId.trim(), features: {} }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setPredError(`Lỗi kết nối: ${e.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const visibleCols = dsRows[0]
    ? Object.keys(dsRows[0]).slice(0, 8)
    : [];

  return (
    <div className="container">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-badge">AI · HMM · XGBoost</div>
        <h1>Loyalty AI Dashboard</h1>
        <p>Phân tích Dịch chuyển Hạng thẻ &amp; Dự báo Ngủ đông · Online Retail II</p>
      </header>

      {/* ── Tabs ── */}
      <div className="tab-bar">
        {[
          { key: "predict", label: "🔮 Dự báo rủi ro" },
          { key: "pipeline", label: "⚙️ Quy trình Data Mining" },
          { key: "raw", label: "📂 Dữ liệu thô" },
          { key: "dataset", label: "📊 Dataset Explorer" },
          { key: "overview", label: "ℹ️ Tổng quan hệ thống" },
        ].map((t) => (
          <button
            key={t.key}
            className={`tab-btn ${tab === t.key ? "active" : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════
          TAB 1: PREDICT
      ══════════════════════════════════════════ */}
      {tab === "predict" && (
        <div className="dashboard-grid">
          {/* ── Form ── */}
          <div className="glass-card">
            <h2 className="card-title">Nhập thông tin khách hàng</h2>
            <form onSubmit={handlePredict}>
              <div className="input-group">
                <label htmlFor="customerName">Tên khách hàng (tuỳ chọn)</label>
                <input
                  id="customerName"
                  type="text"
                  className="input-field"
                  value={customerName}
                  onChange={(e) => setCustomerName(e.target.value)}
                  placeholder="Ví dụ: Anh Tuấn"
                />
              </div>
              <div className="input-group">
                <label htmlFor="customerId">Customer ID *</label>
                <input
                  id="customerId"
                  type="text"
                  className="input-field"
                  value={customerId}
                  onChange={(e) => setCustomerId(e.target.value)}
                  placeholder="Ví dụ: 12347.0"
                  required
                />
              </div>
              <p className="hint-text">
                Hệ thống sẽ tra cứu dữ liệu thật từ dataset để dự báo.
              </p>
              <button type="submit" className="btn" disabled={predicting}>
                {predicting ? (
                  <span className="loading-dots">Đang phân tích<span>...</span></span>
                ) : "Phân tích rủi ro →"}
              </button>
              {predError && <p className="error-msg">{predError}</p>}
            </form>
          </div>

          {/* ── Result ── */}
          {result ? (
            <div className="glass-card result-card">
              <h2 className="card-title">
                Kết quả dự báo
                {customerName && <span className="customer-name-tag"> · {customerName}</span>}
              </h2>

              {/* Tier badge */}
              <div
                className="tier-badge"
                style={{
                  background: TIER_CONFIG[result.tier]?.bg,
                  borderColor: TIER_CONFIG[result.tier]?.color,
                  color: TIER_CONFIG[result.tier]?.color,
                }}
              >
                {TIER_CONFIG[result.tier]?.icon} Hạng {result.tier}
              </div>

              {/* Risk level */}
              <div className="risk-row">
                <span>Mức rủi ro:</span>
                <span
                  className="risk-badge"
                  style={{ background: RISK_CONFIG[result.risk_level]?.color }}
                >
                  {RISK_CONFIG[result.risk_level]?.label}
                </span>
              </div>

              {/* Probability bars */}
              <div className="prob-section">
                <ProbBar
                  label="Xác suất Ngủ đông"
                  value={result.dormancy_probability}
                  color="#f43f5e"
                />
                <ProbBar
                  label="Xác suất Rớt hạng"
                  value={result.downgrade_probability}
                  color="#f59e0b"
                />
              </div>

              {/* Meta */}
              <div className="meta-grid">
                <MetaItem label="Customer ID" value={result.customer_id} />
                <MetaItem label="Loyalty Score" value={result.loyalty_score} />
                <MetaItem label="Hidden State" value={`${result.hidden_state} · ${STATE_LABELS[result.hidden_state] || "N/A"}`} />
              </div>

              {/* Action */}
              <div className="action-box">
                <span className="action-label">Hành động đề xuất</span>
                <div className="action-text">{result.recommended_action}</div>
              </div>
            </div>
          ) : (
            <div className="glass-card empty-card">
              <div className="empty-icon">🔍</div>
              <p>Nhập Customer ID và bấm <strong>Phân tích rủi ro</strong> để xem kết quả dự báo từ model thật.</p>
            </div>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════
          TAB 1.5: PIPELINE (QUY TRÌNH DATA MINING)
      ══════════════════════════════════════════ */}
      {tab === "pipeline" && <PipelineTab />}

      {/* ══════════════════════════════════════════
          TAB 2: RAW DATA
      ══════════════════════════════════════════ */}
      {tab === "raw" && <RawDataTab />}

      {/* ══════════════════════════════════════════
          TAB 3: DATASET EXPLORER
      ══════════════════════════════════════════ */}
      {tab === "dataset" && (
        <div className="glass-card full-width-card">
          <div className="dataset-header">
            <h2 className="card-title">📊 Dataset từ HuggingFace</h2>
            <button
              className="btn btn-sm"
              onClick={() => loadDataset(1)}
              disabled={dsLoading}
            >
              {dsLoading ? "Đang sync..." : "🔄 Sync & Load"}
            </button>
          </div>

          {dsInfo && (
            <div className="ds-info-bar">
              <span>📁 <strong>{dsInfo.source}</strong></span>
              <span>📝 <strong>{dsInfo.rows?.toLocaleString()}</strong> bản ghi</span>
              <span>📐 <strong>{dsInfo.columns?.filter(c => c !== "customer_name").length}</strong> cột</span>
              <span className={`ds-status ${dsInfo.status}`}>{dsInfo.status === "loaded" ? "✅ Loaded" : "⏳ Loading"}</span>
            </div>
          )}

          {dsError && <p className="error-msg">{dsError}</p>}

          {dsRows.length > 0 && (
            <>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {visibleCols.filter(c => c !== "customer_name").map((c) => <th key={c}>{c}</th>)}
                      <th>...</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dsRows.map((row, i) => (
                      <tr key={i}>
                        <td className="row-num">{(page - 1) * PAGE_SIZE + i + 1}</td>
                        {visibleCols.filter(c => c !== "customer_name").map((c) => (
                          <td key={c}>
                            {typeof row[c] === "number"
                              ? Number(row[c]).toFixed(2)
                              : String(row[c])}
                          </td>
                        ))}
                        <td className="ellipsis">…</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="pagination">
                <button
                  className="page-btn"
                  disabled={page <= 1}
                  onClick={() => loadDataset(page - 1)}
                >← Trước</button>
                <span className="page-info">Trang {page} / {totalPages} · {total.toLocaleString()} bản ghi</span>
                <button
                  className="page-btn"
                  disabled={page >= totalPages}
                  onClick={() => loadDataset(page + 1)}
                >Sau →</button>
              </div>
            </>
          )}

          {!dsLoading && dsRows.length === 0 && !dsError && (
            <div className="empty-card" style={{ marginTop: "2rem" }}>
              <div className="empty-icon">📭</div>
              <p>Bấm <strong>Sync &amp; Load</strong> để kéo dữ liệu về hiển thị.</p>
            </div>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════
          TAB 3: OVERVIEW
      ══════════════════════════════════════════ */}
      {tab === "overview" && (
        <div className="dashboard-grid">
          <div className="glass-card">
            <h2 className="card-title">📐 Mô hình & Thuật toán</h2>
            {[
              ["Dataset gốc", "UCI Online Retail II (525,461 giao dịch)"],
              ["Sau xử lý", "13,114 Monthly Snapshot"],
              ["Feature Engineering", "Rolling RFM 3M/6M + Momentum"],
              ["Số features", "22 biến đặc trưng"],
              ["Trạng thái ẩn", "HMM GaussianHMM (4 states)"],
              ["Dự báo Ngủ đông", "XGBoost (200 trees, lr=0.05)"],
              ["Dự báo Rớt hạng", "Random Forest (100 trees)"],
              ["Hạng thẻ", "Bronze / Silver / Gold"],
            ].map(([k, v]) => (
              <div key={k} className="result-item">
                <span className="result-label">{k}:</span>
                <span className="result-value">{v}</span>
              </div>
            ))}
          </div>
          <div className="glass-card">
            <h2 className="card-title">🏗️ Kiến trúc hệ thống</h2>
            {[
              ["Frontend", "Next.js 15 (App Router)"],
              ["Backend", "FastAPI + Uvicorn"],
              ["Database", "PostgreSQL 15"],
              ["ML Models", "HuggingFace Hub (vancevo/loyalty-models)"],
              ["Dataset", "HuggingFace Datasets (vancevo/loyalty-behavior-dataset)"],
              ["Containerization", "Docker Compose (3 services)"],
              ["Deploy", "Vercel (FE + BE Serverless)"],
            ].map(([k, v]) => (
              <div key={k} className="result-item">
                <span className="result-label">{k}:</span>
                <span className="result-value">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RawDataTab() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const PAGE_SIZE = 15;

  const loadRaw = useCallback(async (p = 1, s = "") => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/api/raw-data?page=${p}&page_size=${PAGE_SIZE}&search=${s}`);
      if (!res.ok) throw new Error("Lỗi tải dữ liệu");
      const data = await res.json();
      setRows(data.data || []);
      setTotal(data.total || 0);
      setPage(p);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRaw(1, search);
  }, [loadRaw, search]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="glass-card full-width-card">
      <div className="dataset-header">
        <h2 className="card-title">📂 Dữ liệu thô (Customer ID & Name)</h2>
        <div className="search-box">
          <input
            type="text"
            placeholder="Tìm theo ID hoặc Tên..."
            className="input-field"
            style={{ width: "250px", marginBottom: 0 }}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {error && <p className="error-msg">{error}</p>}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Customer ID</th>
              <th>Customer Name</th>
              <th>Country</th>
              <th>Total Orders</th>
              <th>Total Spend</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan="5" style={{ textAlign: "center", padding: "2rem" }}>Đang tải...</td></tr>
            ) : rows.length > 0 ? (
              rows.map((row) => (
                <tr key={row.customer_id}>
                  <td><strong>{row.customer_id}</strong></td>
                  <td style={{ color: "var(--primary)" }}>{row.customer_name}</td>
                  <td>{row.country}</td>
                  <td>{row.total_orders}</td>
                  <td>${row.total_spend?.toLocaleString()}</td>
                </tr>
              ))
            ) : (
              <tr><td colSpan="5" style={{ textAlign: "center", padding: "2rem" }}>Không tìm thấy dữ liệu</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button
          className="page-btn"
          disabled={page <= 1}
          onClick={() => loadRaw(page - 1, search)}
        >← Trước</button>
        <span className="page-info">Trang {page} / {totalPages} · {total.toLocaleString()} khách hàng</span>
        <button
          className="page-btn"
          disabled={page >= totalPages}
          onClick={() => loadRaw(page + 1, search)}
        >Sau →</button>
      </div>
    </div>
  );
}

function PipelineTab() {
  const [activeStep, setActiveStep] = useState(null);

  const toggleStep = (idx) => {
    setActiveStep(activeStep === idx ? null : idx);
  };

  const steps = [
    {
      title: "1. Thu thập dữ liệu",
      desc: "Nạp bộ dữ liệu giao dịch Online Retail II từ UCI Machine Learning Repository với hơn 500,000 bản ghi.",
      icon: "📥",
      sampleCode: `import pandas as pd
# Đọc file CSV chứa 1,067,371 dòng dữ liệu giao dịch
df = pd.read_csv("online_retail_II.csv")
print(f"Dữ liệu gốc: {df.shape}")`
    },
    {
      title: "2. Làm sạch dữ liệu",
      desc: "Loại bỏ các giao dịch thiếu Customer ID, các đơn hàng bị hủy (Canceled) và các dòng có số lượng/giá trị không hợp lệ.",
      icon: "🧹",
      sampleCode: `# Lọc bỏ missing Customer ID
df = df.dropna(subset=['Customer ID'])
# Lọc bỏ đơn hủy (bắt đầu bằng C)
df = df[~df['Invoice'].astype(str).str.startswith('C')]
# Lọc số lượng và giá hợp lệ
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]`
    },
    {
      title: "3. Chuẩn hóa & Mức khách hàng",
      desc: "Tính toán tổng giá trị từng dòng (Quantity * Price). Chuẩn hóa định dạng thời gian và gom nhóm dữ liệu theo từng tháng (YearMonth).",
      icon: "⚙️",
      sampleCode: `df['TotalSum'] = df['Quantity'] * df['Price']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# Nhóm theo Tháng-Năm để phân tích hành vi theo thời gian
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')`
    },
    {
      title: "4. Snapshot hành vi tháng",
      desc: "Xây dựng dạng chuỗi thời gian: Mỗi khách hàng có một bản ghi cho mỗi tháng, tóm tắt các chỉ số RFM cơ bản (Recency, Frequency, Monetary).",
      icon: "📸",
      sampleCode: `monthly = df.groupby(['Customer ID', 'YearMonth']).agg({
    'InvoiceDate': 'max',  # Tính Recency
    'Invoice': 'nunique',  # Tính Frequency
    'TotalSum': 'sum',     # Tính Monetary
    'Quantity': 'sum'
}).reset_index()

monthly['Recency'] = (monthly['YearMonth'].dt.to_timestamp(how='end') - monthly['InvoiceDate']).dt.days`
    },
    {
      title: "5. Trích xuất đặc trưng (Feature Engineering)",
      desc: "Sử dụng cửa sổ trượt (Rolling 3M/6M) và tính toán động lượng (Momentum) để bắt được xu hướng tiêu dùng thay đổi qua các tháng.",
      icon: "🧬",
      sampleCode: `# Tính Rolling 3 Tháng cho Frequency
monthly['Frequency_3M'] = monthly.groupby('Customer ID')['Frequency'].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)

# Tính Momentum (Động lượng thay đổi so với tháng trước)
monthly['Prev_Monetary'] = monthly.groupby('Customer ID')['Monetary'].shift(1)
monthly['Monetary_Change_1M'] = monthly['Monetary'] - monthly['Prev_Monetary'].fillna(0)`
    },
    {
      title: "6. Điểm Loyalty & Hạng thẻ",
      desc: "Sử dụng thuật toán Quantile (qcut) chia đều R, F, M thành 5 mức điểm. Tính Loyalty Score và gán hạng Bronze, Silver, Gold.",
      icon: "🏅",
      sampleCode: `# Chia R, F, M thành 5 quantile
monthly['R_Score'] = pd.qcut(monthly['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(float)
monthly['F_Score'] = pd.qcut(monthly['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(float)

# Trọng số: 40% R, 30% F, 30% M
monthly['LoyaltyScore'] = 0.4 * monthly['R_Score'] + 0.3 * monthly['F_Score'] + 0.3 * monthly['M_Score']
# Phân hạng: >= 4 (Gold), >= 2.5 (Silver), < 2.5 (Bronze)`
    },
    {
      title: "7. Xây dựng nhãn mục tiêu",
      desc: "Look-ahead 1 tháng để tạo nhãn: Dormancy (Ngủ đông nếu không mua tháng sau) và Downgrade (Rớt hạng nếu hạng tháng sau thấp hơn hiện tại).",
      icon: "🏷️",
      sampleCode: `monthly['Next_YearMonth'] = monthly.groupby('Customer ID')['YearMonth'].shift(-1)
# Nếu tháng mua tiếp theo cách > 1 tháng -> Dormancy = 1
monthly['Dormancy_Label'] = np.where(monthly['Next_Month_Diff'] == 1, 0, 1)

# Nếu hạng tháng tiếp theo nhỏ hơn hạng hiện tại -> Downgrade = 1
monthly['Next_Tier'] = monthly.groupby('Customer ID')['LoyaltyTier'].shift(-1)
monthly['Downgrade_Label'] = np.where(monthly['Next_Tier'] < monthly['LoyaltyTier'], 1, 0)`
    },
    {
      title: "8. Huấn luyện HMM (Markov ẩn)",
      desc: "Dùng GaussianHMM phân tích chuỗi RFM để tìm ra 4 'trạng thái ẩn' (Hidden States) phản ánh giai đoạn vòng đời của khách hàng.",
      icon: "🔄",
      sampleCode: `from hmmlearn import hmm

# Khởi tạo mô hình Markov ẩn với 4 trạng thái
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)

# Huấn luyện trên chuỗi giao dịch của toàn bộ khách hàng
lengths = monthly.groupby('Customer ID').size().values
hmm_model.fit(rfm_scaled, lengths)

# Suy luận trạng thái ẩn hiện tại
monthly['Hidden_State'] = hmm_model.predict(rfm_scaled, lengths)`
    },
    {
      title: "9. Huấn luyện Machine Learning",
      desc: "Train XGBoost cho bài toán phân loại Ngủ đông (có độ mất cân bằng) và Random Forest cho dự báo Rớt hạng với 22 đặc trưng đầu vào.",
      icon: "🧠",
      sampleCode: `from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Mô hình dự báo Ngủ đông
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train_d, y_train_d)

# Mô hình dự báo Rớt hạng
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_dw, y_train_dw)`
    },
    {
      title: "10. Sinh đầu ra & Can thiệp tự động",
      desc: "Hệ thống API trả về xác suất rủi ro, phân loại mức độ (Cao/TB/Thấp) và đề xuất hành động Marketing cụ thể ngay trên Dashboard.",
      icon: "🚀",
      sampleCode: `# Tại FastAPI (Dự báo realtime)
dormancy_prob = xgb_dormancy.predict_proba(df_input)[0][1]
downgrade_prob = rf_downgrade.predict_proba(df_input)[0][1]

if dormancy_prob > 0.80:
    action = "Gửi voucher kích hoạt mua lại ngay"
elif downgrade_prob > 0.60:
    action = "Ưu đãi giữ hạng và chăm sóc VIP"`
    }
  ];

  return (
    <div className="glass-card full-width-card pipeline-card">
      <h2 className="card-title" style={{ textAlign: "center", marginBottom: "2rem" }}>
        Trình tự 10 bước Khai phá Dữ liệu (Data Mining)
      </h2>
      <p style={{ textAlign: "center", color: "#94a3b8", marginBottom: "2rem" }}>
        Click vào từng bước để xem mã nguồn (sample code) minh họa cách hệ thống xử lý.
      </p>

      <div className="pipeline-container">
        {steps.map((step, idx) => (
          <div
            key={idx}
            className={`pipeline-step \${activeStep === idx ? 'active' : ''}`}
            onClick={() => toggleStep(idx)}
          >
            <div className="step-header">
              <div className="step-icon">{step.icon}</div>
              <div className="step-content">
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
              </div>
              <div className="step-toggle">
                {activeStep === idx ? '▲' : '▼'}
              </div>
            </div>

            {activeStep === idx && (
              <div className="step-code-preview">
                <div className="code-header">
                  <span className="dot red"></span>
                  <span className="dot yellow"></span>
                  <span className="dot green"></span>
                  <span className="file-name">Python / Pandas Snippet</span>
                </div>
                <pre><code>{step.sampleCode}</code></pre>
              </div>
            )}
          </div>
        ))}
      </div>

      <style jsx>{`
        .pipeline-card {
          background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        }
        .pipeline-container {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          max-width: 900px;
          margin: 0 auto;
        }
        .pipeline-step {
          background: rgba(255, 255, 255, 0.05);
          border-left: 4px solid var(--primary);
          border-radius: 0 12px 12px 0;
          transition: all 0.3s ease;
          cursor: pointer;
          overflow: hidden;
        }
        .pipeline-step:hover {
          background: rgba(255, 255, 255, 0.08);
          transform: translateX(5px);
        }
        .pipeline-step.active {
          background: rgba(255, 255, 255, 0.1);
          border-left-width: 6px;
        }
        .step-header {
          display: flex;
          align-items: center;
          padding: 1.5rem;
          gap: 1rem;
        }
        .step-icon {
          font-size: 2.5rem;
          background: rgba(99, 102, 241, 0.2);
          width: 60px;
          height: 60px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .step-content {
          flex: 1;
        }
        .step-content h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.1rem;
          color: #e2e8f0;
        }
        .step-content p {
          margin: 0;
          font-size: 0.9rem;
          color: #94a3b8;
          line-height: 1.5;
        }
        .step-toggle {
          color: #64748b;
          font-size: 1.2rem;
          padding: 0.5rem;
        }
        .step-code-preview {
          background: #0f172a;
          margin: 0 1.5rem 1.5rem 1.5rem;
          border-radius: 8px;
          border: 1px solid #1e293b;
          overflow: hidden;
          animation: slideDown 0.3s ease-out forwards;
        }
        .code-header {
          background: #1e293b;
          padding: 0.5rem 1rem;
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }
        .dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }
        .dot.red { background: #ff5f56; }
        .dot.yellow { background: #ffbd2e; }
        .dot.green { background: #27c93f; }
        .file-name {
          margin-left: 1rem;
          color: #94a3b8;
          font-size: 0.8rem;
          font-family: monospace;
        }
        .step-code-preview pre {
          margin: 0;
          padding: 1rem;
          overflow-x: auto;
        }
        .step-code-preview code {
          font-family: 'Fira Code', 'Courier New', Courier, monospace;
          color: #38bdf8;
          font-size: 0.9rem;
          line-height: 1.5;
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div >
  );
}

function ProbBar({ label, value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-header">
        <span>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div className="prob-bar-bg">
        <div
          className="prob-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function MetaItem({ label, value }) {
  return (
    <div className="meta-item">
      <div className="meta-label">{label}</div>
      <div className="meta-value">{value}</div>
    </div>
  );
}
