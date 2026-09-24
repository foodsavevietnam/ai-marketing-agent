"""
AI MARKETING AGENT – bản Python (Streamlit + Gemini)
Chuyển từ workflow n8n:  Excel → Dashboard → AI Chiến lược → AI Viết content
Chạy:  streamlit run app.py
"""
import os
import re
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai

st.set_page_config(page_title="FoodSave · AI Marketing Agent", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# 1. BỘ NÃO – kết nối Gemini
# =====================================================================
SKIP_WORDS = ("embedding", "tts", "image", "imagen", "veo", "audio", "live", "aqa", "robotics", "computer-use")


def get_api_key():
    """Lấy key theo thứ tự: ô nhập ở sidebar → st.secrets (khi deploy) → biến môi trường."""
    key = st.session_state.get("api_key_input", "").strip()
    if key:
        return key
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def list_models(api_key: str):
    """Hỏi Google xem hiện tại key này được dùng những model viết chữ nào (vì Google hay đổi model)."""
    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        actions = m.supported_actions or []
        name = (m.name or "").replace("models/", "")
        if "generateContent" in actions and not any(w in name for w in SKIP_WORDS):
            names.append(name)

    def rank(n):  # ưu tiên: flash → không phải preview → phiên bản mới
        ver = re.findall(r"\d+(?:\.\d+)?", n)
        return ("flash" not in n, "preview" in n or "exp" in n, "lite" in n, -float(ver[0]) if ver else 0)

    return sorted(set(names), key=rank)


def ask_gemini(prompt_or_history, model: str, backup_models=(), retries: int = 2):
    """Gọi Gemini. Nếu model quá tải (503/429) thì tự thử lại, vẫn lỗi thì tự đổi sang model dự phòng."""
    client = genai.Client(api_key=get_api_key())
    last_err = None
    for m in [model, *[b for b in backup_models if b != model]][:4]:
        for attempt in range(retries):
            try:
                res = client.models.generate_content(model=m, contents=prompt_or_history)
                return res.text or "", m
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                if any(x in msg for x in ("503", "UNAVAILABLE", "overloaded", "high demand", "429", "RESOURCE_EXHAUSTED")):
                    time.sleep(3 * (attempt + 1))
                    continue          # thử lại cùng model
                if any(x in msg for x in ("404", "NOT_FOUND", "no longer available")):
                    break             # model bị khai tử → sang model khác
                raise                 # lỗi khác (sai key...) → báo ngay
    raise RuntimeError(f"Gemini đang quá tải, đã thử nhiều model nhưng chưa được. Chi tiết: {last_err}")


# =====================================================================
# 2. DASHBOARD – tự nhận diện cột (giống node Code trong n8n)
# =====================================================================
ID_RE = re.compile(r"(^|_|\s)(id|key|code|sku|ma|mã|stt|index|phone|sdt|email)($|_|\s)", re.I)
DATE_RE = re.compile(r"date|ngay|ngày|thang|tháng|month|time|thoi_gian|thời gian|year|nam|năm|period|ky|kỳ", re.I)
MONEY_RE = re.compile(r"doanh|revenue|sales|value|amount|gia_tri|giá trị|tien|tiền|thu|gmv|profit|loi_nhuan", re.I)
QTY_RE = re.compile(r"qty|so_luong|số lượng|quantity|don|đơn", re.I)


def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    t = s.astype(str).str.replace(r"[\s₫đ%$]|VND", "", regex=True, flags=re.I)
    if t.str.count(r"\.").gt(1).any():            # kiểu VN 1.200.000 → dấu chấm là phân cách nghìn
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif t.str.count(",").gt(1).any() or t.str.contains(r",\d{3}$").all():   # kiểu 1,200,000
        t = t.str.replace(",", "", regex=False)
    else:
        t = t.str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def to_period(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m")
    if pd.api.types.is_numeric_dtype(s):
        if s.dropna().between(1900, 2100).all() and DATE_RE.search(name):
            return s.astype("Int64").astype(str)                                  # cột năm
        if s.dropna().between(20000, 80000).all():                                # số serial ngày Excel
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce").dt.strftime("%Y-%m")
        return pd.Series([None] * len(s), index=s.index)
    txt = s.astype(str)
    if not txt.str.contains(r"[-/.]").mean() > 0.8:
        return pd.Series([None] * len(s), index=s.index)
    d = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=True)
    return d.dt.strftime("%Y-%m")


def build_dashboard(df: pd.DataFrame, top: int = 10, max_dims: int = 6, max_measures: int = 8):
    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)
    info, num_cache, period_cache = {}, {}, {}

    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            info[c] = "trống"
            continue
        uniq = s.astype(str).nunique()
        per = to_period(s, c)
        per_rate = per.notna().mean()
        num = to_number(s)
        num_rate = num.notna().mean()
        if pd.api.types.is_datetime64_any_dtype(s) or (DATE_RE.search(c) and per_rate > 0.8) \
                or (not DATE_RE.search(c) and s.dtype == object and per_rate > 0.9):
            info[c] = "thời gian"
            period_cache[c] = to_period(df[c], c)
        elif num_rate > 0.9:
            ints = (num.dropna() % 1 == 0).all()
            if ID_RE.search(c) and uniq > 30:
                info[c] = "mã/ID"
            elif ID_RE.search(c) and uniq <= 12:
                info[c] = "nhóm"
            elif uniq <= 5 and ints and len(s) > 30 and not QTY_RE.search(c):
                info[c] = "nhóm"
            elif uniq / len(s) > 0.95 and len(s) > 50 and ints and num.iloc[0] > 1e6:
                info[c] = "mã/ID"
            else:
                info[c] = "số"
                num_cache[c] = to_number(df[c]).fillna(0)
        elif ID_RE.search(c) and uniq / len(s) > 0.5:
            info[c] = "mã/ID"
        elif uniq <= max(50, n * 0.05):
            info[c] = "nhóm"
        else:
            info[c] = "văn bản"

    measures = [c for c in df.columns if info[c] == "số"]
    dims = sorted([c for c in df.columns if info[c] == "nhóm"], key=lambda c: df[c].nunique())[:max_dims]
    time_col = next((c for c in df.columns if info[c] == "thời gian"), None)
    sums = {m: float(num_cache[m].sum()) for m in measures}
    main = next((m for m in measures if MONEY_RE.search(m)), None) or (max(sums, key=sums.get) if sums else None)
    use = ([main] + [m for m in measures if m != main])[:max_measures] if main else []

    nums = pd.DataFrame({m: num_cache[m] for m in use})
    totals = {m: {"tổng": sums[m], "trung bình": float(nums[m].mean()),
                  "nhỏ nhất": float(nums[m].min()), "lớn nhất": float(nums[m].max())} for m in use}

    by_dim = {}
    for d in dims:
        g = nums.assign(_k=df[d].astype(str)).groupby("_k")[use].sum()
        g["số dòng"] = df[d].astype(str).value_counts()
        base = g[main] if main else g["số dòng"]
        g["tỷ trọng %"] = (100 * base / (base.sum() or 1)).round(2)
        by_dim[d] = g.sort_values(main or "số dòng", ascending=False).head(top).reset_index().rename(columns={"_k": d})

    trend, change = None, None
    if time_col and use:
        trend = nums.assign(kỳ=period_cache[time_col]).dropna(subset=["kỳ"]).groupby("kỳ")[use].sum().sort_index()
        if len(trend) >= 2:
            last, prev = trend.iloc[-1], trend.iloc[-2]
            change = {"kỳ mới nhất": trend.index[-1], "kỳ trước": trend.index[-2],
                      **{m: (round(100 * (last[m] - prev[m]) / prev[m], 2) if prev[m] else None) for m in use}}

    # Bản tóm tắt ngắn cho AI (giống field tom_tat_cho_AI trong n8n)
    f = lambda v: "-" if v is None or pd.isna(v) else f"{v:,.2f}".rstrip("0").rstrip(".")
    lines = [f"Dữ liệu: {n} dòng. Chỉ số chính: {main or 'số dòng'}. Cột thời gian: {time_col or 'không có'}.",
             "TỔNG: " + "; ".join(f"{m}={f(totals[m]['tổng'])}" for m in use)]
    if change:
        lines.append(f"BIẾN ĐỘNG {change['kỳ mới nhất']} so với {change['kỳ trước']}: " +
                     "; ".join(f"{m} {'+' if (change[m] or 0) > 0 else ''}{f(change[m])}%" for m in use))
    for d in dims:
        rows = by_dim[d].head(5)
        lines.append(f"THEO {d} (top 5): " + " | ".join(
            f"{r[d]} [" + ", ".join(f"{m}={f(r[m])}" for m in use[:5]) + f", tỷ trọng {r['tỷ trọng %']}%]"
            for _, r in rows.iterrows()))
    if trend is not None and main:
        lines.append(f"XU HƯỚNG {main} 12 kỳ gần nhất: " + ", ".join(f"{k}: {f(v)}" for k, v in trend[main].tail(12).items()))

    return {"info": info, "main": main, "measures": use, "dims": dims, "time_col": time_col,
            "totals": totals, "by_dim": by_dim, "trend": trend, "change": change, "summary": "\n".join(lines)}



# =====================================================================
# 3. PROMPT – giống 2 node Gemini trong n8n
# =====================================================================
PROMPT_STRATEGY = """Bạn là Giám đốc Marketing (CMO). Dưới đây là dashboard được tính tự động từ dữ liệu Excel:
{summary}

PHẦN 1 – INSIGHT: nêu 5 insight quan trọng nhất, mỗi insight phải có số liệu dẫn chứng
(doanh thu, tỷ trọng %, tăng/giảm % kỳ gần nhất; nếu dữ liệu có lượt xem và đơn hàng thì tính thêm tỷ lệ chuyển đổi).

PHẦN 2 – CHIẾN LƯỢC MARKETING DO BẠN TỰ QUYẾT ĐỊNH (mỗi quyết định giải thích dựa trên insight nào):
1. Mục tiêu chiến dịch
2. Sản phẩm / nhóm hàng cần đẩy
3. Khách hàng mục tiêu và khu vực ưu tiên
4. Chọn 2–3 kênh quảng cáo phù hợp nhất trong: Facebook, TikTok, Instagram, Google Ads, Shopee Ads, Zalo, Email marketing, YouTube
5. Thông điệp chính và giọng văn
6. Phân bổ ngân sách (%) cho từng kênh đã chọn

Trả lời ngắn gọn, súc tích bằng tiếng Việt, trình bày bằng Markdown."""

PROMPT_CONTENT = """Dưới đây là insight và chiến lược marketing đã được quyết định:
{strategy}

Hãy triển khai chiến lược: với MỖI kênh quảng cáo đã được chọn, viết content sẵn sàng đăng, đúng định dạng của kênh đó:
- Facebook/Instagram: tiêu đề, nội dung, hashtag, lời kêu gọi hành động, gợi ý hình ảnh
- TikTok/YouTube: kịch bản video 15–30 giây (cảnh quay + lời thoại + chữ trên video)
- Google Ads: 3 headline (≤30 ký tự) + 2 description (≤90 ký tự)
- Shopee Ads: tên chương trình, từ khóa, nội dung ưu đãi
- Zalo/Email: tiêu đề + nội dung tin nhắn
Chỉ viết cho các kênh đã được chọn trong chiến lược. Trả lời bằng tiếng Việt, trình bày bằng Markdown."""


def short_num(v):
    """35445705429 → 35.45 tỷ ; 825894 → 825,894"""
    for unit, name in ((1e9, " tỷ"), (1e6, " triệu")):
        if abs(v) >= unit:
            return f"{v / unit:,.2f}{name}"
    return f"{v:,.0f}"


# =====================================================================
# 4. GIAO DIỆN – phong cách FoodSave (xanh lá · vàng · đen, bo tròn)
# =====================================================================
GREEN, GREEN_DARK, YELLOW, INK = "#22C55E", "#0F3D24", "#FACC15", "#0B0B0B"
CHART_COLORS = ["#16A34A", "#FACC15", "#0F3D24", "#86EFAC", "#F59E0B", "#4ADE80", "#A3A3A3", "#15803D"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800;900&display=swap');

.stApp *:not([translate="no"]):not([data-testid*="Icon"]) {{
  font-family: 'Be Vietnam Pro', sans-serif;
}}
.stApp {{ background: radial-gradient(1200px 500px at 85% 10%, #F2FBF4 0%, #FFFFFF 60%); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"], footer, #MainMenu {{ visibility: hidden; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 0.8rem; max-width: 1180px; }}

/* ---------- thanh thông báo trên cùng ---------- */
.fs-topbar {{
  background: linear-gradient(90deg, {GREEN_DARK}, #166534 55%, {GREEN_DARK});
  color: #fff; border-radius: 14px; padding: 9px 18px; margin-bottom: 14px;
  display: flex; gap: 18px; align-items: center; justify-content: center; font-size: 13.5px; font-weight: 600;
}}
.fs-live {{ border: 1px solid {YELLOW}; color: {YELLOW}; border-radius: 999px; padding: 2px 12px;
  font-weight: 800; letter-spacing: 1.5px; font-size: 12px; }}
.fs-live::before {{ content: "●"; margin-right: 6px; animation: blink 1.4s infinite; }}
@keyframes blink {{ 50% {{ opacity: .25; }} }}

/* ---------- logo ---------- */
.fs-logo {{ font-size: 30px; font-weight: 900; letter-spacing: -1px; line-height: 1; color: {INK}; }}
.fs-logo span {{ color: {GREEN}; }}
.fs-logo sup {{ display:inline-block; width:12px; height:12px; background:{YELLOW}; border-radius:50%; vertical-align: top; }}
.fs-logo-underline {{ width: 132px; height: 4px; background: {YELLOW}; border-radius: 4px; margin: 3px 0 5px; }}
.fs-logo-sub {{ font-size: 11.5px; font-weight: 700; letter-spacing: 2.5px; color: #6B7280; }}

/* ---------- hero ---------- */
.fs-chip {{ display:inline-block; background:#ECFDF3; border:1px solid #BBF7D0; color:#15803D; border-radius:999px;
  padding:7px 16px; font-size:12.5px; font-weight:800; letter-spacing:.6px; margin: 18px 0 8px; }}
.fs-h1 {{ font-size: 54px; line-height: 1.08; font-weight: 900; letter-spacing: -2px; color:{INK}; margin: 6px 0 14px; }}
.fs-h1 .g {{ color: {GREEN}; }}
.fs-h1 .u {{ color: {GREEN}; background: linear-gradient(transparent 78%, #FEF08A 78%); }}
.fs-lead {{ font-size: 17px; color:#374151; line-height:1.65; max-width: 560px; }}
.fs-tags {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
.fs-tag {{ border:1px solid #E5E7EB; background:#fff; border-radius:999px; padding:6px 14px; font-size:12px;
  font-weight:700; letter-spacing:.8px; color:#4B5563; }}

.fs-steps {{ background: {GREEN_DARK}; color:#fff; border-radius: 28px; padding: 26px 26px 12px; margin-top: 22px;
  box-shadow: 0 25px 50px -20px rgba(15,61,36,.55); }}
.fs-steps h4 {{ color:{YELLOW}; font-size:13px; letter-spacing:2px; font-weight:800; margin:0 0 14px; }}
.fs-step {{ display:flex; gap:14px; align-items:flex-start; background: rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.1); border-radius:18px; padding:13px 15px; margin-bottom:12px; }}
.fs-step b {{ background:{YELLOW}; color:{INK}; border-radius:12px; min-width:34px; height:34px; display:flex;
  align-items:center; justify-content:center; font-weight:900; }}
.fs-step div {{ font-size:14px; line-height:1.45; }}
.fs-step div strong {{ display:block; font-size:15px; }}

/* ---------- tabs dạng viên thuốc ---------- */
[role="tablist"] {{ gap: 6px; background:#F3F4F6; padding:6px; border-radius:999px; width: fit-content;
  margin-top: 26px; border: none !important; }}
[data-testid="stTab"] {{ border-radius:999px; padding: 10px 24px; transition: all .2s; }}
[data-testid="stTab"] p {{ font-weight:700 !important; font-size:15px !important; }}
[data-testid="stTab"][aria-selected="true"] {{ background:{INK}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color:#fff !important; }}
.react-aria-SelectionIndicator {{ display:none !important; }}

/* ---------- tiêu đề bước ---------- */
.fs-section {{ display:flex; align-items:center; gap:12px; margin: 30px 0 12px; }}
.fs-section .num {{ background:{GREEN}; color:#fff; font-weight:900; border-radius:12px; width:36px; height:36px;
  display:flex; align-items:center; justify-content:center; }}
.fs-section .t {{ font-size: 24px; font-weight: 800; letter-spacing:-.5px; color:{INK}; }}
.fs-section .s {{ color:#6B7280; font-size:14px; }}

/* ---------- thẻ số liệu ---------- */
.fs-kpi {{ background:#fff; border:1px solid #E5E7EB; border-radius:22px; padding:18px 20px;
  box-shadow: 0 10px 30px -18px rgba(0,0,0,.25); height: 100%; }}
.fs-kpi .l {{ font-size:12px; font-weight:800; letter-spacing:1.2px; color:#6B7280; text-transform:uppercase; }}
.fs-kpi .v {{ font-size:30px; font-weight:900; color:{INK}; letter-spacing:-1px; margin:6px 0 8px; }}
.fs-kpi .d {{ display:inline-block; border-radius:999px; padding:3px 10px; font-size:12.5px; font-weight:700; }}
.fs-kpi .up {{ background:#DCFCE7; color:#15803D; }}
.fs-kpi .down {{ background:#FEE2E2; color:#B91C1C; }}
.fs-kpi.main {{ background:{GREEN_DARK}; border-color:{GREEN_DARK}; }}
.fs-kpi.main .l {{ color:#BBF7D0; }} .fs-kpi.main .v {{ color:{YELLOW}; }}

/* ---------- khung có viền (chart, kết quả) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 24px !important; background:#fff; }}
[data-testid="stExpander"] details {{ border-radius: 16px; border-color:#E5E7EB; background:#fff; }}

/* ---------- ô upload ---------- */
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed #86EFAC; background:#F0FDF4; border-radius:22px; padding: 22px; }}

/* ---------- nút bấm ---------- */
.stButton > button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button {{
  border-radius:999px !important; font-weight:700 !important; padding: 10px 22px !important; }}
.stButton > button[kind="primary"] {{ background:{GREEN}; border:1.5px solid {INK}; color:#fff;
  box-shadow: 0 10px 22px -10px rgba(34,197,94,.9); font-size:16px; }}
.stButton > button[kind="primary"]:hover {{ background:#16A34A; color:#fff; border-color:{INK}; }}
[data-testid="stDownloadButton"] button {{ background:{YELLOW} !important; border:none !important; color:{INK} !important; }}
[data-testid="stPopover"] button {{ background:#fff; border:1.5px solid #E5E7EB; }}

/* ---------- chat ---------- */
[data-testid="stChatInput"] {{ border-radius: 999px; border: 1.5px solid #BBF7D0; }}
[data-testid="stChatMessage"] {{ border-radius: 20px; }}

.fs-footer {{ text-align:center; color:#9CA3AF; font-size:13px; margin: 50px 0 10px; }}
@media (max-width: 800px) {{ .fs-h1 {{ font-size: 38px; }} }}
</style>
"""


def section(num, title, sub=""):
    st.markdown(f'<div class="fs-section"><div class="num">{num}</div><div><div class="t">{title}</div>'
                f'<div class="s">{sub}</div></div></div>', unsafe_allow_html=True)


def style_fig(fig, title):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color=INK), x=0.01),
        font=dict(family="Be Vietnam Pro, sans-serif", color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10), legend=dict(orientation="h", y=-0.12), colorway=CHART_COLORS)
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#F1F5F9", title=None)
    return fig


def show_dashboard(d):
    main = d["main"]
    kpis = d["measures"][:4]
    cols = st.columns(len(kpis) or 1)
    for i, m in enumerate(kpis):
        delta = d["change"].get(m) if d["change"] else None
        badge = ""
        if delta is not None:
            cls, arrow = ("up", "▲") if delta >= 0 else ("down", "▼")
            badge = f'<span class="d {cls}">{arrow} {abs(delta)}% so với kỳ trước</span>'
        cols[i].markdown(f'<div class="fs-kpi {"main" if i == 0 else ""}"><div class="l">{m.replace("_", " ")}</div>'
                         f'<div class="v">{short_num(d["totals"][m]["tổng"])}</div>{badge}</div>', unsafe_allow_html=True)

    st.write("")
    if d["trend"] is not None and main:
        with st.container(border=True):
            fig = px.area(d["trend"].reset_index(), x="kỳ", y=main, markers=True)
            fig.update_traces(line=dict(color=GREEN, width=3), fillcolor="rgba(34,197,94,.12)",
                              marker=dict(color=YELLOW, size=7, line=dict(color=GREEN_DARK, width=1.5)))
            st.plotly_chart(style_fig(fig, f"Xu hướng {main.replace('_', ' ')} theo {d['time_col']}"), width="stretch")

    chart_cols = st.columns(2)
    for i, dim in enumerate(d["dims"][:4]):
        t = d["by_dim"][dim]
        y = main or "số dòng"
        if len(t) <= 6:
            fig = px.pie(t, names=dim, values=y, hole=0.55, color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#fff", width=3)))
        else:
            fig = px.bar(t, x=dim, y=y, color_discrete_sequence=[GREEN])
            fig.update_traces(marker_line_width=0)
        with chart_cols[i % 2].container(border=True):
            st.plotly_chart(style_fig(fig, f"{y.replace('_', ' ')} theo {dim.replace('_', ' ')}"), width="stretch")

    c1, c2 = st.columns(2)
    with c1.expander("🔎 Cách agent hiểu các cột dữ liệu"):
        st.dataframe(pd.DataFrame({"cột": list(d["info"]), "loại": list(d["info"].values())}), hide_index=True)
    with c2.expander("📝 Bản tóm tắt gửi cho AI"):
        st.code(d["summary"], language=None)


st.markdown(CSS, unsafe_allow_html=True)

# ---------- thanh thông báo + header ----------
st.markdown('<div class="fs-topbar">🤖 AI Marketing Agent cho cửa hàng đối tác <span class="fs-live">LIVE</span>'
            ' ⚡ Excel → Dashboard → Chiến lược → Content trong 2 phút</div>', unsafe_allow_html=True)

h1, h2 = st.columns([4, 1.3], vertical_alignment="center")
h1.markdown('<div class="fs-logo">FOOD<span>SAVE</span><sup></sup></div><div class="fs-logo-underline"></div>'
            '<div class="fs-logo-sub">AI MARKETING · AGENT</div>', unsafe_allow_html=True)
with h2.popover("⚙️ Cài đặt AI", width="stretch"):
    st.text_input("Gemini API key", type="password", key="api_key_input",
                  help="Lấy tại aistudio.google.com → Get API key. Khi deploy thì để trong Secrets.")
    api_key = get_api_key()
    models = []
    if api_key:
        try:
            models = list_models(api_key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Không kết nối được Gemini: {e}")
    if models:
        st.selectbox("Model (tự lấy danh sách đang hoạt động)", models, key="model")
        st.caption("✅ Đã kết nối. Model quá tải thì agent tự thử lại và tự đổi model dự phòng.")
    else:
        st.info("👉 Nhập API key để bật AI.")
    if st.button("🗑️ Xóa lịch sử chat", width="stretch"):
        st.session_state.messages = []

# ---------- hero ----------
left, right = st.columns([1.35, 1], gap="large")
left.markdown("""
<div class="fs-chip">🏪 DÀNH CHO CỬA HÀNG · QUÁN · BẾP · BAKERY · SIÊU THỊ</div>
<div class="fs-h1">Dữ liệu vào.<br><span class="u">Chiến lược</span> ra.<br>Content <span class="g">sẵn sàng</span> đăng.</div>
<div class="fs-lead">Tải file Excel bán hàng — agent tự dựng dashboard, AI đọc số liệu, rút insight,
<b>tự chọn kênh quảng cáo</b> phù hợp và viết content cho từng nền tảng.</div>
<div class="fs-tags"><span class="fs-tag">📊 DASHBOARD TỰ ĐỘNG</span><span class="fs-tag">🧠 GEMINI AI</span>
<span class="fs-tag">🎯 TỰ CHỌN KÊNH</span><span class="fs-tag">✍️ CONTENT ĐA NỀN TẢNG</span><span class="fs-tag">💬 HỎI ĐÁP DỮ LIỆU</span></div>
""", unsafe_allow_html=True)
right.markdown(f"""
<div class="fs-steps"><h4>CÁCH HOẠT ĐỘNG</h4>
<div class="fs-step"><b>1</b><div><strong>Tải dữ liệu Excel</strong>Tự nhận diện cột: thời gian, số liệu, nhóm</div></div>
<div class="fs-step"><b>2</b><div><strong>Dashboard tự động</strong>KPI, xu hướng, tỷ trọng theo từng nhóm</div></div>
<div class="fs-step"><b>3</b><div><strong>AI chọn chiến lược</strong>Insight · kênh · khách hàng · ngân sách</div></div>
<div class="fs-step"><b>4</b><div><strong>AI viết content</strong>Facebook, TikTok, Google Ads, Shopee...</div></div>
</div>""", unsafe_allow_html=True)

tab_agent, tab_chat = st.tabs(["📊 Marketing Agent", "💬 Hỏi đáp AI"])

# ---------------- TAB 1: quy trình giống n8n ----------------
with tab_agent:
    section("01", "Tải dữ liệu", "File .xlsx, .xls hoặc .csv — dòng đầu tiên là tên cột")
    file = st.file_uploader("Tải file dữ liệu", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
    if file:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            sheets = pd.ExcelFile(file).sheet_names
            sheet = st.selectbox("Chọn sheet", sheets) if len(sheets) > 1 else sheets[0]
            df = pd.read_excel(file, sheet_name=sheet)
        st.success(f"✅ Đã đọc **{len(df):,}** dòng × **{df.shape[1]}** cột")
        with st.expander("Xem dữ liệu gốc"):
            st.dataframe(df.head(200))

        section("02", "Dashboard tự động", "Agent tự tính chỉ số và vẽ biểu đồ, không cần cài đặt")
        dash = build_dashboard(df)
        st.session_state.dash = dash
        show_dashboard(dash)

        section("03", "AI phân tích & viết content", "Gemini đọc dashboard → rút insight → tự chọn kênh → viết bài")
        if not models:
            st.warning("Bấm **⚙️ Cài đặt AI** ở góc trên để nhập API key trước.")
        if st.button("🚀  Chạy AI Marketing Agent", type="primary", disabled=not models):
            model = st.session_state.model
            try:
                with st.status("AI đang làm việc...", expanded=True) as status:
                    st.write("🧠 Đang đọc dashboard, rút insight và chọn chiến lược...")
                    strategy, used1 = ask_gemini(PROMPT_STRATEGY.format(summary=dash["summary"]), model, models)
                    st.write(f"✅ Xong chiến lược (model: {used1})")
                    st.write("✍️ Đang viết content cho các kênh đã chọn...")
                    content, used2 = ask_gemini(PROMPT_CONTENT.format(strategy=strategy), used1, models)
                    st.write(f"✅ Xong content (model: {used2})")
                    status.update(label="Hoàn tất!", state="complete", expanded=False)
                st.session_state.strategy, st.session_state.content = strategy, content
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

        if st.session_state.get("strategy"):
            c1, c2 = st.columns(2, gap="medium")
            with c1.container(border=True):
                st.markdown("### 🧠 Insight & Chiến lược")
                st.markdown(st.session_state.strategy)
            with c2.container(border=True):
                st.markdown("### ✍️ Content quảng cáo")
                st.markdown(st.session_state.content)
            report = f"# Báo cáo AI Marketing Agent\n\n## Dashboard\n```\n{dash['summary']}\n```\n\n" \
                     f"## Insight & Chiến lược\n{st.session_state.strategy}\n\n## Content quảng cáo\n{st.session_state.content}\n"
            st.download_button("⬇️  Tải báo cáo (.md)", report, file_name="bao_cao_marketing.md")

# ---------------- TAB 2: chatbot hỏi đáp (bài của thầy) ----------------
with tab_chat:
    section("💬", "Hỏi đáp cùng AI", "Đã tải dữ liệu ở tab bên cạnh thì AI sẽ trả lời dựa trên dashboard đó")
    st.session_state.setdefault("messages", [])
    if not st.session_state.messages:
        st.markdown('<div class="fs-tags"><span class="fs-tag">GỢI Ý: Nhóm hàng nào nên đẩy quảng cáo?</span>'
                    '<span class="fs-tag">Kênh nào chuyển đổi tốt nhất?</span>'
                    '<span class="fs-tag">Viết 1 caption Facebook cho sản phẩm bán chạy</span></div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖").markdown(msg["text"])

    if question := st.chat_input("Nhập câu hỏi...", disabled=not models):
        st.session_state.messages.append({"role": "user", "text": question})
        st.chat_message("user", avatar="🧑").markdown(question)
        context = ""
        if st.session_state.get("dash"):
            context = "Bạn là trợ lý marketing. Dữ liệu dashboard hiện tại:\n" + st.session_state.dash["summary"] + "\n\n"
        history = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
                   for m in st.session_state.messages]
        history[0]["parts"][0]["text"] = context + history[0]["parts"][0]["text"]
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Đang suy nghĩ..."):
                    answer, _ = ask_gemini(history, st.session_state.model, models)
            except Exception as e:  # noqa: BLE001
                answer = f"⚠️ {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "text": answer})
        st.rerun()   # vẽ lại để khung chat luôn nằm dưới cùng

st.markdown('<div class="fs-footer">FoodSave · AI Marketing Agent — Môn Thương mại điện tử · Powered by Gemini</div>',
            unsafe_allow_html=True)
