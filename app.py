"""
AI MARKETING AGENT – bản Python (Streamlit + Gemini)
Chuyển từ workflow n8n:  Excel → Dashboard → AI Chiến lược → AI Viết content → Ảnh & Video quảng cáo
Chạy:  streamlit run app.py
"""
import base64
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="FoodSave · AI Marketing Agent", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# 0. CẤU HÌNH – chỉnh ở đây (người dùng web không cần chỉnh gì)
# =====================================================================
# Model viết chữ, xếp theo thứ tự "xịn nhất" trước. App tự bỏ qua model không dùng được.
TEXT_MODELS = [
    "gemini-3.1-pro-preview",   # xịn nhất – chỉ chạy khi key đã bật thanh toán
    "gemini-3.8-flash",         # xịn nhất trong gói MIỄN PHÍ
    "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite",
]
IMAGE_MODELS = ["gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"]   # Nano Banana 2 – cần bật thanh toán
VIDEO_MODEL = "gemini-omni-1.1-flash"                                       # video AI – cần bật thanh toán
TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]           # giọng đọc – miễn phí
VOICE = "Kore"
PAID_ONLY = {"gemini-3.1-pro-preview", *IMAGE_MODELS, VIDEO_MODEL}

# =====================================================================
# 1. BỘ NÃO – kết nối Gemini (key đặt trong Secrets, không hiện trên web)
# =====================================================================
SKIP_WORDS = ("embedding", "tts", "image", "imagen", "veo", "audio", "live", "aqa", "robotics", "computer-use", "omni")
BUSY = ("503", "UNAVAILABLE", "overloaded", "high demand")
NO_QUOTA = ("429", "RESOURCE_EXHAUSTED", "quota", "billing", "PERMISSION_DENIED", "403")


def get_api_key():
    """Đọc key từ Secrets (chấp nhận vài cách đặt tên), tự bỏ dấu cách / ngoặc kép thừa."""
    key = ""
    try:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "api_key"):
            if name in st.secrets:
                key = str(st.secrets[name])
                break
    except Exception:  # noqa: BLE001
        pass
    key = key or os.environ.get("GEMINI_API_KEY", "")
    return key.strip().strip('"').strip("'").strip("“”").strip()


def explain_error(e):
    """Dịch lỗi của Google sang tiếng Việt dễ hiểu."""
    msg = str(e)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg or "UNAUTHENTICATED" in msg or "401" in msg:
        return "API key không hợp lệ (sai, thiếu ký tự hoặc đã bị xóa). Tạo key mới tại aistudio.google.com rồi dán lại vào Secrets."
    if "SERVICE_DISABLED" in msg or "has not been used" in msg:
        return "Project của key chưa bật Gemini API. Nên tạo key mới ngay trong aistudio.google.com (Get API key → Create API key)."
    if "PERMISSION_DENIED" in msg or "403" in msg:
        return "Key bị giới hạn quyền (API restrictions / giới hạn website). Tạo key mới không giới hạn trong AI Studio."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Key đã hết lượt dùng miễn phí trong phút/ngày. Chờ một lúc hoặc dùng key khác."
    if "location" in msg.lower() and "not supported" in msg.lower():
        return "Máy chủ ở khu vực Google chưa hỗ trợ Gemini API."
    return msg[:300]


def new_client(api_key, mode="gemini"):
    """mode 'gemini' = key AI Studio (AIza...); mode 'vertex' = key Vertex AI express (thường bắt đầu bằng AQ.)."""
    return genai.Client(vertexai=True, api_key=api_key) if mode == "vertex" else genai.Client(api_key=api_key)


def _list_names(client):
    names = set()
    for m in client.models.list(config={"page_size": 1000}):
        names.add((m.name or "").split("/")[-1])
    return names


def _probe(client):
    """Một số loại key không cho xem danh sách model → thử gọi nhanh vài model phổ biến."""
    found = set()
    for m in TEXT_MODELS + ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]:
        try:
            client.models.generate_content(model=m, contents="hi",
                                           config=types.GenerateContentConfig(max_output_tokens=5))
            found.add(m)
            break
        except Exception as e:  # noqa: BLE001
            if any(x in str(e) for x in ("API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED")):
                raise
    return found


@st.cache_data(ttl=3600, show_spinner=False)
def connect(api_key: str):
    """Tự nhận loại key rồi trả về (mode, danh sách model dùng được)."""
    order = ["vertex", "gemini"] if not api_key.startswith("AIza") else ["gemini", "vertex"]
    first_err = None
    for mode in order:
        client = new_client(api_key, mode)          # giữ client sống suốt lúc gọi (tránh lỗi "client has been closed")
        try:
            names = _list_names(client) if mode == "gemini" else set()
            names = {n for n in names if n.startswith("gemini")} or _probe(client)
        except Exception as e:  # noqa: BLE001
            first_err = first_err or e
            continue
        if names:
            return mode, names
    raise first_err or RuntimeError("Không tìm thấy model Gemini nào dùng được với key này.")


def api_client():
    key = get_api_key()
    return new_client(key, connect(key)[0])


def available_models(api_key: str):
    return connect(api_key)[1]


def pick(candidates, extra_filter=None):
    """Lấy các model trong danh sách ưu tiên mà key dùng được, bỏ model trả phí nếu đã biết key là gói free."""
    avail = available_models(get_api_key())
    chain = [m for m in candidates if m in avail]
    if extra_filter and not chain:          # Google đổi tên model → tự tìm model tương tự
        def rank(n):
            ver = re.findall(r"\d+(?:\.\d+)?", n)
            return ("flash" not in n, "preview" in n, "lite" in n, -float(ver[0]) if ver else 0)
        chain = sorted([n for n in avail if extra_filter(n)], key=rank)
    if st.session_state.get("free_key"):
        chain = [m for m in chain if m not in PAID_ONLY]
    return chain


def text_models():
    return pick(TEXT_MODELS, lambda n: n.startswith("gemini") and not any(w in n for w in SKIP_WORDS))


def ask_gemini(contents, json_mode=False, search=False):
    """Gọi model xịn nhất dùng được. Quá tải → thử lại; không có quyền/hết lượt → tự chuyển model kế tiếp.
    search=True: cho AI tra Google để trả lời thông tin mới (lỗi thì tự tắt tra cứu)."""
    client = api_client()
    last_err = None
    for m in text_models()[:5]:
        use_search = search and not st.session_state.get("no_search")
        for attempt in range(3):
            if json_mode:
                cfg = types.GenerateContentConfig(response_mime_type="application/json")
            elif use_search:
                cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
            else:
                cfg = None
            try:
                res = client.models.generate_content(model=m, contents=contents, config=cfg)
                st.session_state.last_model = m
                return res.text or "", m
            except Exception as e:  # noqa: BLE001
                last_err, msg = e, str(e)
                if any(x in msg for x in BUSY):
                    time.sleep(3 * (attempt + 1))
                    continue
                if use_search:                              # tra Google không được → hỏi lại không tra cứu
                    use_search = False
                    st.session_state.no_search = True
                    continue
                if m in PAID_ONLY and any(x in msg for x in NO_QUOTA):
                    st.session_state.free_key = True        # key gói free → từ giờ bỏ qua model trả phí
                break                                       # 404/429... → sang model kế tiếp
    raise RuntimeError(f"Gemini đang quá tải hoặc hết lượt, thử lại sau ít phút. Chi tiết: {last_err}")


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


# =====================================================================
# 4. HÌNH ẢNH & VIDEO QUẢNG CÁO – tự động, miễn phí
#    Ảnh nền: Pollinations (miễn phí) → Gemini Image (nếu bật thanh toán) → nền thiết kế sẵn
#    Poster: ghép chữ bằng Pillow · Video: khung hình + giọng đọc Gemini TTS (miễn phí) → MP4
# =====================================================================


FONT_DIR = Path(__file__).parent / "fonts"
BRAND_GREEN, BRAND_DARK, BRAND_YELLOW = (34, 197, 94), (15, 61, 36), (250, 204, 21)

PROMPT_BRIEF = """Bạn là Creative Director của agency quảng cáo TikTok/Reels. Dữ liệu dashboard:
{summary}

Chiến lược đã chọn:
{strategy}

Content đã viết:
{content}

Hãy làm STORYBOARD cho 1 video quảng cáo dọc 9:16 (15–20 giây) thật cuốn hút + 1 poster.
Yêu cầu: câu mở đầu gây tò mò trong 2 giây đầu; dùng SỐ LIỆU THẬT từ dashboard; ngôn ngữ trẻ, ngắn, có nhịp.
Chỉ trả về JSON đúng cấu trúc (tiếng Việt có dấu, KHÔNG emoji):
{{
  "thuong_hieu": "tên thương hiệu/nhà bán nếu dữ liệu có ghi rõ, không có thì để rỗng",
  "san_pham": "sản phẩm/nhóm hàng được quảng cáo",
  "mau_chu_dao": "chọn 1 trong: xanh_la, cam, tim, xanh_duong, do, den_vang (hợp cảm xúc sản phẩm)",
  "headline": "tiêu đề poster tối đa 7 từ",
  "subline": "câu phụ poster tối đa 14 từ",
  "cta": "lời kêu gọi tối đa 3 từ",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "image_prompt": "English: vivid commercial photo of the product for ad background, studio light, no text, no logo",
  "scenes": [
    {{"loai": "hook", "kicker": "nhãn 2-3 từ", "text": "câu mở gây tò mò tối đa 8 từ", "nhan_manh": "1-2 từ trong text cần tô nổi", "voice": "lời đọc 8-12 từ", "image_prompt": "English, no text"}},
    {{"loai": "stat", "kicker": "nhãn 2-4 từ", "so_lieu": "1 con số thật, vd 58% hoặc 35,4 tỷ", "text": "giải thích con số tối đa 8 từ", "voice": "lời đọc 8-12 từ"}},
    {{"loai": "benefits", "text": "tiêu đề tối đa 5 từ", "y": ["lợi ích 1 tối đa 5 từ", "lợi ích 2", "lợi ích 3"], "voice": "lời đọc 10-14 từ", "image_prompt": "English, no text"}},
    {{"loai": "offer", "so_lieu": "ưu đãi rất ngắn: -20% / FREESHIP / 1+1", "text": "mô tả ưu đãi tối đa 8 từ", "nhan_manh": "1-2 từ", "voice": "lời đọc 8-12 từ"}},
    {{"loai": "cta", "text": "câu chốt tối đa 6 từ", "voice": "lời kêu gọi 4-8 từ"}}
  ]
}}"""


def clean_text(s):
    """Bỏ emoji và ký tự lạ mà font không vẽ được."""
    s = re.sub(r"[\U00010000-\U0010FFFF☀-➿️‍]", "", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for w in text.split():
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=fnt) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_text(draw, text, max_w, max_lines, start, weight="ExtraBold", min_size=28):
    """Tự giảm cỡ chữ cho vừa khung."""
    size = start
    while size > min_size:
        f = font(size, weight)
        lines = wrap(draw, text, f, max_w)
        if len(lines) <= max_lines:
            return f, lines
        size -= 4
    f = font(min_size, weight)
    return f, wrap(draw, text, f, max_w)[:max_lines]


def cover(img, w, h):
    """Cắt ảnh cho vừa khung (giống background-size: cover)."""
    r = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * r) + 1, int(img.height * r) + 1), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def designed_background(w, h, variant=0):
    """Nền thiết kế sẵn tông xanh – vàng (dùng khi không có ảnh AI)."""
    top = np.array(BRAND_DARK if variant % 2 == 0 else (22, 101, 52), dtype=float)
    bottom = np.array((6, 30, 18), dtype=float)
    t = np.linspace(0, 1, h)[:, None]
    grad = (top * (1 - t) + bottom * t).astype(np.uint8)
    img = Image.fromarray(np.repeat(grad[:, None, :], w, axis=1).reshape(h, w, 3))
    d = ImageDraw.Draw(img, "RGBA")
    s = min(w, h)
    d.ellipse((w - s * 0.55, -s * 0.2, w + s * 0.35, s * 0.7), fill=(*BRAND_YELLOW, 235))
    d.ellipse((-s * 0.3, h - s * 0.55, s * 0.45, h + s * 0.2), fill=(*BRAND_GREEN, 200))
    d.ellipse((w * 0.62, h * 0.42, w * 0.62 + s * 0.12, h * 0.42 + s * 0.12), fill=(255, 255, 255, 60))
    return img.filter(ImageFilter.GaussianBlur(2))


def fetch_pollinations(prompt, w, h, seed=7):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:400]) +
           f"?width={w}&height={h}&seed={seed}&model=flux&nologo=true")
    r = requests.get(url, timeout=120)
    if r.ok and r.headers.get("content-type", "").startswith("image"):
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise RuntimeError(f"Pollinations trả về {r.status_code}")


def fetch_gemini_image(prompt, model, aspect="1:1"):
    client = api_client()
    try:
        cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                          image_config=types.ImageConfig(aspect_ratio=aspect))
    except Exception:  # noqa: BLE001
        cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
    res = client.models.generate_content(model=model, contents=prompt, config=cfg)
    for part in res.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Gemini không trả về ảnh")


STYLE = (", professional advertising photography, vibrant colors, soft studio lighting, shallow depth of field, "
         "high detail, clean composition, no text, no letters, no watermark, no logo")


def get_background(prompt, w, h, seed=7, variant=0, log=None, allow_free_api=True, theme=None):
    """Ảnh nền tự động: Gemini Nano Banana (nếu key trả phí) → Pollinations (miễn phí) → nền gradient theo màu chủ đạo."""
    prompt = clean_text(prompt)[:300] + STYLE
    for m in pick(IMAGE_MODELS):
        try:
            img = fetch_gemini_image(prompt, m, "9:16" if h > w else "1:1")
            return cover(img, w, h), f"Gemini ({m})"
        except Exception as e:  # noqa: BLE001
            if any(x in str(e) for x in NO_QUOTA):
                st.session_state.free_key = True
                break
    if allow_free_api:
        try:
            return cover(fetch_pollinations(prompt, w, h, seed), w, h), "Pollinations (miễn phí)"
        except Exception as e:  # noqa: BLE001
            log and log(f"⚠️ Chưa lấy được ảnh AI miễn phí ({str(e)[:80]}) → dùng nền gradient.")
    if theme:
        return Animated(w, h, theme, None, variant).frame(1.5 + variant).convert("RGB"), "nền gradient"
    return designed_background(w, h, variant), "nền gradient"


def dark_fade(w, h, start=0.35, strength=235):
    """Lớp phủ tối dần xuống dưới để chữ luôn dễ đọc."""
    a = np.clip((np.linspace(0, 1, h) - start) / (1 - start), 0, 1) ** 1.3 * strength
    alpha = np.repeat(a[:, None], w, axis=1).astype(np.uint8)
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 3] = alpha
    return Image.fromarray(layer, "RGBA")


def draw_brand(d, x, y, brand, size):
    brand = clean_text(brand)
    if not brand:
        return
    f = font(size, "ExtraBold")
    d.text((x, y), brand.upper(), font=f, fill="white")
    tw = d.textlength(brand.upper(), font=f)
    r = size * 0.22
    d.ellipse((x + tw + 6, y + 2, x + tw + 6 + 2 * r, y + 2 + 2 * r), fill=BRAND_YELLOW)
    d.rounded_rectangle((x, y + size * 1.2, x + tw * 0.55, y + size * 1.2 + max(4, size // 7)),
                        radius=3, fill=BRAND_YELLOW)


def make_poster(bg, brief, brand, w, h):
    """Poster: ảnh nền + lớp phủ tối + chữ xếp từ dưới lên (không bao giờ tràn khung)."""
    th = theme_of(brief)
    img = bg.convert("RGBA")
    img.alpha_composite(dark_fade(w, h, 0.18 if h <= w else 0.3))
    d = ImageDraw.Draw(img)
    pad = int(w * 0.07)
    draw_brand(d, pad, pad, brand, int(w * 0.045))

    story = h > w * 1.3
    tag = clean_text(brief.get("san_pham", ""))[:40].upper()
    tf = font(int(w * 0.026), "Bold")
    hf, hl = fit_text(d, clean_text(brief.get("headline", "")), w - 2 * pad, 3, int(w * (0.1 if story else 0.08)))
    sf, sl = fit_text(d, clean_text(brief.get("subline", "")), w - 2 * pad, 2, int(w * (0.04 if story else 0.034)), "Medium", 20)
    cta = clean_text(brief.get("cta", "Mua ngay")).upper()
    cf = font(int(w * (0.042 if story else 0.036)), "ExtraBold")
    tags = " ".join(clean_text(t) for t in brief.get("hashtags", [])[:4])
    gf = font(int(w * 0.026), "Medium")

    chip_h = int(tf.size * 2.1) if tag else 0
    head_h = int(hf.size * 1.12) * len(hl)
    sub_h = int(sf.size * 1.4) * len(sl)
    cta_h = int(cf.size * 2.1)
    tags_h = int(gf.size * 1.6) if tags else 0
    gap = int(w * 0.03)
    block = chip_h + gap + head_h + gap // 2 + sub_h + gap + cta_h
    y = h - pad - tags_h - (gap if tags else 0) - block

    if tag:
        tw = d.textlength(tag, font=tf)
        d.rounded_rectangle((pad, y, pad + tw + 36, y + chip_h), radius=40, fill=th["accent2"])
        d.text((pad + 18, y + (chip_h - tf.size) // 2 - 2), tag, font=tf, fill=(17, 17, 17))
        y += chip_h + gap
    for ln in hl:
        d.text((pad, y), ln, font=hf, fill="white")
        y += int(hf.size * 1.12)
    y += gap // 2
    for ln in sl:
        d.text((pad, y), ln, font=sf, fill=(229, 231, 235))
        y += int(sf.size * 1.4)
    y += gap
    cw = d.textlength(cta, font=cf)
    d.rounded_rectangle((pad, y, pad + cw + 64, y + cta_h), radius=100, fill=th["accent"])
    d.text((pad + 32, y + (cta_h - cf.size) // 2 - 3), cta, font=cf, fill=(17, 17, 17))
    if tags:
        d.text((pad, h - pad - tags_h), tags, font=gf, fill=(187, 247, 208))
    return img.convert("RGB")


def to_png(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def make_voice(text, voice=VOICE):
    """Gemini TTS (miễn phí) → trả về (bytes WAV, số giây)."""
    client = api_client()
    tts = pick(TTS_MODELS, lambda n: "tts" in n)
    if not tts:
        raise RuntimeError("không tìm thấy model giọng đọc")
    res = client.models.generate_content(
        model=tts[0], contents=f"Đọc bằng giọng quảng cáo tươi vui, rõ ràng: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
    pcm = res.candidates[0].content.parts[0].inline_data.data
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return buf.getvalue(), len(pcm) / 2 / 24000


# ---------------------------------------------------------------------
# VIDEO "MOTION AD" MIỄN PHÍ: AI làm đạo diễn (storyboard) → code dựng chuyển động,
# số liệu chạy, chữ nảy từng từ, phụ đề karaoke, nhạc nền tự tổng hợp, giọng đọc Gemini TTS.
# ---------------------------------------------------------------------
THEMES = {
    "xanh_la": dict(bg1=(10, 60, 34), bg2=(2, 18, 10), accent=(250, 204, 21), accent2=(34, 197, 94)),
    "cam": dict(bg1=(154, 52, 18), bg2=(38, 10, 3), accent=(253, 224, 71), accent2=(251, 146, 60)),
    "tim": dict(bg1=(76, 29, 149), bg2=(17, 5, 38), accent=(244, 114, 182), accent2=(192, 132, 252)),
    "xanh_duong": dict(bg1=(12, 54, 112), bg2=(2, 10, 32), accent=(56, 189, 248), accent2=(250, 204, 21)),
    "do": dict(bg1=(153, 27, 27), bg2=(34, 4, 4), accent=(253, 224, 71), accent2=(251, 113, 133)),
    "den_vang": dict(bg1=(39, 39, 42), bg2=(0, 0, 0), accent=(250, 204, 21), accent2=(163, 230, 53)),
}
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/bevietnampro/BeVietnamPro-{w}.ttf"
_font_cache = {}


def font(size, weight="ExtraBold"):
    """Font tiếng Việt. Thiếu thư mục fonts trên GitHub thì tự tải về (sửa lỗi vỡ chữ)."""
    size = int(size)
    k = (size, weight)
    if k in _font_cache:
        return _font_cache[k]
    for folder in (FONT_DIR, Path(tempfile.gettempdir()) / "fs_fonts"):
        p = folder / f"BeVietnamPro-{weight}.ttf"
        if not p.exists() and folder != FONT_DIR:
            try:
                folder.mkdir(exist_ok=True)
                r = requests.get(FONT_URL.format(w=weight), timeout=20)
                if r.ok:
                    p.write_bytes(r.content)
            except Exception:  # noqa: BLE001
                pass
        if p.exists():
            try:
                _font_cache[k] = ImageFont.truetype(str(p), size)
                return _font_cache[k]
            except Exception:  # noqa: BLE001
                pass
    _font_cache[k] = ImageFont.load_default(size)
    return _font_cache[k]


def ease_back(t):
    t = max(0.0, min(1.0, t))
    c1 = 1.70158
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def theme_of(brief):
    return THEMES.get(str(brief.get("mau_chu_dao", "")).strip(), THEMES["xanh_la"])


class Animated:
    """Nền động: ảnh AI (zoom chậm + phủ màu chủ đạo) hoặc gradient với các đốm sáng trôi."""

    def __init__(self, w, h, theme, image=None, seed=0):
        self.w, self.h, self.th, self.seed = w, h, theme, seed
        t = np.linspace(0, 1, h)[:, None, None]
        grad = np.array(theme["bg1"], float) * (1 - t) + np.array(theme["bg2"], float) * t
        self.base = np.repeat(grad, w, axis=1).astype(np.float32)
        self.img = image.convert("RGB").resize((int(w * 1.16), int(h * 1.16)), Image.LANCZOS) if image else None
        if self.img is not None:
            tint = Image.fromarray(self.base.astype(np.uint8))
            self.tint = tint.convert("RGBA")
            self.tint.putalpha(115)
        fade = np.clip((np.linspace(0, 1, h) - 0.35) / 0.65, 0, 1) ** 1.2 * 200
        self.fade = Image.fromarray(np.dstack([np.zeros((h, w, 3), np.uint8),
                                               np.repeat(fade[:, None], w, 1).astype(np.uint8)]), "RGBA")

    def frame(self, tt, punch=0.0):
        w, h = self.w, self.h
        if self.img is not None:
            z = 1.0 + 0.10 * min(1, tt / 5) + punch
            cw, ch = self.img.width / z, self.img.height / z
            left = (self.img.width - cw) / 2 + 20 * math.sin(tt * 0.6 + self.seed)
            top = (self.img.height - ch) / 2
            f = self.img.crop((int(left), int(top), int(left + cw), int(top + ch))).resize((w, h), Image.BILINEAR)
            f = f.convert("RGBA")
            f.alpha_composite(self.tint)
        else:
            sw, sh = w // 6, h // 6
            small = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
            d = ImageDraw.Draw(small)
            for i, col in enumerate([self.th["accent2"], self.th["accent"], self.th["accent2"]]):
                cx = sw * (0.5 + 0.42 * math.sin(tt * (0.45 + 0.1 * i) + i * 2.1 + self.seed))
                cy = sh * (0.35 + 0.35 * math.cos(tt * (0.35 + 0.07 * i) + i * 1.3))
                r = sw * (0.34 - 0.06 * i)
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*col, 120 - 25 * i))
            glow = small.filter(ImageFilter.GaussianBlur(9)).resize((w, h), Image.BILINEAR)
            f = Image.fromarray(self.base.astype(np.uint8)).convert("RGBA")
            f.alpha_composite(glow)
        f.alpha_composite(self.fade)
        return f


def text_img(text, fnt, fill, pad=0, bg=None, radius=18):
    """Chữ thành ảnh RGBA riêng (chiều cao theo font nên các từ luôn thẳng hàng)."""
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    asc, desc = fnt.getmetrics()
    W = int(d0.textlength(text, font=fnt)) + 2 * pad + 8
    H = asc + desc + 2 * pad + 6
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bg:
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=radius, fill=bg)
    else:
        d.text((pad + 5, pad + 5), text, font=fnt, fill=(0, 0, 0, 110))                 # bóng đổ
    d.text((pad + 4, pad + 3), text, font=fnt, fill=fill)
    return im


def paste_scaled(canvas, im, cx, cy, scale=1.0, alpha=1.0):
    if scale <= 0.02 or alpha <= 0.01:
        return
    if abs(scale - 1) > 0.01:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 0.99:
        a = np.array(im.getchannel("A"), np.float32) * alpha
        im = im.copy()
        im.putalpha(Image.fromarray(a.astype(np.uint8)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def words_layout(text, fnt, max_w, space):
    """Chia từ thành các dòng vừa khung, trả về [(dòng, [ (từ, rộng) ])]."""
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines, cur, cur_w = [], [], 0
    for wd in text.split():
        ww = d0.textlength(wd, font=fnt)
        if cur and cur_w + space + ww > max_w:
            lines.append(cur)
            cur, cur_w = [], 0
        cur.append((wd, ww))
        cur_w += (space if len(cur) > 1 else 0) + ww
    if cur:
        lines.append(cur)
    return lines


class Scene:
    def __init__(self, sc, brief, th, w, h, dur, voice_line):
        self.sc, self.brief, self.th, self.w, self.h, self.dur = sc, brief, th, w, h, dur
        self.type = str(sc.get("loai", "hook")).lower()
        self.text = clean_text(sc.get("text", ""))
        self.voice = clean_text(voice_line)
        hl = clean_text(sc.get("nhan_manh", "")).lower()
        self.hl = set(hl.split()) if hl else set()
        self.pre = {}
        self._prepare()

    # --- chuẩn bị các mảnh chữ 1 lần ---
    def _prepare(self):
        w, th = self.w, self.th
        maxw = w * 0.84
        size = 88
        while size > 46:
            f = font(size)
            lines = words_layout(self.text, f, maxw, size * 0.28)
            if len(lines) <= (3 if self.type in ("hook", "cta") else 2):
                break
            size -= 6
        self.fsize, self.lines = size, words_layout(self.text, font(size), maxw, size * 0.28)
        self.word_imgs = []
        for line in self.lines:
            row = []
            for wd, ww in line:
                key = re.sub(r"[^\w%]", "", wd.lower())
                is_hl = key in self.hl or (key and any(key == re.sub(r"[^\w%]", "", x) for x in self.hl))
                im = text_img(wd, font(size), (17, 17, 17) if is_hl else (255, 255, 255), pad=10 if is_hl else 0,
                              bg=(*th["accent"], 255) if is_hl else None, radius=14)
                row.append(im)
            self.word_imgs.append(row)
        kicker = clean_text(self.sc.get("kicker", ""))
        self.kicker = text_img(kicker.upper(), font(30, "Bold"), (17, 17, 17), pad=14, bg=(*th["accent2"], 255),
                               radius=30) if kicker else None
        if self.type == "stat":
            raw = clean_text(self.sc.get("so_lieu", "")) or "100%"
            m = re.search(r"[-+]?\d+(?:[.,]\d+)?", raw)
            self.num_val = float(m.group(0).replace(",", ".")) if m else 0
            self.num_dec = len(m.group(0).split(",")[-1].split(".")[-1]) if m and re.search(r"[.,]", m.group(0)) else 0
            self.num_pre, self.num_suf = (raw[:m.start()], raw[m.end():]) if m else ("", raw)
            self.is_pct = "%" in raw
            d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            self.num_size = 150
            while self.num_size > 60 and d0.textlength(raw, font=font(self.num_size)) > self.w * 0.5:
                self.num_size -= 8
        if self.type == "benefits":
            items = [clean_text(x) for x in self.sc.get("y", [])][:3] or [self.text]
            self.items = [text_img(it, font(42, "Bold"), (255, 255, 255)) for it in items]
        if self.type == "offer":
            self.badge_txt = clean_text(self.sc.get("so_lieu", "")) or "HOT"
        if self.type == "cta":
            cta = clean_text(self.brief.get("cta", "Mua ngay")).upper()
            self.btn = text_img(cta, font(50), (17, 17, 17), pad=26, bg=(*th["accent"], 255), radius=60)
            tags = " ".join(clean_text(t) for t in self.brief.get("hashtags", [])[:3])
            self.tags = text_img(tags, font(30, "Medium"), (220, 252, 231)) if tags else None
            name = clean_text(self.brief.get("thuong_hieu") or self.brief.get("san_pham", "")).upper()
            self.name = text_img(name[:28], font(40, "Bold"), (*th["accent"],)) if name else None

    # --- vẽ 1 khung hình ---
    def draw(self, cv, t):
        w, h, th = self.w, self.h, self.th
        d = ImageDraw.Draw(cv)
        top_y = h * (0.24 if self.type in ("benefits",) else 0.36)
        if self.type == "stat":
            top_y = h * 0.66
        lh = self.fsize * (1.32 if self.hl else 1.18)
        if self.kicker is not None and self.type not in ("stat",):
            k = ease_back(t / 0.35)
            ky = top_y - (len(self.lines) - 1) * lh / 2 - lh / 2 - 60
            paste_scaled(cv, self.kicker, w / 2, ky, 0.6 + 0.4 * k, min(1, t / 0.2))

        if self.type == "stat":                                       # số liệu chạy + vòng tròn %
            cx, cy, R = w / 2, h * 0.34, w * 0.32
            p = ease_out(t / 1.2)
            if self.is_pct:
                frac = min(1.0, self.num_val / 100) * p
                d.arc((cx - R, cy - R, cx + R, cy + R), 0, 360, fill=(255, 255, 255, 50), width=22)
                d.arc((cx - R, cy - R, cx + R, cy + R), -90, -90 + 360 * frac, fill=th["accent"], width=22)
            val = self.num_val * p
            s = f"{val:,.{self.num_dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            num = text_img(f"{self.num_pre}{s}{self.num_suf}", font(self.num_size), th["accent"])
            paste_scaled(cv, num, cx, cy, 0.85 + 0.15 * ease_back(t / 0.5))
            if self.kicker is not None:
                paste_scaled(cv, self.kicker, w / 2, h * 0.12, ease_back(t / 0.35))

        if self.type == "offer":                                      # huy hiệu xoay + nhịp đập
            cx, cy = w / 2, h * 0.30
            R = w * 0.27 * (1 + 0.04 * math.sin(t * 7)) * ease_back(t / 0.45)
            pts = []
            for i in range(36):
                ang = t * 0.8 + i * math.pi / 18
                rr = R if i % 2 == 0 else R * 0.84
                pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
            if R > 5:
                d.polygon(pts, fill=th["accent"])
                bt = text_img(self.badge_txt, font(96 if len(self.badge_txt) < 5 else 70), (17, 17, 17))
                paste_scaled(cv, bt, cx, cy, ease_back(t / 0.5))
            top_y = h * 0.56

        if self.type == "benefits":                                   # tiêu đề + 3 ý trượt vào
            y0 = h * 0.47
            for i, im in enumerate(self.items):
                p = ease_out((t - 0.45 - i * 0.4) / 0.45)
                if p <= 0:
                    continue
                x = w * 0.18 - (1 - p) * 260
                y = y0 + i * 150
                r = 30
                d.ellipse((x - 70 - r, y - r, x - 70 + r, y + r), fill=th["accent2"])
                d.line([(x - 84, y), (x - 73, y + 12), (x - 55, y - 12)], fill=(17, 17, 17), width=7)
                paste_scaled(cv, im, x + im.width / 2 - 20, y, 1, p)

        # tiêu đề: từng từ nảy lên
        n_words = sum(len(r) for r in self.word_imgs)
        start_y = top_y - (len(self.lines) - 1) * lh / 2
        idx = 0
        for li, (line, row) in enumerate(zip(self.lines, self.word_imgs)):
            gap = self.fsize * 0.28
            tot = sum(im.width for im in row) + gap * (len(row) - 1)
            x = (w - tot) / 2
            for im in row:
                delay = 0.08 + idx * min(0.12, 0.9 / max(1, n_words))
                k = ease_back((t - delay) / 0.28)
                paste_scaled(cv, im, x + im.width / 2, start_y + li * lh, 0.4 + 0.6 * k, min(1, max(0, (t - delay) / 0.12)))
                x += im.width + gap
                idx += 1

        if self.type == "cta":                                        # nút bấm đập + mũi tên + hashtag
            if self.name is not None:
                paste_scaled(cv, self.name, w / 2, h * 0.16, 1, min(1, t / 0.3))
            s = ease_back((t - 0.5) / 0.4) * (1 + 0.05 * math.sin(t * 6))
            by = h * 0.62
            paste_scaled(cv, self.btn, w / 2, by, s)
            ay = by + 120 + 14 * math.sin(t * 5)
            if t > 0.8:
                d.polygon([(w / 2 - 24, ay), (w / 2 + 24, ay), (w / 2, ay - 30)], fill=th["accent"])
            if self.tags is not None:
                paste_scaled(cv, self.tags, w / 2, h * 0.86, 1, min(1, max(0, (t - 0.9) / 0.4)))

        if self.voice and self.type != "cta":                         # phụ đề karaoke
            self._caption(cv, t)

    def _caption(self, cv, t):
        w, h, th = self.w, self.h, self.th
        f = font(34, "Bold")
        words = self.voice.split()
        cur = int(min(len(words) - 1, max(0, t / max(0.1, self.dur * 0.95)) * len(words)))
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lines, line, lw = [], [], 0
        for i, wd in enumerate(words):
            ww = d0.textlength(wd + " ", font=f)
            if line and lw + ww > w * 0.82:
                lines.append(line)
                line, lw = [], 0
            line.append((i, wd, ww))
            lw += ww
        if line:
            lines.append(line)
        li = next((k for k, ln in enumerate(lines) if any(i == cur for i, _, _ in ln)), 0)
        show = lines[max(0, li - 1): li + 1] if li > 0 else lines[:2]
        box_h = 58 * len(show) + 26
        y0 = h * 0.905 - box_h
        d = ImageDraw.Draw(cv)
        d.rounded_rectangle((w * 0.06, y0, w * 0.94, y0 + box_h), radius=22, fill=(0, 0, 0, 150))
        for k, ln in enumerate(show):
            tot = sum(ww for _, _, ww in ln)
            x = (w - tot) / 2
            for i, wd, ww in ln:
                col = th["accent"] if i == cur else ((255, 255, 255) if i < cur else (200, 200, 200))
                d.text((x, y0 + 16 + k * 58), wd, font=f, fill=col)
                x += ww


def synth_music(sec, sr=24000, bpm=112, seed=0):
    """Nhạc nền tự tổng hợp (không bản quyền): trống + bass + hợp âm nhẹ."""
    n = int(sec * sr)
    out = np.zeros(n, np.float32)
    beat = 60 / bpm
    tt = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    kick_len = int(0.22 * sr)
    kt = np.arange(kick_len) / sr
    kick = np.sin(2 * np.pi * (45 * kt + 70 * (1 - np.exp(-kt * 25)) / 25 * 25 / 25)) * np.exp(-kt * 16)
    hat = rng.normal(0, 1, int(0.05 * sr)).astype(np.float32)
    hat = np.diff(hat, prepend=0) * np.exp(-np.arange(len(hat)) / sr * 90)
    chords = [(261.6, 329.6, 392.0), (196.0, 246.9, 293.7), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6)]
    k = 0
    while k * beat < sec:
        s = int(k * beat * sr)
        e = min(n, s + kick_len)
        out[s:e] += 0.9 * kick[: e - s]
        hs = int((k + 0.5) * beat * sr)
        he = min(n, hs + len(hat))
        if hs < n:
            out[hs:he] += 0.18 * hat[: he - hs]
        k += 1
    bar = 4 * beat
    for b in range(int(sec / bar) + 1):
        ch = chords[b % 4]
        s, e = int(b * bar * sr), min(n, int((b + 1) * bar * sr))
        if s >= n:
            break
        seg = tt[s:e] - tt[s]
        env = np.minimum(1, seg / 0.4) * np.exp(-seg * 0.35)
        pad = sum(np.sin(2 * np.pi * f * seg) for f in ch) / 3
        bass = np.sin(2 * np.pi * ch[0] / 2 * seg) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * seg / beat * 2)))
        out[s:e] += 0.16 * pad * env + 0.22 * bass * np.exp(-(seg % beat) * 5)
    fade = np.minimum(1, np.minimum(tt / 0.6, (sec - tt) / 1.0))
    out *= np.clip(fade, 0, 1)
    return out / (np.abs(out).max() + 1e-6) * 0.9


def pcm_from_wav(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768, wf.getframerate()


def wav_from_pcm(x, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def storyboard(brief):
    """Lấy danh sách cảnh từ brief (AI viết); thiếu thì tự tạo từ headline."""
    scenes = [s for s in brief.get("scenes", []) if isinstance(s, dict) and (s.get("text") or s.get("so_lieu"))]
    if not scenes:
        scenes = [{"loai": "hook", "text": brief.get("headline", "")}]
    if not any(str(s.get("loai")) == "cta" for s in scenes):
        scenes.append({"loai": "cta", "text": brief.get("headline", ""), "voice": ""})
    return scenes[:6]


def make_video(backgrounds, brief, brand=None, voice_wav=None, voice_sec=0.0, w=720, h=1280, fps=24):
    """Dựng video quảng cáo dọc 9:16 có chuyển động, phụ đề, nhạc nền + giọng đọc."""
    import imageio_ffmpeg

    th = theme_of(brief)
    scenes = storyboard(brief)
    base_dur = {"hook": 3.0, "stat": 3.4, "benefits": 4.0, "offer": 3.2, "cta": 3.4}
    lines = [clean_text(s.get("voice", "")) for s in scenes]
    if voice_wav and voice_sec > 1:
        weights = [max(4, len(x.split())) for x in lines]
        total = voice_sec + 1.2
        durs = [max(2.4, total * wt / sum(weights)) for wt in weights]
    else:
        durs = [base_dur.get(str(s.get("loai", "hook")), 3.0) for s in scenes]
    objs = [Scene(s, brief, th, w, h, durs[i], lines[i]) for i, s in enumerate(scenes)]
    imgs, bgs = list(backgrounds or []), []
    for i, s in enumerate(scenes):                 # ảnh AI cho cảnh mở đầu / lợi ích, cảnh số liệu – ưu đãi – chốt dùng nền màu
        img = imgs.pop(0) if imgs and s.get("loai") in ("hook", "benefits") else None
        bgs.append(Animated(w, h, th, img, i))

    tmp = Path(tempfile.mkdtemp())
    silent, final = tmp / "silent.mp4", tmp / "final.mp4"
    writer = imageio_ffmpeg.write_frames(str(silent), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         quality=7, macro_block_size=1)
    writer.send(None)
    total = sum(durs)
    elapsed = 0.0
    for i, (sc, bg) in enumerate(zip(objs, bgs)):
        nf = int(durs[i] * fps)
        for k in range(nf):
            t = k / fps
            punch = 0.06 * (1 - ease_out(t / 0.25)) if i > 0 else 0
            cv = bg.frame(elapsed + t, punch)
            sc.draw(cv, t)
            d = ImageDraw.Draw(cv)
            d.rounded_rectangle((36, 28, 36 + (w - 72) * ((elapsed + t) / total), 36), radius=4, fill=th["accent"])
            if i > 0 and t < 0.16:                                      # chớp sáng khi chuyển cảnh
                cv.alpha_composite(Image.new("RGBA", (w, h), (255, 255, 255, int(150 * (1 - t / 0.16)))))
            writer.send(np.asarray(cv.convert("RGB")).tobytes())
        elapsed += durs[i]
    writer.close()

    music = synth_music(total, seed=len(brief.get("headline", "")))
    if voice_wav:
        v, sr = pcm_from_wav(voice_wav)
        v = np.concatenate([np.zeros(int(0.35 * sr), np.float32), v])[: len(music)]
        v = np.pad(v, (0, len(music) - len(v)))
        env = np.convolve(np.abs(v), np.ones(2400) / 2400, mode="same")
        mix = v * 1.0 + music * 0.28 * (1 - np.clip(env * 6, 0, 0.65))
    else:
        mix = music * 0.7
    wav = tmp / "audio.wav"
    wav.write_bytes(wav_from_pcm(mix / max(1.0, np.abs(mix).max()), 24000))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
    return final.read_bytes()


def make_brief(strategy, content, summary=""):
    prompt = PROMPT_BRIEF.format(summary=summary[:4000], strategy=strategy[:5000], content=content[:5000])
    for _ in range(2):
        text, _m = ask_gemini(prompt, json_mode=True)
        try:
            data = json.loads(re.sub(r"^```(json)?|```$", "", text.strip()))
            return data[0] if isinstance(data, list) else data
        except json.JSONDecodeError:
            continue
    raise RuntimeError("AI chưa trả về brief đúng định dạng, bấm tạo lại nhé.")


def make_ai_video(prompt):
    """Video AI thật bằng Gemini Omni (chỉ chạy khi key đã bật thanh toán)."""
    if VIDEO_MODEL not in pick([VIDEO_MODEL]):
        return None
    try:
        client = api_client()
        it = client.interactions.create(model=VIDEO_MODEL, input=prompt,
                                        response_format={"aspect_ratio": "9:16", "resolution": "720p"})
        data = it.output_video.data
        return base64.b64decode(data) if isinstance(data, str) else data
    except Exception as e:  # noqa: BLE001
        if any(x in str(e) for x in NO_QUOTA):
            st.session_state.free_key = True
        return None


def short_num(v):
    """35445705429 → 35.45 tỷ ; 825894 → 825,894"""
    for unit, name in ((1e9, " tỷ"), (1e6, " triệu")):
        if abs(v) >= unit:
            return f"{v / unit:,.2f}{name}"
    return f"{v:,.0f}"


# =====================================================================
# 5. GIAO DIỆN – ưu tiên dễ đọc
# =====================================================================
GREEN, GREEN_DARK, YELLOW, INK, MUTED = "#16A34A", "#0F3D24", "#FACC15", "#111827", "#4B5563"
CHART_COLORS = ["#16A34A", "#FACC15", "#0F3D24", "#86EFAC", "#F59E0B", "#4ADE80", "#9CA3AF", "#15803D"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
.stApp *:not([translate="no"]):not([data-testid*="Icon"]) {{ font-family: 'Be Vietnam Pro', sans-serif; }}
.stApp {{ background: #F5F8F5; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], footer, #MainMenu {{ visibility: hidden; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1140px; }}
.stApp p, .stApp li {{ font-size: 16px; line-height: 1.65; color: {INK}; }}

/* header */
.fs-bar {{ background: {GREEN_DARK}; color: #fff; border-radius: 16px; padding: 10px 20px; font-size: 14px;
  font-weight: 600; display: flex; justify-content: space-between; align-items: center; }}
.fs-bar .live {{ color: {YELLOW}; font-weight: 800; letter-spacing: 1px; }}
.fs-bar .live::before {{ content: "● "; animation: blink 1.4s infinite; }}
@keyframes blink {{ 50% {{ opacity: .3; }} }}
.fs-logo {{ font-size: 28px; font-weight: 800; letter-spacing: -.5px; color: {INK}; line-height: 1; }}
.fs-logo i {{ margin-left: 4px; }}
.fs-logo span {{ color: #22C55E; }}
.fs-logo i {{ display: inline-block; width: 10px; height: 10px; background: {YELLOW}; border-radius: 50%; vertical-align: top; }}
.fs-logo-sub {{ font-size: 12px; font-weight: 700; letter-spacing: 2px; color: {MUTED}; margin-top: 6px; }}
.fs-status {{ display: inline-block; border-radius: 999px; padding: 6px 14px; font-size: 13.5px; font-weight: 700; }}
.fs-status.on {{ background: #DCFCE7; color: #166534; }}
.fs-status.off {{ background: #FEF3C7; color: #92400E; }}

/* hero */
.fs-hero {{ background: #fff; border: 1px solid #E5E7EB; border-radius: 24px; padding: 30px 34px; margin-top: 14px; }}
.fs-hero h1 {{ font-size: 38px !important; line-height: 1.2 !important; font-weight: 800 !important; color: {INK};
  letter-spacing: -1px; margin: 0 0 10px !important; padding: 0 !important; }}
.fs-hero h1 em {{ font-style: normal; color: {GREEN}; background: linear-gradient(transparent 70%, #FEF08A 70%); }}
.fs-hero p {{ font-size: 17px !important; color: {MUTED} !important; margin: 0 !important; max-width: 720px; }}

/* stepper */
.fs-steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 22px; }}
.fs-step {{ border-radius: 16px; padding: 14px 16px; border: 1.5px solid #E5E7EB; background: #F9FAFB; }}
.fs-step .n {{ font-size: 12.5px; font-weight: 800; letter-spacing: 1px; color: #9CA3AF; }}
.fs-step .t {{ font-size: 15.5px; font-weight: 700; color: {INK}; margin-top: 2px; }}
.fs-step.done {{ background: #F0FDF4; border-color: #86EFAC; }}
.fs-step.done .n {{ color: {GREEN}; }}
.fs-step.now {{ background: #FFFBEB; border-color: {YELLOW}; box-shadow: 0 6px 18px -10px rgba(250,204,21,.9); }}
.fs-step.now .n {{ color: #B45309; }}
@media (max-width: 760px) {{ .fs-steps {{ grid-template-columns: repeat(2, 1fr); }} .fs-hero h1 {{ font-size: 30px !important; }} }}

/* tabs dạng nút */
[role="tablist"] {{ gap: 6px; background: #E9EFEA; padding: 6px; border-radius: 999px; width: fit-content;
  margin: 22px 0 6px; border: none !important; }}
[data-testid="stTab"] {{ border-radius: 999px; padding: 9px 22px; }}
[data-testid="stTab"] p {{ font-weight: 700 !important; font-size: 15px !important; }}
[data-testid="stTab"][aria-selected="true"] {{ background: {INK}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #fff !important; }}
.react-aria-SelectionIndicator {{ display: none !important; }}

/* khung trắng cho từng bước */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 22px !important; background: #fff; border-color: #E5E7EB !important; }}
.fs-sec {{ display: flex; gap: 14px; align-items: center; margin: 4px 0 14px; }}
.fs-sec .num {{ background: {GREEN}; color: #fff; font-weight: 800; border-radius: 12px; min-width: 40px; height: 40px;
  display: flex; align-items: center; justify-content: center; font-size: 15px; }}
.fs-sec .t {{ font-size: 21px; font-weight: 800; color: {INK}; line-height: 1.2; }}
.fs-sec .s {{ font-size: 14.5px; color: {MUTED}; }}

/* KPI */
.fs-kpi {{ background: #fff; border: 1px solid #E5E7EB; border-radius: 18px; padding: 16px 18px; height: 100%; }}
.fs-kpi .l {{ font-size: 13px; font-weight: 700; color: {MUTED}; text-transform: capitalize; }}
.fs-kpi .v {{ font-size: 28px; font-weight: 800; color: {INK}; margin: 4px 0 8px; letter-spacing: -.5px; }}
.fs-kpi .d {{ display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 13px; font-weight: 700; }}
.fs-kpi .up {{ background: #DCFCE7; color: #166534; }} .fs-kpi .down {{ background: #FEE2E2; color: #991B1B; }}
.fs-kpi.main {{ border: 2px solid {GREEN}; background: #F0FDF4; }}

/* upload, nút, expander */
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed #86EFAC; background: #F7FEF9; border-radius: 18px; }}
.stButton > button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button {{
  border-radius: 999px !important; font-weight: 700 !important; padding: 10px 22px !important; }}
.stButton > button[kind="primary"] {{ background: {GREEN}; border: none; color: #fff; font-size: 16px;
  box-shadow: 0 10px 20px -10px rgba(22,163,74,.9); }}
.stButton > button[kind="primary"]:hover {{ background: #15803D; color: #fff; }}
[data-testid="stDownloadButton"] button {{ background: {YELLOW} !important; border: none !important; color: {INK} !important; }}
[data-testid="stExpander"] details {{ border-radius: 14px; border-color: #E5E7EB; }}
[data-testid="stChatInput"] {{ border-radius: 999px; border: 1.5px solid #86EFAC; }}
.fs-hint {{ display: inline-block; background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; border-radius: 999px;
  padding: 6px 14px; margin: 0 6px 8px 0; font-size: 14px; font-weight: 600; }}
.fs-media-t {{ font-weight: 800; font-size: 15px; color: {INK}; margin-bottom: 6px; }}
iframe[title*="streamlit_js_eval"] {{ height: 0 !important; border: 0; display: block; }}
.fs-footer {{ text-align: center; color: #9CA3AF; font-size: 13px; margin-top: 40px; }}
</style>
"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def section(num, title, sub=""):
    html(f'<div class="fs-sec"><div class="num">{num}</div><div><div class="t">{title}</div>'
         f'<div class="s">{sub}</div></div></div>')


def style_fig(fig, title):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color=INK), x=0.01),
        font=dict(family="Be Vietnam Pro, sans-serif", color="#374151", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15), colorway=CHART_COLORS)
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#EEF2EF", title=None)
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
        fig = px.area(d["trend"].reset_index(), x="kỳ", y=main, markers=True)
        fig.update_traces(line=dict(color=GREEN, width=3), fillcolor="rgba(22,163,74,.10)",
                          marker=dict(color=YELLOW, size=7, line=dict(color=GREEN_DARK, width=1.5)))
        st.plotly_chart(style_fig(fig, f"Xu hướng {main.replace('_', ' ')} theo {d['time_col']}"), width="stretch")
    chart_cols = st.columns(2)
    for i, dim in enumerate(d["dims"][:4]):
        t = d["by_dim"][dim]
        y = main or "số dòng"
        if len(t) <= 6:
            fig = px.pie(t, names=dim, values=y, hole=0.58, color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#fff", width=3)))
        else:
            fig = px.bar(t, x=dim, y=y, color_discrete_sequence=[GREEN])
        chart_cols[i % 2].plotly_chart(style_fig(fig, f"{y.replace('_', ' ')} theo {dim.replace('_', ' ')}"),
                                       width="stretch")
    c1, c2 = st.columns(2)
    with c1.expander("🔎 Cách agent hiểu các cột dữ liệu"):
        st.dataframe(pd.DataFrame({"cột": list(d["info"]), "loại": list(d["info"].values())}), hide_index=True)
    with c2.expander("📝 Bản tóm tắt gửi cho AI"):
        st.code(d["summary"], language=None)


LS_KEY = "foodsave_ai_chats_v1"


def load_chats():
    """Đọc lịch sử chat đã lưu trong trình duyệt (localStorage) – mỗi người dùng chỉ thấy lịch sử của mình."""
    if st.session_state.get("chats_loaded"):
        return
    raw = streamlit_js_eval(js_expressions=f"localStorage.getItem('{LS_KEY}') || '{{}}'", key="ls_load")
    if raw is None:                       # trình duyệt chưa trả lời, lần chạy sau sẽ có
        return
    try:
        saved = json.loads(raw)
    except Exception:  # noqa: BLE001
        saved = {}
    if isinstance(saved, dict):
        saved.update(st.session_state.get("chats", {}))
        st.session_state.chats = saved
    st.session_state.chats_loaded = True


def save_chats():
    """Ghi lịch sử chat vào trình duyệt (chỉ khi có thay đổi)."""
    if not st.session_state.pop("need_save", False):
        return
    st.session_state.save_n = st.session_state.get("save_n", 0) + 1
    data = dict(sorted(st.session_state.get("chats", {}).items(), reverse=True)[:40])   # giữ 40 cuộc gần nhất
    payload = json.dumps(json.dumps(data, ensure_ascii=False))
    streamlit_js_eval(js_expressions=f"localStorage.setItem('{LS_KEY}', {payload})",
                      key=f"ls_save_{st.session_state.save_n}")


# ---------------- khung trang ----------------
html(CSS)
api_key = get_api_key()
ready, api_error = False, ""
if api_key:
    try:
        ready = bool(text_models())
        if not ready:
            api_error = "Key hợp lệ nhưng không tìm thấy model Gemini viết chữ nào."
    except Exception as e:  # noqa: BLE001
        api_error = explain_error(e)

html('<div class="fs-bar"><span>Excel → Dashboard → Chiến lược → Content → Ảnh & Video · tự động 100%</span>'
     '<span class="live">LIVE</span></div>')
html('<div style="margin-top:16px"><div class="fs-logo">FOOD<span>SAVE</span><i></i></div>'
     '<div class="fs-logo-sub">AI MARKETING AGENT</div></div>')

model_now = st.session_state.get("last_model") or (text_models()[0] if ready else "")
if ready:
    status = f'<span class="fs-status on">● AI sẵn sàng · {model_now}</span>'
elif api_key:
    status = (f'<span class="fs-status off">● Không kết nối được Gemini – {api_error} '
              f'(key đang dùng: {api_key[:6]}…{api_key[-4:]}, dài {len(api_key)} ký tự)</span>')
else:
    status = '<span class="fs-status off">● Chưa cấu hình GEMINI_API_KEY trong Secrets</span>'

has_file = st.session_state.get("data_file") is not None
flags = [has_file, has_file, bool(st.session_state.get("strategy")), bool(st.session_state.get("media"))]
names = ["Tải dữ liệu", "Dashboard tự động", "AI chiến lược & content", "Ảnh & video quảng cáo"]
first_todo = next((i for i, f in enumerate(flags) if not f), 4)
steps = "".join(
    f'<div class="fs-step {"done" if flags[i] else ("now" if i == first_todo else "")}">'
    f'<div class="n">{"✓ XONG" if flags[i] else f"BƯỚC {i + 1}"}</div><div class="t">{names[i]}</div></div>'
    for i in range(4))
html(f'<div class="fs-hero">{status}<h1 style="margin-top:14px !important">Dữ liệu vào. <em>Chiến lược</em> ra. '
     f'Quảng cáo <em>sẵn sàng</em>.</h1>'
     f'<p>Chỉ cần tải file Excel — AI tự dựng dashboard, rút insight, chọn kênh quảng cáo, '
     f'viết content và tạo luôn poster, video. Không cần cài đặt gì.</p><div class="fs-steps">{steps}</div></div>')

tab_agent, tab_chat = st.tabs(["📊 Marketing Agent", "💬 Hỏi đáp AI"])

# ---------------- TAB 1 ----------------
with tab_agent:
    with st.container(border=True):
        section("01", "Tải dữ liệu", "File .xlsx, .xls hoặc .csv — dòng đầu tiên là tên cột")
        file = st.file_uploader("Tải file dữ liệu", type=["xlsx", "xls", "csv"], key="data_file",
                                label_visibility="collapsed")
        df = None
        if file:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                sheets = pd.ExcelFile(file).sheet_names
                sheet = st.selectbox("Chọn sheet", sheets) if len(sheets) > 1 else sheets[0]
                df = pd.read_excel(file, sheet_name=sheet)
            st.success(f"Đã đọc **{len(df):,}** dòng × **{df.shape[1]}** cột")
            with st.expander("Xem dữ liệu gốc"):
                st.dataframe(df.head(200))

    if df is not None:
        with st.container(border=True):
            section("02", "Dashboard tự động", "Agent tự nhận diện cột, tính chỉ số và vẽ biểu đồ")
            dash = build_dashboard(df)
            st.session_state.dash = dash
            show_dashboard(dash)

        with st.container(border=True):
            section("03", "AI phân tích & viết content", "Gemini đọc dashboard → rút insight → tự chọn kênh → viết bài")
            if st.button("🚀  Chạy AI Marketing Agent", type="primary", disabled=not ready):
                try:
                    with st.status("AI đang làm việc...", expanded=True) as s_:
                        st.write("🧠 Đang đọc dashboard, rút insight và chọn chiến lược...")
                        strategy, used1 = ask_gemini(PROMPT_STRATEGY.format(summary=dash["summary"]))
                        st.write(f"✅ Xong chiến lược (model: {used1})")
                        st.write("✍️ Đang viết content cho các kênh đã chọn...")
                        content, used2 = ask_gemini(PROMPT_CONTENT.format(strategy=strategy))
                        st.write(f"✅ Xong content (model: {used2})")
                        s_.update(label="Hoàn tất!", state="complete", expanded=False)
                    st.session_state.strategy, st.session_state.content = strategy, content
                    st.session_state.pop("media", None)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
            if st.session_state.get("strategy"):
                t1, t2 = st.tabs(["🧠 Insight & Chiến lược", "✍️ Content quảng cáo"])
                t1.markdown(st.session_state.strategy)
                t2.markdown(st.session_state.content)
                report = f"# Báo cáo AI Marketing Agent\n\n## Dashboard\n```\n{dash['summary']}\n```\n\n" \
                         f"## Insight & Chiến lược\n{st.session_state.strategy}\n\n## Content quảng cáo\n{st.session_state.content}\n"
                st.download_button("⬇️  Tải báo cáo (.md)", report, file_name="bao_cao_marketing.md")

        if st.session_state.get("strategy"):
            with st.container(border=True):
                section("04", "Ảnh & video quảng cáo", "AI tự viết brief → tạo poster Facebook/TikTok và video dọc có giọng đọc")
                if st.button("🎬  Tạo ảnh & video quảng cáo", type="primary", disabled=not ready):
                    try:
                        with st.status("Đang sản xuất quảng cáo...", expanded=True) as s_:
                            log = st.write
                            log("📝 AI đang viết brief sáng tạo (tiêu đề, CTA, kịch bản cảnh)...")
                            brief = make_brief(st.session_state.strategy, st.session_state.content,
                                               st.session_state.dash["summary"])
                            brand = clean_text(brief.get("thuong_hieu", ""))
                            th = theme_of(brief)
                            base = brief.get("image_prompt") or "premium product on a clean studio background"
                            log(f"🎨 Màu chủ đạo AI chọn: {brief.get('mau_chu_dao', 'xanh_la')}")
                            log("🖼️ Đang tạo ảnh nền...")
                            sq, src = get_background(base, 1080, 1080, seed=11, variant=0, log=log, theme=th)
                            story_bg, _ = get_background(base + ", vertical composition", 1080, 1920, seed=12,
                                                         variant=1, log=log, allow_free_api=src != "nền gradient", theme=th)
                            log(f"✅ Ảnh nền: {src}")
                            poster_sq = make_poster(sq, brief, brand, 1080, 1080)
                            poster_story = make_poster(story_bg, brief, brand, 1080, 1920)

                            log("🎬 Đang làm video...")
                            video = make_ai_video(brief.get("video_prompt") or base)
                            video_src = "Gemini Omni (video AI)"
                            if not video:
                                video_src = "motion video: chữ động, số liệu chạy, phụ đề, nhạc nền + giọng đọc AI"
                                vid_imgs = []
                                if src != "nền gradient":
                                    vid_imgs.append(story_bg)
                                    ben = next((x for x in brief.get("scenes", []) if x.get("loai") == "benefits"), None)
                                    if ben and ben.get("image_prompt"):
                                        if src.startswith("Pollinations"):
                                            time.sleep(15)                  # giới hạn của gói miễn phí
                                        b, s2 = get_background(ben["image_prompt"], 720, 1280, seed=21, variant=2, log=log)
                                        if s2 != "nền gradient":
                                            vid_imgs.append(b)
                                voice_text = " ".join(clean_text(x.get("voice", "")) for x in brief.get("scenes", []))
                                audio, sec = None, 0.0
                                try:
                                    log("🎙️ Đang thu giọng đọc...")
                                    audio, sec = make_voice(voice_text or clean_text(brief.get("headline", "")))
                                except Exception as e:  # noqa: BLE001
                                    log(f"⚠️ Chưa tạo được giọng đọc ({str(e)[:90]}) → video chỉ có nhạc nền.")
                                log("🎞️ Đang dựng chuyển động (khoảng 20–40 giây)...")
                                video = make_video(vid_imgs, brief, brand, audio, sec)
                            log(f"✅ Video: {video_src}")
                            st.session_state.media = {"brief": brief, "sq": to_png(poster_sq), "story": to_png(poster_story),
                                                      "video": video, "src": src, "video_src": video_src}
                            s_.update(label="Hoàn tất!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Lỗi khi tạo quảng cáo: {e}")

                media = st.session_state.get("media")
                if media:
                    st.caption(f"Ảnh nền: {media['src']} · Video: {media['video_src']}")
                    m1, m2, m3 = st.columns([1.25, 0.85, 0.85], gap="medium")
                    with m1:
                        html('<div class="fs-media-t">📘 Poster Facebook / Instagram (1:1)</div>')
                        st.image(media["sq"], width="stretch")
                        st.download_button("⬇️ Tải poster 1:1", media["sq"], "poster_1x1.png", "image/png")
                    with m2:
                        html('<div class="fs-media-t">📱 Story / TikTok (9:16)</div>')
                        st.image(media["story"], width="stretch")
                        st.download_button("⬇️ Tải poster 9:16", media["story"], "poster_9x16.png", "image/png")
                    with m3:
                        html('<div class="fs-media-t">🎬 Video quảng cáo (9:16)</div>')
                        st.video(media["video"])
                        st.download_button("⬇️ Tải video .mp4", media["video"], "video_quang_cao.mp4", "video/mp4")
                    with st.expander("📋 Brief sáng tạo AI đã viết (tiêu đề, kịch bản, lời đọc)"):
                        st.json(media["brief"])

# ---------------- TAB 2 ----------------
with tab_chat:
    load_chats()
    chats = st.session_state.setdefault("chats", {})
    cur = st.session_state.get("cur_chat")
    if cur not in chats:
        cur = None
    has_data = bool(st.session_state.get("dash")) and has_file

    left, right = st.columns([1, 2.6], gap="medium")

    # ----- cột trái: lịch sử chat -----
    with left.container(border=True):
        html('<div class="fs-media-t">🗂️ Lịch sử chat</div>')
        if st.button("➕ Cuộc trò chuyện mới", width="stretch", type="primary"):
            st.session_state.cur_chat = None
            st.rerun()
        if not chats:
            st.caption("Chưa có cuộc trò chuyện nào. Lịch sử được lưu ngay trên trình duyệt của bạn.")
        for cid in sorted(chats, reverse=True)[:40]:
            c = chats[cid]
            icon = "📎" if c.get("data_name") else "💬"
            if st.button(f"{icon} {c.get('title', 'Cuộc trò chuyện')}", key=f"open_{cid}", width="stretch",
                         type="primary" if cid == cur else "secondary", help=c.get("time", "")):
                st.session_state.cur_chat = cid
                st.rerun()
        with st.expander("💾 Sao lưu / khôi phục"):
            st.download_button("⬇️ Tải toàn bộ lịch sử (.json)", json.dumps(chats, ensure_ascii=False, indent=1),
                               "lich_su_chat.json", "application/json", width="stretch")
            up = st.file_uploader("Khôi phục từ file .json", type=["json"], key="chat_restore")
            if up and st.button("Khôi phục", width="stretch"):
                try:
                    chats.update(json.loads(up.read().decode("utf-8")))
                    st.session_state.need_save = True
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"File không hợp lệ: {e}")

    # ----- cột phải: khung chat -----
    with right.container(border=True):
        chat = chats.get(cur) if cur else None
        data_name = chat.get("data_name") if chat else (st.session_state.data_file.name if has_data else None)
        section("💬", "Hỏi đáp cùng AI",
                f"📎 Đang dùng dữ liệu: {data_name}" if data_name else
                "Hỏi gì cũng được — kiến thức chung, tin tức, marketing, học tập… (không cần tải dữ liệu)")
        if not ready:
            st.warning("AI chưa sẵn sàng – xem thông báo màu vàng ở đầu trang.")

        picked = None
        if not chat:
            suggestions = (["Nhóm hàng nào nên đẩy quảng cáo?", "Kênh nào chuyển đổi tốt nhất?",
                            "Viết caption Facebook cho sản phẩm bán chạy nhất"] if has_data else
                           ["Gợi ý 5 ý tưởng content TikTok cho quán cà phê", "Thương mại điện tử là gì?",
                            "Xu hướng marketing mới nhất năm nay"])
            st.caption("Gợi ý nhanh — bấm để hỏi:")
            for col, q in zip(st.columns(3), suggestions):
                if col.button(q, width="stretch", disabled=not ready, key=f"sg_{q}"):
                    picked = q
        else:
            for msg in chat["messages"]:
                st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖").markdown(msg["text"])

        question = st.chat_input("Hỏi bất cứ điều gì...", disabled=not ready) or picked
        if question:
            if not chat:                                        # tạo cuộc trò chuyện mới + gắn dữ liệu đang có
                cur = str(int(time.time() * 1000))
                chat = {"title": clean_text(question)[:45], "time": time.strftime("%d/%m/%Y %H:%M"),
                        "messages": [], "data_name": None, "data_summary": None}
                chats[cur] = chat
                st.session_state.cur_chat = cur
            if has_data and not chat.get("data_summary"):
                chat["data_name"] = st.session_state.data_file.name
                chat["data_summary"] = st.session_state.dash["summary"]
            chat["messages"].append({"role": "user", "text": question})
            st.chat_message("user", avatar="🧑").markdown(question)

            role = ("Bạn là FoodSave AI – trợ lý AI đa năng. Trả lời MỌI câu hỏi của người dùng ở mọi lĩnh vực "
                    "(kiến thức chung, học tập, công nghệ, đời sống, tin tức...), bằng tiếng Việt, rõ ràng, chính xác; "
                    "đặc biệt giỏi marketing, bán hàng và thương mại điện tử. Cần thông tin mới thì tra cứu Google.\n")
            if chat.get("data_summary"):
                role += (f"Cuộc trò chuyện này gắn với dữ liệu '{chat['data_name']}'. Khi câu hỏi liên quan, "
                         f"hãy dùng số liệu sau:\n{chat['data_summary']}\n")
            history = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
                       for m in chat["messages"][-11:]]
            history[0]["parts"][0]["text"] = role + "\n" + history[0]["parts"][0]["text"]
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    with st.spinner("Đang suy nghĩ..."):
                        answer, _ = ask_gemini(history, search=True)
                except Exception as e:  # noqa: BLE001
                    answer = f"⚠️ {e}"
                st.markdown(answer)
            chat["messages"].append({"role": "assistant", "text": answer})
            st.session_state.need_save = True
            st.rerun()

        if chat:
            b1, b2 = st.columns(2)
            md = "\n\n".join(f"**{'Bạn' if m['role'] == 'user' else 'AI'}:** {m['text']}" for m in chat["messages"])
            b1.download_button("⬇️ Tải cuộc trò chuyện (.md)", f"# {chat['title']}\n\n{md}", "cuoc_tro_chuyen.md",
                               width="stretch")
            if b2.button("🗑️ Xóa cuộc trò chuyện này", width="stretch"):
                chats.pop(cur, None)
                st.session_state.cur_chat = None
                st.session_state.need_save = True
                st.rerun()

    save_chats()

html('<div class="fs-footer">FoodSave · AI Marketing Agent — Môn Thương mại điện tử · Powered by Google Gemini</div>')
