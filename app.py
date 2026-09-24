"""
AI MARKETING AGENT – bản Python (Streamlit + Gemini)
Chuyển từ workflow n8n:  Excel → Dashboard → AI Chiến lược → AI Viết content → Ảnh & Video quảng cáo
Chạy:  streamlit run app.py
"""
import base64
import io
import json
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

st.set_page_config(page_title="AI Marketing Agent", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

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
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("GEMINI_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(api_key: str):
    """Danh sách model key này đang được dùng (Google hay đổi/khai tử model)."""
    return {(m.name or "").replace("models/", "") for m in genai.Client(api_key=api_key).models.list()}


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


def ask_gemini(contents, json_mode=False):
    """Gọi model xịn nhất dùng được. Quá tải → thử lại; không có quyền/hết lượt → tự chuyển model kế tiếp."""
    client = genai.Client(api_key=get_api_key())
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    last_err = None
    for m in text_models()[:5]:
        for attempt in range(2):
            try:
                res = client.models.generate_content(model=m, contents=contents, config=cfg)
                st.session_state.last_model = m
                return res.text or "", m
            except Exception as e:  # noqa: BLE001
                last_err, msg = e, str(e)
                if any(x in msg for x in BUSY):
                    time.sleep(3 * (attempt + 1))
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

PROMPT_BRIEF = """Bạn là Creative Director. Dựa trên chiến lược và content quảng cáo dưới đây:
{strategy}

---
{content}

Hãy tạo BRIEF SÁNG TẠO cho 1 poster quảng cáo và 1 video dọc 9:16 dài khoảng 15–20 giây.
Chỉ trả về JSON đúng cấu trúc sau (tiếng Việt, KHÔNG dùng emoji):
{{
  "thuong_hieu": "tên thương hiệu/nhà bán NẾU dữ liệu có ghi rõ, nếu không có thì để chuỗi rỗng",
  "san_pham": "tên sản phẩm/nhóm hàng được quảng cáo",
  "headline": "tiêu đề chính, tối đa 7 từ, thật cuốn hút",
  "subline": "câu phụ, tối đa 14 từ",
  "cta": "lời kêu gọi, tối đa 4 từ",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "image_prompt": "English prompt: photorealistic commercial product photo for the ad background, studio lighting, no text, no letters, no logo",
  "scenes": [
    {{"text": "chữ trên màn hình, tối đa 7 từ", "image_prompt": "English prompt, no text"}},
    {{"text": "...", "image_prompt": "..."}},
    {{"text": "...", "image_prompt": "..."}}
  ],
  "voiceover": "lời đọc tiếng Việt 35–45 từ, tự nhiên, kết thúc bằng lời kêu gọi",
  "video_prompt": "English prompt for an 8-second vertical 9:16 commercial video of the product, cinematic, no text on screen"
}}"""


def clean_text(s):
    """Bỏ emoji và ký tự lạ mà font không vẽ được."""
    s = re.sub(r"[\U00010000-\U0010FFFF☀-➿️‍]", "", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def font(size, weight="ExtraBold"):
    try:
        return ImageFont.truetype(str(FONT_DIR / f"BeVietnamPro-{weight}.ttf"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default(size)


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
    client = genai.Client(api_key=get_api_key())
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


def get_background(prompt, w, h, seed=7, variant=0, log=None, allow_free_api=True):
    """Ảnh nền tự động: Gemini Nano Banana (nếu key trả phí) → Pollinations (miễn phí) → nền thiết kế sẵn."""
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
            log and log(f"⚠️ Chưa lấy được ảnh AI miễn phí ({str(e)[:80]}) → dùng nền thiết kế.")
    return designed_background(w, h, variant), "nền thiết kế sẵn"


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
        d.rounded_rectangle((pad, y, pad + tw + 36, y + chip_h), radius=40, fill=BRAND_GREEN)
        d.text((pad + 18, y + (chip_h - tf.size) // 2 - 2), tag, font=tf, fill="white")
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
    d.rounded_rectangle((pad, y, pad + cw + 64, y + cta_h), radius=100, fill=BRAND_YELLOW)
    d.text((pad + 32, y + (cta_h - cf.size) // 2 - 3), cta, font=cf, fill=(11, 11, 11))
    if tags:
        d.text((pad, h - pad - tags_h), tags, font=gf, fill=(187, 247, 208))
    return img.convert("RGB")


def to_png(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def make_voice(text, voice=VOICE):
    """Gemini TTS (miễn phí) → trả về (bytes WAV, số giây)."""
    client = genai.Client(api_key=get_api_key())
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


def scene_layer(w, h, text, brand, idx, total, big=False):
    """Lớp chữ trong suốt cho 1 cảnh video."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pad = int(w * 0.08)
    draw_brand(d, pad, int(h * 0.05), brand, int(w * 0.05))
    f, lines = fit_text(d, clean_text(text), w - 2 * pad, 4, int(w * (0.12 if big else 0.10)))
    lh = int(f.size * 1.15)
    y = int(h * 0.62) - (len(lines) * lh) // 2
    for ln in lines:
        d.text((pad, y), ln, font=f, fill="white", stroke_width=2, stroke_fill=(0, 0, 0, 90))
        y += lh
    d.rounded_rectangle((pad, y + 16, pad + int(w * 0.22), y + 26), radius=6, fill=BRAND_YELLOW)
    return layer


def cta_layer(w, h, brief, brand):
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pad = int(w * 0.08)
    label = clean_text(brand or brief.get("san_pham", "")).upper()
    bf, bl = fit_text(d, label, w - 2 * pad, 2, int(w * 0.09))
    by = h * 0.26
    for ln in bl:
        bw = d.textlength(ln, font=bf)
        d.text(((w - bw) / 2, by), ln, font=bf, fill="white")
        by += bf.size * 1.15
    d.rounded_rectangle((w * 0.38, by + 14, w * 0.62, by + 24), radius=5, fill=BRAND_YELLOW)
    hf, lines = fit_text(d, clean_text(brief.get("headline", "")), w - 2 * pad, 3, int(w * 0.075))
    y = h * 0.45
    for ln in lines:
        lw = d.textlength(ln, font=hf)
        d.text(((w - lw) / 2, y), ln, font=hf, fill="white")
        y += hf.size * 1.15
    cta = clean_text(brief.get("cta", "Mua ngay")).upper()
    cf = font(int(w * 0.055), "ExtraBold")
    cw = d.textlength(cta, font=cf)
    y += 40
    d.rounded_rectangle(((w - cw) / 2 - 40, y, (w + cw) / 2 + 40, y + cf.size * 2.1), radius=100, fill=BRAND_YELLOW)
    d.text(((w - cw) / 2, y + cf.size * 0.5), cta, font=cf, fill=(11, 11, 11))
    tags = " ".join(clean_text(t) for t in brief.get("hashtags", [])[:3])
    tf = font(int(w * 0.035), "Medium")
    tw = d.textlength(tags, font=tf)
    d.text(((w - tw) / 2, h * 0.86), tags, font=tf, fill=(187, 247, 208))
    return layer


def make_video(backgrounds, brief, brand, audio=None, audio_sec=0.0, w=720, h=1280, fps=20):
    """Ghép các cảnh thành MP4: zoom chậm (Ken Burns) + chữ hiện dần + cảnh kết CTA + giọng đọc."""
    import imageio_ffmpeg

    scenes = [s for s in brief.get("scenes", []) if s.get("text")][:4] or [{"text": brief.get("headline", "")}]
    n = len(scenes) + 1                                         # + cảnh kết
    total = max(12.0, audio_sec + 0.8) if audio else 3.2 * n
    dur = total / n
    fade_bg = dark_fade(w, h, 0.25, 215)
    layers = [scene_layer(w, h, s["text"], brand, i, n) for i, s in enumerate(scenes)] + [cta_layer(w, h, brief, brand)]
    bgs = [b.resize((int(w * 1.18), int(h * 1.18)), Image.LANCZOS) for b in backgrounds]
    end_bg = designed_background(int(w * 1.18), int(h * 1.18), 0)

    tmp = Path(tempfile.mkdtemp())
    silent, final = tmp / "silent.mp4", tmp / "final.mp4"
    writer = imageio_ffmpeg.write_frames(str(silent), (w, h), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=1)
    writer.send(None)
    frames_per = int(dur * fps)
    for si in range(n):
        bg = end_bg if si == n - 1 else bgs[si % len(bgs)]
        layer = layers[si]
        la = np.array(layer, dtype=np.float32)
        for k in range(frames_per):
            p = k / max(1, frames_per - 1)
            z = 1.0 + 0.15 * p                                  # zoom chậm
            cw_, ch_ = bg.width / z, bg.height / z
            left = (bg.width - cw_) / 2 + (18 * p if si % 2 else -18 * p)
            top = (bg.height - ch_) / 2
            frame = bg.crop((int(left), int(top), int(left + cw_), int(top + ch_))).resize((w, h), Image.BILINEAR)
            frame = frame.convert("RGBA")
            frame.alpha_composite(fade_bg)
            alpha = min(1.0, k / (0.35 * fps))                  # chữ hiện dần
            fl = la.copy()
            fl[..., 3] *= alpha
            rise = int((1 - alpha) * 30)
            txt = Image.fromarray(fl.astype(np.uint8), "RGBA")
            frame.alpha_composite(txt, (0, rise))
            fd = ImageDraw.Draw(frame)                          # thanh tiến trình
            prog = (si + p) / n
            fd.rounded_rectangle((40, 24, 40 + (w - 80) * prog, 30), radius=3, fill=BRAND_YELLOW)
            writer.send(np.asarray(frame.convert("RGB")).tobytes())
    writer.close()

    if audio:
        wav = tmp / "voice.wav"
        wav.write_bytes(audio)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-af", "adelay=300|300", "-shortest", str(final)],
                       check=True)
        return final.read_bytes()
    return silent.read_bytes()


def make_brief(strategy, content):
    prompt = PROMPT_BRIEF.format(strategy=strategy[:6000], content=content[:6000])
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
        client = genai.Client(api_key=get_api_key())
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


# ---------------- khung trang ----------------
html(CSS)
api_key = get_api_key()
ready = False
if api_key:
    try:
        ready = bool(text_models())
    except Exception:  # noqa: BLE001
        ready = False

html('<div class="fs-bar"><span>Excel → Dashboard → Chiến lược → Content → Ảnh & Video · tự động 100%</span>'
     '<span class="live">LIVE</span></div>')
html('<div style="margin-top:16px"><div class="fs-logo">AI <span>Marketing</span> Agent<i></i></div>'
     '<div class="fs-logo-sub">TỰ ĐỘNG HÓA MARKETING TỪ DỮ LIỆU BÁN HÀNG</div></div>')

model_now = st.session_state.get("last_model") or (text_models()[0] if ready else "")
if ready:
    status = f'<span class="fs-status on">● AI sẵn sàng · {model_now}</span>'
elif api_key:
    status = '<span class="fs-status off">● Không kết nối được Gemini – kiểm tra lại API key</span>'
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
                            brief = make_brief(st.session_state.strategy, st.session_state.content)
                            brand = clean_text(brief.get("thuong_hieu", ""))
                            base = brief.get("image_prompt") or "premium product advertising photo, studio lighting"
                            log("🖼️ Đang tạo ảnh nền...")
                            sq, src = get_background(base, 1080, 1080, seed=11, variant=0, log=log)
                            story_bg, _ = get_background(base + ", vertical composition", 1080, 1920,
                                                         seed=12, variant=1, log=log)
                            log(f"✅ Ảnh nền: {src}")
                            poster_sq = make_poster(sq, brief, brand, 1080, 1080)
                            poster_story = make_poster(story_bg, brief, brand, 1080, 1920)

                            log("🎬 Đang làm video...")
                            video = make_ai_video(brief.get("video_prompt") or base)
                            video_src = "Gemini Omni (video AI)"
                            if not video:
                                video_src = "dựng tự động từ ảnh + giọng đọc AI"
                                vid_bgs = [story_bg.resize((720, 1280))]
                                ai_bg = not src.startswith("nền")
                                for i, sc in enumerate(brief.get("scenes", [])[1:3]):
                                    if ai_bg and src.startswith("Pollinations"):
                                        time.sleep(15)              # giới hạn của gói miễn phí
                                    b, _ = get_background(sc.get("image_prompt") or base, 720, 1280, seed=20 + i,
                                                          variant=i + 1, log=log, allow_free_api=ai_bg)
                                    vid_bgs.append(b)
                                audio, sec = None, 0.0
                                try:
                                    log("🎙️ Đang thu giọng đọc...")
                                    audio, sec = make_voice(clean_text(brief.get("voiceover", "")))
                                except Exception as e:  # noqa: BLE001
                                    log(f"⚠️ Chưa tạo được giọng đọc ({str(e)[:90]}) → video không tiếng.")
                                video = make_video(vid_bgs, brief, brand, audio, sec)
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
    with st.container(border=True):
        section("💬", "Hỏi đáp cùng AI", "Đã tải dữ liệu ở tab bên cạnh thì AI trả lời dựa trên dashboard đó")
        st.session_state.setdefault("messages", [])
        if not st.session_state.messages:
            html('<span class="fs-hint">Nhóm hàng nào nên đẩy quảng cáo?</span>'
                 '<span class="fs-hint">Kênh nào chuyển đổi tốt nhất?</span>'
                 '<span class="fs-hint">Viết 1 caption Facebook cho sản phẩm bán chạy</span>')
        for msg in st.session_state.messages:
            st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖").markdown(msg["text"])
        question = st.chat_input("Nhập câu hỏi...", disabled=not ready)
        if question:
            st.session_state.messages.append({"role": "user", "text": question})
            context = ""
            if st.session_state.get("dash"):
                context = "Bạn là trợ lý marketing. Dữ liệu dashboard hiện tại:\n" + st.session_state.dash["summary"] + "\n\n"
            history = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
                       for m in st.session_state.messages]
            history[0]["parts"][0]["text"] = context + history[0]["parts"][0]["text"]
            try:
                with st.spinner("Đang suy nghĩ..."):
                    answer, _ = ask_gemini(history)
            except Exception as e:  # noqa: BLE001
                answer = f"⚠️ {e}"
            st.session_state.messages.append({"role": "assistant", "text": answer})
            st.rerun()
        if st.session_state.messages and st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()

html('<div class="fs-footer">AI Marketing Agent — Môn Thương mại điện tử · Powered by Google Gemini</div>')
