import os
import re
import json
import time
import requests
import urllib3
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
from google import genai

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
KAKAO_REST_API_KEY = st.secrets["KAKAO_REST_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# 기본 함수
# =========================

def get_daangn_data(keyword, region_slug):
    url = "https://www.daangn.com/kr/buy-sell/"

    params = {
        "search": keyword,
        "in": str(region_slug)
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.daangn.com/",
    }

    res = requests.get(url, params=params, headers=headers, verify=False)
    res.encoding = "utf-8"

    if res.status_code != 200:
        raise Exception(f"요청 실패: {res.status_code}")

    return res.text


def extract_remix_context(html):
    soup = BeautifulSoup(html, "html.parser")

    target_script = None
    for script in soup.find_all("script"):
        txt = script.get_text()
        if "window.__remixContext" in txt:
            target_script = txt
            break

    if target_script is None:
        raise Exception("window.__remixContext 못 찾음")

    match = re.search(
        r"window\.__remixContext\s*=\s*(\{.*?\});",
        target_script,
        re.S
    )

    if not match:
        raise Exception("remixContext JSON 추출 실패")

    return json.loads(match.group(1))


def collect_product_like_items(obj):
    items = []

    product_keys = {
        "title", "price", "priceText", "region",
        "regionName", "imageUrl", "image_url",
        "href", "url", "id"
    }

    if isinstance(obj, dict):
        keys = set(obj.keys())

        if len(keys & product_keys) >= 2:
            items.append(obj)

        for v in obj.values():
            items.extend(collect_product_like_items(v))

    elif isinstance(obj, list):
        for x in obj:
            items.extend(collect_product_like_items(x))

    return items


def get_json_value(x, key):
    if isinstance(x, dict):
        return x.get(key)

    if isinstance(x, str):
        try:
            return json.loads(x).get(key)
        except:
            return None

    return None


def geocode_address(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {"query": address}

    res = requests.get(url, headers=headers, params=params, verify=False)

    if res.status_code != 200:
        return None, None

    data = res.json()

    if not data.get("documents"):
        return None, None

    first = data["documents"][0]

    lat = float(first["y"])
    lon = float(first["x"])

    return lat, lon


def calc_distance_km(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def split_region_name(name):
    m = re.match(r"^([가-힣]+)(\d+(?:[,.]\d+)+)([가-힣]+)$", str(name))

    if not m:
        return [name]

    prefix = m.group(1)
    nums = re.split(r"[,.]", m.group(2))
    suffix = m.group(3)

    return [f"{prefix}{n}{suffix}" for n in nums]


@st.cache_data
def load_region_data():
    dong_geo_df = pd.read_csv("region_with_coords.csv", encoding="utf-8-sig")
    daangn_region_df = pd.read_csv("daangn_region_map.csv", encoding="utf-8-sig")

    daangn_region_df = daangn_region_df[[
        "daangn_region_id", "region_name"
    ]].rename(columns={
        "daangn_region_id": "region_id"
    }).drop_duplicates()

    daangn_region_df["region_split"] = daangn_region_df["region_name"].apply(split_region_name)
    daangn_region_df = daangn_region_df.explode("region_split").copy()
    daangn_region_df["region_name"] = daangn_region_df["region_split"]
    daangn_region_df = daangn_region_df.drop(columns=["region_split"])

    daangn_region_df["region_id"] = daangn_region_df["region_id"].astype(str)

    return dong_geo_df, daangn_region_df


def find_nearest_regions_by_input(dong_geo_df, input_text, top_n=10):
    base_lat, base_lon = geocode_address(input_text)

    if base_lat is None:
        raise Exception(f"좌표 변환 실패: {input_text}")

    df = dong_geo_df.dropna(subset=["lat", "lon"]).copy()

    df["distance_km"] = df.apply(
        lambda row: calc_distance_km(base_lat, base_lon, row["lat"], row["lon"]),
        axis=1
    )

    return df.sort_values("distance_km").head(top_n).reset_index(drop=True)


def crawl_by_region_and_keyword(keyword, region_slug):
    html = get_daangn_data(keyword, region_slug)
    data = extract_remix_context(html)

    loader_data = data["state"]["loaderData"]
    page = loader_data["routes/kr.buy-sell._index"]

    items = collect_product_like_items(page)
    df = pd.DataFrame(items)

    if df.empty or len(df) <= 1:
        return pd.DataFrame()

    raw_df = df.iloc[1:].reset_index(drop=True)

    required_cols = ["price", "title", "content", "status", "createdAt", "user", "region", "href", "thumbnail"]
    for col in required_cols:
        if col not in raw_df.columns:
            raw_df[col] = None

    result_df = pd.DataFrame({
        "price": raw_df["price"],
        "title": raw_df["title"],
        "content": raw_df["content"],
        "status": raw_df["status"],
        "createdAt": raw_df["createdAt"],

        "user_dbid": raw_df["user"].apply(lambda x: get_json_value(x, "dbId")),
        "user_nickname": raw_df["user"].apply(lambda x: get_json_value(x, "nickname")),

        "region_dbid": raw_df["region"].apply(lambda x: get_json_value(x, "dbId")),
        "region_name": raw_df["region"].apply(lambda x: get_json_value(x, "name")),

        "href": raw_df["href"],
        "thumbnail": raw_df["thumbnail"],
    })

    return result_df


def build_result_df(keyword, input_region, top_n=10):
    dong_geo_df, daangn_region_df = load_region_data()

    nearest_n = find_nearest_regions_by_input(
        dong_geo_df,
        input_text=input_region,
        top_n=top_n
    )

    region_id_list = (
        daangn_region_df[
            daangn_region_df["region_name"].isin(nearest_n["dong"])
        ]["region_id"]
        .drop_duplicates()
        .tolist()
    )

    result_df = pd.DataFrame()

    for region_id in region_id_list:
        temp_df = crawl_by_region_and_keyword(keyword, region_id)
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        time.sleep(0.2)

    if result_df.empty:
        return result_df, nearest_n

    result_df["searched_region"] = input_region

    result_df["distance_km"] = result_df["region_name"].apply(
        lambda x: nearest_n[nearest_n["dong"] == x]["distance_km"].values[0]
        if not nearest_n[nearest_n["dong"] == x].empty
        else None
    )

    result_df = result_df[result_df["status"] == "Ongoing"].copy()
    result_df = result_df.drop_duplicates(subset=["href"]).reset_index(drop=True)

    return result_df, nearest_n


# =========================
# Gemini 필터
# =========================

def safe_json_loads(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


def filter_df_with_gemini_light(
    result_df,
    condition,
    batch_size=30,
    model="gemini-2.5-flash-lite",
    sleep_sec=8,
    content_len=50,
):
    all_judges = []

    usage_summary = {
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "batch_size": batch_size,
        "model": model,
        "success_batches": 0,
        "failed_batches": 0,
    }

    progress = st.progress(0)
    status_box = st.empty()

    for start in range(0, len(result_df), batch_size):
        batch = result_df.iloc[start:start + batch_size]

        items = []
        for idx, row in batch.iterrows():
            items.append({
                "idx": int(idx),
                "title": str(row.get("title", ""))[:80],
                "content": str(row.get("content", ""))[:content_len],
                "status": str(row.get("status", "")),
                "price": row.get("price", None),
            })

        prompt = f"""
조건:{condition}

출력은 JSON 배열만.
형식:[{{"idx":0,"keep":true}}]

items:
{json.dumps(items, ensure_ascii=False)}
"""

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            judges = safe_json_loads(response.text)

            judges = [
                {
                    "idx": int(x.get("idx")),
                    "keep": bool(x.get("keep", False))
                }
                for x in judges
                if "idx" in x
            ]

            all_judges.extend(judges)

            usage = response.usage_metadata

            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            used_tokens = getattr(usage, "total_token_count", 0) or 0

            usage_summary["total_prompt_tokens"] += prompt_tokens
            usage_summary["total_output_tokens"] += output_tokens
            usage_summary["total_tokens"] += used_tokens
            usage_summary["success_batches"] += 1

            status_box.info(
                f"{start}~{start + len(batch) - 1} 처리 완료 / "
                f"이번 토큰 {used_tokens} / 누적 {usage_summary['total_tokens']}"
            )

            time.sleep(sleep_sec)

        except Exception as e:
            usage_summary["failed_batches"] += 1
            status_box.warning(f"{start}~{start + len(batch) - 1} 실패: {e}")

            all_judges.extend([
                {"idx": int(idx), "keep": False}
                for idx in batch.index
            ])

            time.sleep(sleep_sec * 3)

        progress.progress(min((start + batch_size) / len(result_df), 1.0))

    judge_df = pd.DataFrame(all_judges)

    checked_df = result_df.copy()

    if not judge_df.empty:
        judge_df = judge_df.drop_duplicates(subset=["idx"], keep="last")
        judge_df = judge_df.set_index("idx")
        checked_df = checked_df.join(judge_df)

    filtered_df = checked_df[checked_df["keep"] == True].copy()

    return filtered_df, checked_df, usage_summary


def to_numeric_price(x):
    try:
        if pd.isna(x):
            return None

        x = str(x)
        x = x.replace(",", "").replace("원", "").strip()

        return float(x)
    except:
        return None


def sort_df(df, sort_option):
    if df.empty:
        return df

    view_df = df.copy()

    if "price" in view_df.columns:
        view_df["_price_num"] = view_df["price"].apply(to_numeric_price)

    if "distance_km" in view_df.columns:
        view_df["_distance_num"] = pd.to_numeric(
            view_df["distance_km"],
            errors="coerce"
        )

    if sort_option == "거리순":
        view_df = view_df.sort_values("_distance_num", ascending=True, na_position="last")

    elif sort_option == "가격 낮은순":
        view_df = view_df.sort_values("_price_num", ascending=True, na_position="last")

    elif sort_option == "가격 높은순":
        view_df = view_df.sort_values("_price_num", ascending=False, na_position="last")

    elif sort_option == "최신순":
        view_df["_created_dt"] = pd.to_datetime(view_df["createdAt"], errors="coerce")
        view_df = view_df.sort_values("_created_dt", ascending=False, na_position="last")

    drop_cols = [c for c in ["_price_num", "_distance_num", "_created_dt"] if c in view_df.columns]
    view_df = view_df.drop(columns=drop_cols)

    return view_df


def format_price(x):
    try:
        if pd.isna(x):
            return ""
        return f"{int(float(x)):,}원"
    except:
        return str(x)


def display_product_table(df):
    show_cols = [
        "title", "price", "region_name",
        "distance_km", "content", "href"
    ]

    view_df = df.copy()

    if "price" in view_df.columns:
        view_df["price"] = view_df["price"].apply(format_price)

    if "distance_km" in view_df.columns:
        view_df["distance_km"] = view_df["distance_km"].apply(
            lambda x: round(x, 2) if pd.notna(x) else ""
        )

    exist_cols = [c for c in show_cols if c in view_df.columns]

    st.dataframe(
        view_df[exist_cols],
        use_container_width=True,
        hide_index=True
    )

def display_product_table_with_image(df):
    show_cols = [
        "thumbnail", "title", "price", "region_name",
        "distance_km", "content", "href"
    ]

    view_df = df.copy()

    if "price" in view_df.columns:
        view_df["price"] = view_df["price"].apply(format_price)

    if "distance_km" in view_df.columns:
        view_df["distance_km"] = view_df["distance_km"].apply(
            lambda x: round(x, 2) if pd.notna(x) else ""
        )

    exist_cols = [c for c in show_cols if c in view_df.columns]

    st.dataframe(
        view_df[exist_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "thumbnail": st.column_config.ImageColumn(
                "이미지",
                width="small"
            ),
            "href": st.column_config.LinkColumn(
                "링크"
            ),
        }
    )

# =========================
# Streamlit 화면
# =========================

st.set_page_config(
    page_title="당근 매물 LLM 필터",
    page_icon="🥕",
    layout="wide"
)

st.title("🥕 당근 매물 검색 + LLM 필터")

if "result_df" not in st.session_state:
    st.session_state.result_df = pd.DataFrame()

if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()

with st.chat_message("assistant"):
    st.write("어떤 항목을 어느 지역에서 찾을까요? 예: `아이폰`, `이태원동`")

with st.form("search_form"):
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        keyword = st.text_input("찾을 항목", placeholder="예: 아이폰, 갤럭시, 에어팟")

    with col2:
        input_region = st.text_input("기준 지역", placeholder="예: 이태원동, 한남동, 성수동")

    with col3:
        top_n = st.number_input("근처 동 개수", min_value=1, max_value=30, value=10)

    search_submitted = st.form_submit_button("1차 검색")

if search_submitted:
    if not keyword or not input_region:
        st.error("항목과 지역을 둘 다 입력해주세요.")
    else:
        with st.spinner("당근 매물 수집 중..."):
            try:
                result_df, nearest_n = build_result_df(keyword, input_region, top_n=int(top_n))
                st.session_state.result_df = result_df
                st.session_state.filtered_df = pd.DataFrame()

                st.success(f"1차 수집 완료: {len(result_df)}개")
                st.write("검색 대상 근처 지역")
                st.dataframe(nearest_n[["full_name", "dong", "distance_km"]], hide_index=True)

            except Exception as e:
                st.error(f"검색 실패: {e}")

result_df = st.session_state.result_df

if not result_df.empty:
    st.subheader("1차 검색 결과")

    sort_option_1 = st.selectbox(
        "1차 결과 정렬",
        ["기본", "거리순", "가격 낮은순", "가격 높은순", "최신순"],
        key="sort_1"
    )

    sorted_result_df = sort_df(result_df, sort_option_1)
    display_product_table_with_image(sorted_result_df)

    st.divider()

    st.subheader("LLM 조건 필터링")

    default_condition = f"""
{keyword} 관련 매물만 true.
고장, 부품용, 파손, 침수, 계정잠김, 사기 의심은 false.
본문에 직접 근거 없으면 false.
비슷한 상품 포함 금지.
애매하면 false.
"""

    condition = st.text_area(
        "필터 조건",
        value=default_condition,
        height=160
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        batch_size = st.number_input("batch_size", min_value=5, max_value=100, value=30)

    with col2:
        content_len = st.number_input("본문 글자 수 제한", min_value=10, max_value=200, value=50)

    with col3:
        sleep_sec = st.number_input("호출 간격 초", min_value=1, max_value=60, value=8)

    if st.button("LLM으로 필터링"):
        with st.spinner("Gemini 필터링 중..."):
            filtered_df, checked_df, usage_summary = filter_df_with_gemini_light(
                result_df=result_df,
                condition=condition,
                batch_size=int(batch_size),
                sleep_sec=int(sleep_sec),
                content_len=int(content_len),
            )

            st.session_state.filtered_df = filtered_df
            st.session_state.checked_df = checked_df
            st.session_state.usage_summary = usage_summary

            st.success(f"LLM 필터링 완료: {len(filtered_df)}개 남음")
            st.json(usage_summary)

filtered_df = st.session_state.filtered_df

if not filtered_df.empty:
    st.divider()
    st.subheader("최종 필터링 결과")

    sort_option_2 = st.selectbox(
        "최종 결과 정렬",
        ["기본", "거리순", "가격 낮은순", "가격 높은순", "최신순"],
        key="sort_2"
    )

    sorted_filtered_df = sort_df(filtered_df, sort_option_2)
    display_product_table(sorted_filtered_df)

    csv = sorted_filtered_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSV 다운로드",
        csv,
        file_name="filtered_daangn_result.csv",
        mime="text/csv"
    )
