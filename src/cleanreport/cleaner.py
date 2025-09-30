from dataclasses import dataclass
from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np
import yaml
import os, textwrap
from typing import Optional

@dataclass
class StepResult:
    name: str
    changed_rows: int
    changed_cols: List[str]
    notes: Dict[str, Any]

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Cleaner:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.log: List[StepResult] = []
        # 步驟註冊（先做最小集合，之後再擴充）
        self.registry: Dict[str, Callable[[pd.DataFrame], StepResult]] = {
            "drop_empty_like": self.step_drop_empty_like,
            "deduplicate_rows": self.step_deduplicate_rows,
            "type_coercion": self.step_type_coercion,
            "impute_numeric": self.step_impute_numeric,
            "impute_categorical": self.step_impute_categorical,
        }

    # ---------- pipeline entry ----------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.cfg.get("pipeline", []):
            fn = self.registry.get(step)
            if fn is None:
                # 未實作的步驟先跳過，保留擴充彈性
                continue
            res = fn(df)
            self.log.append(res)
        return df

    # ---------- profiling（先做 MVP 指標） ----------
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "missing_rate_by_col": {c: float(df[c].isna().mean()) for c in df.columns},
            "nunique_by_col": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        }

    # ---------- steps (MVP) ----------
    def step_drop_empty_like(self, df: pd.DataFrame) -> StepResult:
        th = self.cfg["rules"]["drop_threshold_missing_col"]
        to_drop = [c for c in df.columns
                   if df[c].isna().mean() >= th or df[c].nunique(dropna=True) <= 1]
        before = df.shape[1]
        df.drop(columns=to_drop, inplace=True, errors="ignore")
        return StepResult("drop_empty_like", 0, to_drop, {"dropped_count": before - df.shape[1]})

    def step_deduplicate_rows(self, df: pd.DataFrame) -> StepResult:
        before = len(df)
        df.drop_duplicates(inplace=True)
        return StepResult("deduplicate_rows", before - len(df), [], {})

    def step_type_coercion(self, df: pd.DataFrame) -> StepResult:
        changed = []
        for c in df.columns:
            if df[c].dtype == "object":
                num_try = pd.to_numeric(df[c], errors="coerce")
            if num_try.notna().mean() >= 0.9:   # 成功率門檻可調
                df[c] = num_try
                changed.append(c)
                continue
            # 再嘗試 datetime（coerce）
            dt_try = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt_try.notna().mean() >= 0.9:
                df[c] = dt_try
                changed.append(c)
        return StepResult("type_coercion", 0, changed, {})
    def step_impute_numeric(self, df: pd.DataFrame) -> StepResult:
        changed = []
        for c in df.select_dtypes(include=[np.number]).columns:
            miss = df[c].isna().mean()
            if miss == 0:
                continue
            if miss <= 0.2:
                df[c].fillna(df[c].median(), inplace=True); changed.append(c)
            elif miss <= 0.6:
                # MVP：先用 median；之後你可以換成 KNN/Iterative
                df[c].fillna(df[c].median(), inplace=True); changed.append(c)
            else:
                # 高缺失先保留；之後加 *_was_na 標記
                pass
        return StepResult("impute_numeric", 0, changed, {})

    def step_impute_categorical(self, df: pd.DataFrame) -> StepResult:
        token = self.cfg.get("categorical", {}).get("impute", "__MISSING__")
        changed = []
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if df[c].isna().any():
                df[c].fillna(token, inplace=True); changed.append(c)
            if self.cfg.get("categorical", {}).get("standardize_case") == "lower":
                df[c] = df[c].astype(str).str.strip().str.lower()
        return StepResult("impute_categorical", 0, changed, {})

# ==== AI suggestions (optional, first-phase report) ====
    

def _summarize_for_llm(before_profile: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    只回傳「必要摘要」給 LLM：欄位名稱、缺失率、唯一值數、型態、(可選)少量 head 取樣。
    預設帶 0 行取樣（完全不送資料內容）；你可改 sample_rows > 0 才會夾帶小樣本。
    """
    # 由 config 控制是否附樣本與樣本行數（預設 0 = 不附樣本）
    ai_cfg = cfg.get("ai", {}) if isinstance(cfg, dict) else {}
    sample_rows = int(ai_cfg.get("sample_head_rows", 0))  # 建議先 0，安全
    include_schema_only = bool(ai_cfg.get("schema_only", True))

    # 推型態資訊
    dtypes = {c: str(dt) for c, dt in df.dtypes.to_dict().items()}
    schema = {
        "columns": list(df.columns),
        "dtypes": dtypes,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
    }

    payload = {
        "schema": schema,
        "missing_rate_by_col": before_profile.get("missing_rate_by_col", {}),
        "nunique_by_col": before_profile.get("nunique_by_col", {}),
        "notes": {
            "drop_threshold_missing_col": (cfg.get("rules", {}) or {}).get("drop_threshold_missing_col"),
        }
    }

    if not include_schema_only and sample_rows > 0 and len(df) > 0:
        payload["sample_head"] = df.head(sample_rows).astype(str).to_dict(orient="records")

    return payload

def make_ai_suggestion(before_profile: Dict[str, Any],
                    df: pd.DataFrame,
                    model: str = "gpt-4o-mini",
                    cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    呼叫 OpenAI 產出「清洗/建模前」的文字建議（Markdown）。
    - 若沒設定 OPENAI_API_KEY，會回傳本地 fallback 建議。
    - 僅用摘要與（可選）極少量樣本，避免發送整份資料。
    """
    cfg = cfg or {}
    summary = _summarize_for_llm(before_profile, df, cfg)

    prompt = f"""
You are a data quality assistant. Based on the provided dataset summary (schema, dtypes, per-column missing rate, nunique),
suggest pragmatic first-pass cleaning and EDA steps **only as text**. 
Do NOT request data. Assume the user will run steps manually.

Return a concise Markdown with sections:
1) Quick Risks (3-6 bullets)
2) Cleaning Priorities (ordered steps)
3) Column-specific Tips (only top 8 risky columns)
4) Modeling Readiness (what to check before training)
5) Nice-to-have (optional ideas)

Dataset summary (JSON):
{summary}
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # fallback：不連網時的本地建議
        bullets = []
        miss_sorted = sorted(summary["missing_rate_by_col"].items(), key=lambda x: x[1], reverse=True)[:5]
        for c, r in miss_sorted:
            bullets.append(f"- `{c}` missing {r:.2%}")
        return textwrap.dedent(f"""\
        # AI Suggestions (offline fallback)
        _No OPENAI_API_KEY detected; returning local heuristic guidance._

        ## Quick Risks
        {os.linesep.join(bullets) if bullets else "- Missing rates look acceptable overall."}

        ## Cleaning Priorities
        1. Drop columns with missing rate ≥ threshold (if business-irrelevant).
        2. Deduplicate rows.
        3. Type coercion for object→numeric/datetime when safe.
        4. Impute numeric (median) and categorical (__MISSING__).
        5. Review high-missing columns; consider flags or removal.

        ## Modeling Readiness
        - Check target leakage, time order, and outliers.
        - Standardize categories; limit high-cardinality.

        ## Nice-to-have
        - Add *_was_na flags; plan outlier handling (IQR/winsorize).
        """)
    # 真正呼叫 OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior data quality engineer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        # 保險處理：避免 None
        return text or "# AI Suggestions\n(No content returned.)"
    except Exception as e:
        return f"# AI Suggestions (error fallback)\nError: {e}\n\nProceed with default cleaning priorities."
