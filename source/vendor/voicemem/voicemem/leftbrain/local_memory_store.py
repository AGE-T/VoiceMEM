"""左脑向量存储的共享工具：embedder 接口/实现、检索命中类型、记忆根目录解析、
时间类问句的词面/时间加分辅助函数。实际的存储与检索现在由
``mem0_backend_store.py``（真实 mem0/Qdrant 后端）实现；本文件不再包含存储层
本身，只保留仍被该后端和上层复用的这些工具。

默认库根目录：仓库根下 ``memory/leftbrain/``。可用环境变量
``VOICEMEM_MEMORY_ROOT`` 覆盖。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date as _date_cls, datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

# 本文件位于 voicemem/leftbrain/，parents[2] == 本仓库根目录
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MEMORY_SQLITE = "voicemem_leftbrain.sqlite"
_DEFAULT_LEFTBRAIN_MEMORY_ROOT = _REPO_ROOT / "memory" / "leftbrain"


def default_memory_root() -> Path:
    """本地记忆文件根目录。

    优先级：``VOICEMEM_MEMORY_ROOT`` → 仓库 ``memory/leftbrain/``。
    """
    env = (os.environ.get("VOICEMEM_MEMORY_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    _DEFAULT_LEFTBRAIN_MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_LEFTBRAIN_MEMORY_ROOT.resolve()


def default_local_memory_db_path() -> Path:
    """默认 SQLite 路径：``default_memory_root() / voicemem_leftbrain.sqlite``。"""
    return default_memory_root() / _DEFAULT_MEMORY_SQLITE


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 词面 + 问题类型的检索补强 ────────────────────────────────────────────────
# 纯余弦对「多久 / 什么时候」这类问题不友好：问句里根本没有 years/月 这种词，
# 而答案恰恰落在带时长或日期的那几条记忆上，向量距离又不近，于是被埋在几十名开外。
# 这里在余弦之上叠两个近乎零成本的信号（纯正则，不额外调 LLM、不重新入库）：
#   1) 词面重合——问句实词（art / hike / friends）在记忆原文里出现了多少；
#   2) 问题类型——问「多久」就给含时长表达的记忆加分，问「何时」就给含日期的加分。

_DURATION_RE = re.compile(
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)[\s-]*"
    r"(?:year|month|week|day|hour|minute|decade)s?\b"
    r"|\bsince\s+(?:19|20)\d{2}\b"
    r"|\bhalf\s+a\s+(?:year|month)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}-\d{2}-\d{2}\b"
    r"|\b(?:last|next)\s+(?:year|month|week)\b"
    # 中文日期。抽取归一后的事实全是"2026年8月29日（周六）"这种写法，只认英文的话
    # memory_ids_with_time_expr() 在中文库上永远返回空集，时间扩候选等于没开。
    r"|(?:(?:19|20)\d{2}年)?\d{1,2}月\d{1,2}[日号]",
    re.I,
)
_DURATION_Q_RE = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:year|month|week|day|hour)s?\b|多久|多长时间", re.I
)
_DATE_Q_RE = re.compile(
    r"\bwhen\b|\bwhat\s+(?:date|day|time)\b|哪天|什么时候"
    # 中文相对时间词。"我下周有什么安排"以前判不出是时间问题，于是
    # _widen_for_time_question 不触发——日程一旦被槽位过滤掉（约饭归 relationships，
    # 问的却是 schedule）就再也捞不回来，实测三条下周日程只剩两条。
    r"|今天|明天|后天|昨天|前天|这周|本周|下周|下星期|下个星期|上周|上个星期"
    r"|接下来|这几天|安排|日程|行程"
    # [CONTROLLED FORK - patch VM-LOCAL-010] 匈牙利语时间问句线索（外部审计
    # v0.6.3 CD-3：Parakeet 正确转写 HU，但检索层对 HU 时间问题全盲）。
    # "mi lesz jövő héten" / "mit mondtam tegnap" 以前都不是 date-问题，
    # 时间加分与时间扩候选从不触发。匈牙利语带变音符号——\b 在 Python 3
    # unicode 模式下把 ő/ű/é 当词字符，边界计算正确。
    r"|\bmikor\b|\bmelyik\s+nap\b|\bmúlt\s+hét\w*\b|\bmúlthéten\b"
    r"|\bjövő\s+hét\w*\b|\bjövőhéten\b|\bezen\s+a\s+héten\b|\bez\s+a\s+hét\b"
    r"|\bezhéten\b|\btegnap\b|\btegnapelőtt\b|\bholnap\b|\bholnapután\b"
    r"|\bhétvég\w*\b|\bnapirend\b|\bbeosztás\b|\bmenetrend\b|\bprogramom\b"
    r"|\bmúlt\s+hónap\w*\b|\bjövő\s+hónap\w*\b",
    re.I,
)

# 问句里没有区分度的高频词，参与词面匹配只会制造噪声
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at for from by with about as into over
    is are was were be been being do does did have has had can could will would
    shall should may might must what when where who whom which why how many much
    long time you your their his her its it they them he she we us our i me my
    that this these those there here not no yes so than then get got make made
    """.split()
)

_LEX_WEIGHT = 0.15   # 词面全中时加的分（余弦本身量级约 0.2~0.6）
_TIME_WEIGHT = 0.10  # 问题类型命中时额外加的分
_DATE_MATCH_WEIGHT = 0.12  # 问句展开出来的日期，跟记忆正文里的日期对上时加的分

#: 中文日期字面，如 "2026年8月26日" / "8月26日"。抽取归一后的事实就是这个写法。
_CJK_DATE_RE = re.compile(r"(?:(?:19|20)\d{2}年)?\d{1,2}月\d{1,2}日")


def query_dates(query: str) -> frozenset[str]:
    """问句里出现的中文日期字面。

    多数问句本身没有日期——是 ``time_expand.expand_relative_dates()`` 把"下周"
    展开成那七天拼上去的。取出来给 ``date_overlap_bonus`` 做精确比对。
    """
    return frozenset(_CJK_DATE_RE.findall(query or ""))


def date_overlap_bonus(q_dates: frozenset[str], mem_text: str) -> float:
    """记忆正文里的日期，在不在问句涉及的那几天里。

    展开出来的日期只进了向量，而余弦对"这条日期在不在范围内"分辨力很弱：实测
    "我下周有什么安排"三条下周日程只排上来两条，被日期不在下周、语义却更近的
    条目（面试 8/22、咖啡厅 8/20）挤掉了。这里补一次精确字面比对——加的是
    "在范围内"这个硬事实，不是相似度。
    """
    if not q_dates:
        return 0.0
    return _DATE_MATCH_WEIGHT if any(d in mem_text for d in q_dates) else 0.0


# ── [CONTROLLED FORK - patch VM-LOCAL-010] 多语种日期字面 → date 值 ──────────
# 外部审计 v0.6.3 CD-3 的另一半：date_overlap_bonus 的字面比对只在问句与
# 正文**同格式**时才命中（"2026-09-14" 撞不上 "2026. szeptember 14."）。
# 这里把四种日期家族（ISO / 中文 / 匈牙利语 / 英语）都解析成 date 值，
# 值级比对与书写格式无关。time_expand 的 ISO 扩展戳与 LLM 归一化产物
# 在这里都能被解析。
_MONTHS_HU = {
    "január": 1, "február": 2, "március": 3, "április": 4, "május": 5,
    "június": 6, "július": 7, "augusztus": 8, "szeptember": 9,
    "október": 10, "november": 11, "december": 12,
}
_MONTHS_EN = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_HU_MONTH_ALT = "|".join(_MONTHS_HU)
_ISO_DATE_RE2 = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")
_CJK_YMD_RE = re.compile(r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日")
_HU_YMD_RE = re.compile(
    rf"\b((?:19|20)\d{{2}})\.?\s?({_HU_MONTH_ALT})\s?(\d{{1,2}})\.?\b", re.I)
_HU_MD_RE = re.compile(
    rf"\b({_HU_MONTH_ALT})\s?(\d{{1,2}})(?:-án|-én|-ára|-áig|-tól|-ig)?\b", re.I)
_EN_MDY_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})"
    r"(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b", re.I)


def parse_date_values(text: str, today: _date_cls | None = None) -> frozenset:
    """文本里出现的**日期值**（四种格式家族都解析）。无年份的月日按当年补全。

    返回 ``frozenset[date]``。解析失败的片段静默跳过（日期正则宽松，
    "2026-13-45" 这类不是日期）。"""
    if not text:
        return frozenset()
    today = today or _date_cls.today()
    out: set = set()

    def _add(y: int, m: int, d: int) -> None:
        try:
            out.add(_date_cls(y, m, d))
        except ValueError:
            pass

    for y, m, d in _ISO_DATE_RE2.findall(text):
        _add(int(y), int(m), int(d))
    for y, m, d in _CJK_YMD_RE.findall(text):
        _add(int(y), int(m), int(d))
    for y, mon, d in _HU_YMD_RE.findall(text):
        _add(int(y), _MONTHS_HU[mon.lower()], int(d))
    for mon, d in _HU_MD_RE.findall(text):
        _add(today.year, _MONTHS_HU[mon.lower()], int(d))
    for mon, d, y in _EN_MDY_RE.findall(text):
        _add(int(y), _MONTHS_EN[mon.lower()], int(d))
    return frozenset(out)


def query_date_values(query: str, today: _date_cls | None = None) -> frozenset:
    """问句里出现的日期值（含 time_expand 拼上去的 ISO 扩展戳）。"""
    return parse_date_values(query, today)


def date_overlap_bonus_values(q_values: frozenset, mem_text: str,
                              today: _date_cls | None = None) -> float:
    """[VM-LOCAL-010] 值级日期重叠加分：与书写格式无关。

    问句展开出的 date 值（任何格式来源）与记忆正文里解析出的 date 值
    有交集 → 同一个 _DATE_MATCH_WEIGHT 硬事实加分。旧的字面版
    ``date_overlap_bonus`` 保留（CJK 字面路径不变，向后兼容）。"""
    if not q_values:
        return 0.0
    mem_vals = parse_date_values(mem_text, today)
    return _DATE_MATCH_WEIGHT if (q_values & mem_vals) else 0.0


# ── [CONTROLLED FORK - patch VM-LOCAL-011] 时效性（recency）加分 ──────────────
# 外部审计 v0.6.3 CD-4（P1）：observed_at 一路存到库里、进了命中对象，
# 却从不参与排序——"我最近说过什么" 不偏向新记忆。这里加一个指数衰减的
# 时效加分（量级与 _TIME_WEIGHT 同级，语义相关的老记忆不会被无脑压死；
# 30 天半衰期：一周内的观察 ~0.085，一个月 ~0.05，一年 ~0.0）。
# 未来日期（日程类"下周三体检"）按今天算——它们是最"新鲜"的。
_RECENCY_WEIGHT = 0.10
_RECENCY_HALF_LIFE_DAYS = 30.0


def recency_bonus(observed_at: str, today: _date_cls | None = None) -> float:
    """[VM-LOCAL-011] 一条命中的时效加分（事件时间越近越高，无日期 = 0）。

    ``observed_at`` 是 hit 上已归一的 ``YYYY-MM-DD``（事件时间，非写入时间）；
    旧数据无该字段 → 0.0，排序行为不变。"""
    s = str(observed_at or "")[:10]
    if not (len(s) == 10 and s[:4].isdigit() and s[4] == "-"):
        return 0.0
    today = today or _date_cls.today()
    try:
        d = _date_cls(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:
        return 0.0
    days = (today - d).days
    if days < 0:
        days = 0
    return _RECENCY_WEIGHT * (0.5 ** (days / _RECENCY_HALF_LIFE_DAYS))


def time_question_kind(query: str) -> str | None:
    """问句是不是在问时间：``"duration"``（多久）/ ``"date"``（何时）/ ``None``。"""
    if _DURATION_Q_RE.search(query):
        return "duration"
    if _DATE_Q_RE.search(query):
        return "date"
    return None


def _content_words(text: str) -> set[str]:
    """取出可做词面匹配的实词：小写、去停用词、长度 ≥ 3。"""
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _lexical_time_bonus(q_words: set[str], want_dur: bool, want_date: bool,
                        mem_text: str) -> tuple[float, bool]:
    """一条记忆的「词面重合 + 时间类型」加分，返回 ``(加分, 是否命中时间类型)``。"""
    if not q_words:
        return 0.0, False
    overlap = len(q_words & _content_words(mem_text)) / len(q_words)
    bonus = _LEX_WEIGHT * overlap
    # 时间加分只给「和问题沾边」的记忆，否则库里所有带日期的条目会被一并抬上来
    time_hit = bool(overlap > 0 and (
        (want_dur and _DURATION_RE.search(mem_text))
        or (want_date and _DATE_RE.search(mem_text))
    ))
    if time_hit:
        bonus += _TIME_WEIGHT
    return bonus, time_hit


@runtime_checkable
class TextEmbedder(Protocol):
    """将文本批量编码为向量（与存储维度一致）。"""

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class MemorySearchHit:
    memory_id: str
    text: str
    score: float
    attributed_to: str
    metadata: dict[str, Any]
    # 加分前的纯余弦。上层据此区分"语义本来就相关"和"靠词面/时间加分才上来"的，
    # 好让加分只用于救回被埋的记忆，而不是把语义最相关的挤下去。
    base_score: float = 0.0
    # 这条命中了问题所问的时间类型（问"多久"且自带时长表达 / 问"何时"且自带日期）
    time_boost: bool = False
    #: 这条记忆记的事**发生在哪天**（YYYY-MM-DD，来自 Ingest 的 observed_at）。
    #: 没有它，"这事在那事之前吗"这类问题就没有依据——事实正文里通常只有
    #: "上周""一个多月了"这种相对说法，脱离绝对日期推不出先后。
    observed_at: str = ""
    #: [CONTROLLED FORK - patch VM-LOCAL-008] 非空 = 这条观察已被更新的事实取代
    #: （新行 id）。行本身保留、可检索（历史），检索排序时排在取代者之后
    #: （当前值优先）。与 right-brain 的 superseded_by 元数据同一语义。
    superseded_by: str = ""
    #: [CONTROLLED FORK - patch VM-LOCAL-011] 时效加分（事件时间越近越高，
    #: 见 recency_bonus）。参与排序：base_score + recency_boost。
    recency_boost: float = 0.0
    #: [CONTROLLED FORK - patch VM-LOCAL-012] 同一观察被独立确认的次数
    #: （"我对花生过敏" 说过 5 次 → 5）。首次入库不带该键（渲染端按 1 处理），
    #: 重复确认经 count_occurrence() 累加。上次观察到的时间另存
    #: last_observed_at（ISO 时间戳）。
    occurrence_count: int = 0
    last_observed_at: str = ""


@dataclass
class OpenAILocalEmbedderConfig:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None  # skip probe if known

    def resolved_model(self) -> str:
        return (
            self.model
            or os.environ.get("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small"
        ).strip()


class OpenAILocalEmbedder:
    """OpenAI Embeddings API（开发默认 ``text-embedding-3-small``）。"""

    def __init__(self, config: OpenAILocalEmbedderConfig | None = None) -> None:
        self._cfg = config or OpenAILocalEmbedderConfig()
        self._model = self._cfg.resolved_model()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("本地向量需要: pip install openai>=1.0") from e

        api_key = self._cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY（或 OpenAILocalEmbedderConfig(api_key=...)）")

        # timeout 必须显式设：openai 客户端默认超时长达 10 分钟，一个悬住的
        # 请求会让整条流水线停摆
        kw: dict[str, Any] = {"api_key": api_key, "timeout": 60.0, "max_retries": 2}
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        self._client = OpenAI(**kw)

        # Use pre-known dimensions to skip the probe API call
        self._dims = self._cfg.dimensions if self._cfg.dimensions else self._probe_dimensions()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims

    def _probe_dimensions(self) -> int:
        v = self.embed_texts(["probe"])[0]
        return len(v)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # 同一段文本在一次 ingest 里会被反复要（fact 3 次、实体 2 轮、slot 描述
        # 每轮重算），缓存掉。向量是确定的，不改变语义。
        from voicemem.utils.common import embed_cache
        return embed_cache.resolve(self._model, texts, self._embed_uncached)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        kw = {"model": self._model, "input": texts, "encoding_format": "float"}
        # 走 OpenRouter 时把 provider 钉死在 OpenAI：实测它会把 text-embedding-3-small
        # 偶发路由到 Google（gemini-embedding，3072 维），整库向量维度就变了，之后的
        # 查询全部 "shapes not aligned"。allow_fallbacks=False 宁可报错也不换供应商。
        if "openrouter" in str(getattr(self._client, "base_url", "") or "").lower():
            kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        r = self._client.embeddings.create(**kw)
        _exp = int(os.environ.get("VOICEMEM_EMBED_DIM", "1536"))
        if r.data and len(r.data[0].embedding) != _exp:
            raise RuntimeError(
                f"embedding 维度 {len(r.data[0].embedding)} != 期望 {_exp}（model={self._model}，"
                f"响应 model={getattr(r, 'model', '?')}）——供应商被换掉了，拒绝写入以免污染库")
        # 部分 OpenAI 兼容后端（如 Gemini）批量返回时 index 是 None，
        # 只在 index 齐全时才按它重排，否则保持 API 返回顺序
        data = r.data
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [list(map(float, row.embedding)) for row in data]


def mock_embedder(dim: int = 8, seed: int = 0) -> TextEmbedder:
    """确定性伪向量，仅供单元测试（无需 API）。"""

    class _Mock:
        model_name = "mock-deterministic"
        dimensions = dim

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            state = seed
            for s in texts:
                vec: list[float] = []
                x = state + sum(ord(c) for c in s[:200])
                for j in range(dim):
                    x = (1103515245 * x + 12345) % (2**31)
                    vec.append(float(x % 1000) / 1000.0)
                state = x
                out.append(vec)
            return out

    return _Mock()
