"""把问句里的相对时间词展开成绝对日期，再拿去检索。

为什么需要：抽取会把"下周三下午三点体检"归一成"Jiaqi 将在 2026年8月26日（周三）
下午三点进行体检"——库里存的是绝对日期。而用户问的是"我下周有什么安排"，这句话里
**一个绝对日期都没有**，向量对不上，实测三条下周日程一条都检索不到::

    「我下周有什么安排」   日程命中 0/3
    「8月26号我要干嘛」    日程命中 3/3

差别只在问法。所以在检索前把"下周"就地展开成那七天的日期，拼在问句后面——
向量里有了 8月26日 这样的字面，才够得着库里那条。

只改**拿去检索的那份文本**，不改用户说的话，也不写进记忆。

    expand_relative_dates("我下周有什么安排")
    → "我下周有什么安排（2026年8月24日 2026年8月25日 … 2026年8月30日）"

识别不到相对时间词就原样返回，一分钱不花（纯正则，无模型）。

[CONTROLLED FORK - patch VM-LOCAL-010] HU/EN temporal cues (external audit
v0.6.3 finding CD-3, P1): the Chinese-only word set meant a Hungarian
assistant was deaf to its own users' relative time — Parakeet transcribes
"mi lesz jövő héten" / "what did I say yesterday" correctly, but retrieval
never expanded those cues, so temporal queries missed their dates entirely.
This patch adds the Hungarian and English day/few-day/week vocabulary as a
SECOND, word-boundary regex (Latin scripts must not substring-match: "ma"
inside "magyar" would be garbage), case-insensitive. Latin-script queries
get ISO "2026-09-14" stamps (the format LLM normalization and the store's
date parsers both understand — see local_memory_store.parse_date_values);
CJK queries keep the original 中文 stamp format, unchanged.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

#: 相对时间词 → (起始偏移, 天数)。偏移是相对"今天"的天数。
#: 周相关的偏移在 _resolve 里按当天星期几现算，这里用 None 占位。
_SPANS: dict[str, tuple[int | None, int]] = {
    "前天":     (-2, 1),
    "昨天":     (-1, 1),
    "今天":     (0, 1),
    "今日":     (0, 1),
    "明天":     (1, 1),
    "明日":     (1, 1),
    "后天":     (2, 1),
    "大后天":   (3, 1),
    "这几天":   (0, 3),
    "最近几天": (-3, 4),
    "接下来几天": (0, 4),
    "未来几天": (0, 4),
    # 周：偏移按当天星期几算
    "上周":     (None, 7),
    "上个星期": (None, 7),
    "这周":     (None, 7),
    "本周":     (None, 7),
    "这个星期": (None, 7),
    "下周":     (None, 7),
    "下个星期": (None, 7),
    "下星期":   (None, 7),
}

#: 周相关的词 → 相对"本周一"的周偏移
_WEEK_OFFSET = {
    "上周": -1, "上个星期": -1,
    "这周": 0, "本周": 0, "这个星期": 0,
    "下周": 1, "下个星期": 1, "下星期": 1,
}

#: [VM-LOCAL-010] 拉丁字母（匈牙利语 + 英语）天/近几天词 → (起始偏移, 天数)。
#: 只用带词边界（\b）的独立词匹配——"ma" 绝不能命中 "magyar" 的中间。
_LATIN_SPANS: dict[str, tuple[int, int]] = {
    # ── magyar napok ──
    "tegnapelőtt":  (-2, 1),
    "tegnap":       (-1, 1),
    "ma":           (0, 1),
    "holnapután":   (2, 1),
    "holnap":       (1, 1),
    # ── English days ──
    "the day before yesterday": (-2, 1),
    "day after tomorrow":       (2, 1),
    "yesterday":    (-1, 1),
    "today":        (0, 1),
    "tomorrow":     (1, 1),
    # ── magyar "néhány nap" ──
    "az elmúlt napokban":     (-3, 4),
    "elmúlt napokban":        (-3, 4),
    "az elkövetkező napokban": (0, 4),
    "elkövetkező napokban":   (0, 4),
    "következő napokban":     (0, 4),
    "az elkövetkezendő napokban": (0, 4),
    "elkövetkezendő napokban": (0, 4),
    # ── English few-day spans ──
    "in the coming days":  (0, 4),
    "in the next few days": (0, 4),
    "next few days":       (0, 4),
    "past few days":       (-3, 4),
    "the past few days":   (-3, 4),
    "in recent days":      (-3, 4),
}

#: [VM-LOCAL-010] 拉丁字母周词（HU 各格 + EN）→ 相对"本周一"的周偏移。
#: 匈牙利语常合写（"jövőhéten"）和分写（"jövő héten"）——两种都收录。
_LATIN_WEEK_OFFSET = {
    # ── magyar ──
    "múlt héten": -1, "múlt hét": -1, "múlthéten": -1,
    "előző héten": -1, "előző hét": -1, "előzőhéten": -1,
    "ezen a héten": 0, "ezen héten": 0, "ez a hét": 0, "ezhéten": 0,
    "aktuális hét": 0,
    "jövő héten": 1, "jövő hét": 1, "jövőhéten": 1,
    "következő héten": 1, "következő hét": 1, "következőhéten": 1,
    "elkövetkező héten": 1, "elkövetkezendő héten": 1,
    # ── English ──
    "last week": -1, "the last week": -1, "past week": -1, "the past week": -1,
    "this week": 0, "the current week": 0,
    "next week": 1, "the next week": 1, "coming week": 1, "the coming week": 1,
}

#: 长的词要先匹配，否则"下个星期"会先被"下周"之外的短词切碎。
_WORDS = sorted(_SPANS, key=len, reverse=True)
_RE = re.compile("|".join(re.escape(w) for w in _WORDS))

#: [VM-LOCAL-010] 拉丁词表按长度降序 + 词边界 + 大小写不敏感。
_LATIN_WORDS = sorted({**_LATIN_SPANS, **_LATIN_WEEK_OFFSET}, key=len, reverse=True)
_LATIN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _LATIN_WORDS) + r")\b",
    re.IGNORECASE,
)

#: 合并视图（_resolve 用）：所有词 → 天数跨度；所有周词 → 周偏移。
_SPANS_ALL: dict[str, tuple[int | None, int]] = {**_SPANS, **_LATIN_SPANS}
_WEEK_ALL: dict[str, int] = {**_WEEK_OFFSET, **_LATIN_WEEK_OFFSET}

#: 一次最多展开几个日期。问"最近三个月"这种展开出来上百个日期，
#: 会把问句本身的语义冲淡，反而检索更差。
_MAX_DAYS = 8


def _resolve(word: str, today: date) -> list[date]:
    """一个相对时间词覆盖哪几天。"""
    if word in _WEEK_ALL:
        monday = today - timedelta(days=today.weekday())      # 本周一
        start = monday + timedelta(weeks=_WEEK_ALL[word])
        return [start + timedelta(days=i) for i in range(7)]
    offset, days = _SPANS_ALL[word]
    start = today + timedelta(days=offset or 0)
    return [start + timedelta(days=i) for i in range(days)]


def expand_relative_dates(query: str, today: date | None = None) -> str:
    """在问句后面补上它涉及的绝对日期。没有相对时间词就原样返回。"""
    if not query:
        return query
    today = today or date.today()
    # [VM-LOCAL-010] 两套词表分别扫：CJK 无边界（中文没有"词边界"概念），
    # 拉丁字母 \b + 忽略大小写。哪个命中了决定日期戳的格式。
    cjk_words = _RE.findall(query)
    latin_words = [m.group(0).lower() for m in _LATIN_RE.finditer(query)]
    # a regex a nap- ES a het-szavakat is eszleli; a szotar-lookup mindketto
    # föle kell nezzen (_SPANS_ALL = napok, _WEEK_ALL = hetek)
    latin_words = [w for w in latin_words
                   if w in _SPANS_ALL or w in _WEEK_ALL]
    words = list(cjk_words) + latin_words
    if not words:
        return query

    days: list[date] = []
    for word in words:
        for d in _resolve(word, today):
            if d not in days:
                days.append(d)
    if not days or len(days) > _MAX_DAYS:
        return query

    days.sort()
    if latin_words:
        # 拉丁字母问句 → ISO 日期戳：与 LLM 归一化产物、store 的
        # parse_date_values()、mem0 created_at 同一族格式，字面命中与
        # 值级比对（date_overlap_bonus_values）都能吃到。
        stamps = " ".join(d.isoformat() for d in days)
        return f"{query} ({stamps})"
    # 写成"2026年8月26日"这种格式，跟抽取归一后的写法对齐——格式不一样就白展开了。
    stamps = " ".join(f"{d.year}年{d.month}月{d.day}日" for d in days)
    return f"{query}（{stamps}）"
