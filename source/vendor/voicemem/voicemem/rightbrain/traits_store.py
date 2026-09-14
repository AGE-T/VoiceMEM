"""右脑 v2：一个节点 = 一条关于这个人的判断，证据挂在下面。

    rb_traits                          节点
      claim      压力大时想被安抚        写入时就定好，5-15 字
      slot       五选一（情绪/应对方式/表达风格/思维模式/喜好与厌恶）
      embedding  claim 的向量 —— 右脑终于能按语义检索
    rb_evidence                        证据
      quote      你先别给方案，让我说完   用户原话
      emotion    烦躁                  情绪是证据的属性，不是节点
      cause_id   ← 左脑那条 fact

**为什么要换掉 slot → entity → heartnote 那套**：``entity`` 那一层身兼三职——
有时是关于人的判断（"讨厌被打断"），有时是话题（"手冲咖啡""NUS"），有时是情绪词
（"焦虑"）。三种东西混在一层，后果是实测到的这些：

  · 所有悲伤的事堆进「悲伤」一个节点（61 条），所有话都链到「佳琪」（52 条）
  · 标题格式无法统一——三类东西本来就没有统一写法
  · 描述靠事后的巩固批处理补，跑得少，于是大量节点没描述

这里只放**关于人的判断**。话题实体归左脑的认知图；情绪降级成证据的属性；
助手的自我复盘（response_experience）不进这张表。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# [VM-LOCAL-015] 立场扫描器：纯函数模块（只依赖 re），模块级导入——
# 不能在方法体内懒导入：测试套件里有模拟「包不存在」的降级场景会从
# sys.modules 里摘掉 voicemem（见 tests/unit/test_voicemem_bridge 的
# degraded 模拟），方法级 from-import 在运行时重新解析包名会炸。
from voicemem.rightbrain import stance as _stance

#: 两条 claim 像到这个程度就算同一条，证据并进去。
#: 0.95 是实测出来的分界：本地 E5 对中文短语的基线相似度就有 0.9+，
#: 「喜欢手冲咖啡」↔「偏好手冲咖啡」是 0.964（该合并），
#: 「讨厌吃饭吧唧嘴」↔「讨厌被打断」是 0.934（不该合并）。
MERGE_THRESHOLD = 0.95

#: [CONTROLLED FORK - patch VM-LOCAL-015] 反义取代带（v0.9.0 语义门）。
#: 实测（scripts/measure_semantic_matrix.py，本地 multilingual-e5-small）：
#: 直接否定/反义对聚在 0.90–0.95（C-negation-en 0.9142 … hu2 0.9496，
#: claim-pos-neg 0.9198，antipathy-en 0.9066，flip-back 0.92-0.95），
#: 且部分反义对（utálja 0.9685、used to 0.9865）直接落在 0.95 合并带内
#: ——纯余弦无法区分「同意」与「反对」。取代决策因此用独立的、更宽的
#: 相似带：立场相反 + 带内相似 + 话题词重叠才允许翻转。
#: 0.90 = 实测反义簇的下沿（antipathy-en 0.9066、stopped liking 0.9023）。
SUPERSEDE_MIN_SIM = 0.90

#: 预设性否定（"no longer"/"gave up"/"már nem" —— 语用上预设了它要
#: 取代的前状态）允许更宽的带：0.88 = 实测 E5 话题命中下沿（代码库
#: 自己的门槛校准：真命中 0.89~0.92、噪音 0.82~0.86 → 0.88）。
#: "gave up coffee"↔"likes coffee" 实测 0.8824，恰好需要这条带。
#: 话题词重叠守卫（stance.topic_overlaps）保证宽带不跨话题：
#: "likes motorcycles"↔"no longer likes bicycles"（实测 0.8876）不重叠 → 不取代。
SUPERSEDE_MIN_SIM_PRESUPPOSITION = 0.88

#: [CONTROLLED FORK - patch VM-LOCAL-013] 置信度动力学（外部审计 v0.5.0 F-D：
#: confidence 写入后永远不动、排序也不读它——「惰性字段」）。
#: 强化规则：每次同一判断被再次确认，confidence 渐近上移、永不封顶到 1：
#:     c' = c + (1 - c) * TRAIT_REINFORCE_STEP
#: 0.9 → 0.93 → 0.951 → 0.9657…（确认越多越接近 1，但永远不会等于 1）。
TRAIT_REINFORCE_STEP = 0.30

#: [CONTROLLED FORK - patch VM-LOCAL-013] 读时衰减：90 天观察宽限期后按
#: 180 天半衰期指数衰减（与左脑 VM-LOCAL-011 的「未标注旧行不惩罚」同一
#: 策略：last_seen 解析不出来的 legacy 行拿到权重 1.0，永不受罚）。
TRAIT_DECAY_GRACE_DAYS = 90.0
TRAIT_DECAY_HALFLIFE_DAYS = 180.0


def effective_trait_confidence(confidence: float, last_seen: str) -> float:
    """[CONTROLLED FORK - patch VM-LOCAL-013] 读时置信度 = 写入置信度 × 时间衰减。

    宽限期内不衰减；之后按半衰期指数下降。加法式、非破坏性：只影响排序
    权重，数据库里的原始 confidence 不变。legacy 行（无 last_seen）返回原值。
    """
    try:
        if last_seen:
            from datetime import datetime, timezone
            age_days = (datetime.now(timezone.utc)
                        - datetime.fromisoformat(last_seen)).total_seconds() / 86400.0
            if age_days > TRAIT_DECAY_GRACE_DAYS:
                excess = age_days - TRAIT_DECAY_GRACE_DAYS
                confidence = confidence * (0.5 ** (excess / TRAIT_DECAY_HALFLIFE_DAYS))
    except Exception:
        pass                      # 解析失败 → 原值（宽客策略，同 VM-LOCAL-011）
    return float(min(max(confidence, 0.0), 1.0))


#: [CONTROLLED FORK - ports upstream 91d2e42 + VM-LOCAL-004]
#: 维度不符只提醒一次，别每轮刷屏（upstream 用 print；这里走 logging）。
_WARNED_DIM: set = set()

#: [CONTROLLED FORK - patch VM-LOCAL-004] _vec 失败只警这一次（见 _vec）。
_WARNED_VEC: set = set()

#: 五个 slot。去掉了原来的「人物地点态度」——它存的是话题（手冲咖啡/NUS/佳琪），
#: 本来就该在左脑，也正是「佳琪 ×52」那个大杂烩的来源。
SLOTS = ("情绪", "应对方式", "表达风格", "思维模式", "喜好与厌恶")

#: 五个 slot → UI 的三类
SLOT_TO_CLUSTER = {
    "情绪":       "emotion",
    "应对方式":    "personality",
    "表达风格":    "personality",
    "思维模式":    "personality",
    "喜好与厌恶":  "preference",
}


@dataclass
class Evidence:
    quote: str
    emotion: str = ""
    cause: str = ""            # 左脑 fact 的原文（渲染时当"为什么"）
    cause_id: str = ""
    at: str = ""


@dataclass
class Trait:
    id: str
    slot: str
    claim: str
    confidence: float = 0.9
    evidence: list[Evidence] = field(default_factory=list)
    updated_at: str = ""
    # [CONTROLLED FORK - patch VM-LOCAL-013] 观察记账（外部审计 F-C：
    # first_seen / last_seen / occurrence_count 此前不存在——
    # 「多久/最近一次」类问题无从回答）。合并时递增，首次写入时 1。
    first_seen: str = ""
    last_seen: str = ""
    occurrence_count: int = 1
    # [CONTROLLED FORK - patch VM-LOCAL-015] 语义状态（v0.9.0）。
    # stance：这条判断写入时的立场分类（stance.py 的枚举：pos/neg/
    # past/qualified/uncertain，空=中立/未知）；superseded_by/superseded_at：
    # 被哪条新观察取代、何时——与左脑 VM-LOCAL-008 和 heartnote 的
    # run_cleanup 取代链同构（追加式，永不改写旧行正文）。
    stance: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    superseded_at: str = ""

    @property
    def cluster(self) -> str:
        return SLOT_TO_CLUSTER.get(self.slot, "personality")

    @property
    def eff_confidence(self) -> float:
        """读时置信度（含时间衰减，见 :func:`effective_trait_confidence`）。"""
        return effective_trait_confidence(self.confidence, self.last_seen)


#: claim 前面常见的主语。节点标题是「讨厌被打断」而不是「用户讨厌被打断」——
#: 整张图讲的都是同一个人，每个标题都顶着「用户」两个字纯属噪音。
_SUBJECTS = ("用户可能", "用户似乎", "用户倾向于", "用户", "他/她", "对方", "我")


def normalize_claim(claim: str) -> str:
    """把 claim 收拾成节点标题该有的样子：无主语、无句号、一句短话。

    几条写入路径的产出质量不一样——合并抽取那条有明确格式要求，助手复盘那条
    （response_experience 的 user_trait）没有，实测吐出过
    「用户喜欢分享自己的经历，可能不太关注助手的问候。」这种带主语的整句。
    与其在每条路径上各写一遍要求，不如在入口统一收口。
    """
    c = (claim or "").strip().strip("「」\"'").rstrip("。.！!；;，,")
    for s in _SUBJECTS:
        if c.startswith(s) and len(c) > len(s) + 2:
            c = c[len(s):].lstrip("，,、 ")
            break
    # 「A，可能B」这种双句只留前半句——后半句几乎都是模型加的推测
    if "，" in c and len(c) > 15:
        head = c.split("，")[0].strip()
        if len(head) >= 5:
            c = head
    return c.strip()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(ts: str):
    """[VM-LOCAL-015] 容错解析 ISO 时间戳（日期或完整时间）；失败返回 None。"""
    s = (ts or "").strip()
    if not s:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class TraitStore:
    """rb_traits / rb_evidence 两张表，跟其余结构化存储共用 space 那个 sqlite。"""

    def __init__(self, db_path, embed) -> None:
        self._db = str(db_path)
        self._embed = embed                 # fn(text) -> list[float]
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS rb_traits (
                id             TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                slot           TEXT NOT NULL,
                claim          TEXT NOT NULL,
                embedding      BLOB,
                confidence     REAL NOT NULL DEFAULT 0.9,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS rb_evidence (
                id         TEXT PRIMARY KEY,
                trait_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                quote      TEXT NOT NULL,
                emotion    TEXT NOT NULL DEFAULT '',
                cause      TEXT NOT NULL DEFAULT '',
                cause_id   TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )""")
            # [CONTROLLED FORK - patch VM-LOCAL-013] 旧库加法迁移：观察记账列。
            # PRAGMA 探测 + ALTER TABLE，幂等（重跑安全），旧行拿到默认值
            # （first_seen/last_seen 空，occurrence_count 1——读到时按宽客策略）。
            have = {row[1] for row in c.execute("PRAGMA table_info(rb_traits)")}
            for col, ddl in (
                ("first_seen", "ALTER TABLE rb_traits ADD COLUMN first_seen TEXT NOT NULL DEFAULT ''"),
                ("last_seen", "ALTER TABLE rb_traits ADD COLUMN last_seen TEXT NOT NULL DEFAULT ''"),
                ("occurrence_count",
                 "ALTER TABLE rb_traits ADD COLUMN occurrence_count INTEGER NOT NULL DEFAULT 1"),
                # [VM-LOCAL-015] 语义状态列：stance（写入时立场分类）+
                # superseded_by/superseded_at（取代链，同 VM-LOCAL-008 的
                # 标记模式）。旧行默认值：stance 空（读时按 claim 文本懒推
                # 断），未被取代。
                ("stance", "ALTER TABLE rb_traits ADD COLUMN stance TEXT NOT NULL DEFAULT ''"),
                ("supersedes", "ALTER TABLE rb_traits ADD COLUMN supersedes TEXT NOT NULL DEFAULT ''"),
                ("superseded_by", "ALTER TABLE rb_traits ADD COLUMN superseded_by TEXT NOT NULL DEFAULT ''"),
                ("superseded_at", "ALTER TABLE rb_traits ADD COLUMN superseded_at TEXT NOT NULL DEFAULT ''"),
            ):
                if col not in have:
                    c.execute(ddl)
            c.execute("CREATE INDEX IF NOT EXISTS idx_ev_trait ON rb_evidence(trait_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tr_user ON rb_traits(user_id, slot)")

    def _conn(self):
        c = sqlite3.connect(self._db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ── 写 ────────────────────────────────────────────────────────────────────

    def add(self, user_id: str, slot: str, claim: str, ev: Evidence) -> str:
        """加一条判断 + 它的证据。

        [CONTROLLED FORK - patch VM-LOCAL-015] v0.9.0 语义门：合并决策不再
        只看余弦。新观察先过立场分类（stance.py，确定性 0-LLM），再在
        三个**显式**结果里选一个：

        * MERGE（强化） —— 同立场（或中立入参）且相似 ≥ 0.95：
          证据追加、occurrence_count 递增、confidence 渐近上移（v0.8.0
          语义完全保留）；
        * SUPERSEDE（取代） —— 立场相反（pos↔neg）且相似落在取代带
          （≥0.90；预设性否定 ≥0.88）且话题词重叠且时间不倒退：新行
          成为当前状态，旧行**只加** superseded_by/superseded_at 标记
          （与左脑 VM-LOCAL-008 同构，永不改写正文），旧行的
          occurrence/confidence 冻结——矛盾绝不作为强化计数；
        * SEPARATE（分离） —— 其余一切（限定句/过去时/不确定/立场
          异类/带外相似）：新行独立成节点，各自带证据共存，推迟裁决。

        已经被取代的行不参与匹配（活跃链外）；当前活跃行被翻转后，
        再来的同立场观察沿取代链自然前进（S←N←R 链完整保留在行里）。
        """
        claim = normalize_claim(claim)
        if not claim or slot not in SLOTS:
            return ""

        # [VM-LOCAL-015] 立场：claim 和用户原话（quote）都扫——抽取器
        # 可能把否定从标签里归一掉，但原话是地面真相（ground truth）。
        s_in = _stance.observation_stance(claim, ev.quote or "")

        vec = self._vec(claim)
        match = self._best_active_match(user_id, slot, vec, claim, s_in, ev)
        now = _now()
        with self._conn() as c:
            historical = None
            if match is not None and match[3]:
                # [VM-LOCAL-015] 倒退重放：翻转被守卫拒绝，但这条观察与
                # 链上某个**已被取代**的同立场行同题（≥0.95，合并级身份）——
                # 老陈述的证据归老状态：仅追加证据到那条历史行，行本身
                # 保持冻结（不改 occurrence/confidence、不复活）。当前
                # 状态不被污染（重放不再以「当前」面目出现）。
                historical = self._superseded_agreeing_match(
                    c, user_id, slot, vec, s_in)
            if historical is not None:
                tid = historical        # evidence-only attach (frozen row)
            elif match is None or match[3]:
                # SEPARATE（或首见 / 被守卫拒绝且链上无可挂的历史行）：
                # 新节点，自己的立场随行记录。倒退重放绝不合并进当前行。
                tid = self._insert_trait(c, user_id, slot, claim, vec, s_in, now)
            else:
                tid, target_stance, supersede = match[0], match[1], match[2]
                if supersede:
                    # SUPERSEDE：新行是当前状态；旧行只加标记（非破坏）。
                    new_id = self._insert_trait(c, user_id, slot, claim, vec, s_in,
                                                now, supersedes=tid)
                    c.execute(
                        "UPDATE rb_traits SET superseded_by=?, superseded_at=? "
                        "WHERE id=?",
                        (new_id, now, tid))
                    tid = new_id
                else:
                    # MERGE：[VM-LOCAL-013] 非破坏性记账——证据照旧追加
                    # （下方 INSERT 不变），节点上递增 occurrence_count、
                    # 刷新 last_seen、渐近强化 confidence（旧值永不丢失）。
                    row = c.execute(
                        "SELECT confidence, occurrence_count, first_seen FROM rb_traits WHERE id=?",
                        (tid,)).fetchone()
                    prev_conf = float(row["confidence"]) if row is not None else 0.9
                    prev_occ = int(row["occurrence_count"]) if row is not None else 1
                    prev_first = (row["first_seen"] if row is not None else "") or now
                    new_conf = prev_conf + (1.0 - prev_conf) * TRAIT_REINFORCE_STEP
                    c.execute(
                        "UPDATE rb_traits SET updated_at=?, last_seen=?, "
                        "occurrence_count=?, confidence=?, first_seen=?, stance=? WHERE id=?",
                        (now, now, prev_occ + 1, round(new_conf, 4), prev_first,
                         target_stance or s_in, tid))
            # 每个字段都过一遍 str()：证据常常来自旧数据或 LLM 输出，
            # 缺字段时是 None，而这几列都是 NOT NULL，直接插会整轮写入失败。
            c.execute("INSERT INTO rb_evidence "
                      "(id,trait_id,user_id,quote,emotion,cause,cause_id,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, tid, user_id, str(ev.quote or ""),
                       str(ev.emotion or ""), str(ev.cause or ""),
                       str(ev.cause_id or ""), str(ev.at or now)))
        return tid

    def _insert_trait(self, c, user_id, slot, claim, vec, stance_val, now,
                      supersedes: str = "") -> str:
        """[VM-LOCAL-015] 新建一行判断（SEPARATE/SUPERSEDE 共用）。"""
        tid = uuid.uuid4().hex
        c.execute("INSERT INTO rb_traits "
                  "(id,user_id,slot,claim,embedding,confidence,created_at,updated_at,"
                  "first_seen,last_seen,occurrence_count,stance,supersedes) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (tid, user_id, slot, claim,
                   vec.astype(np.float32).tobytes() if vec is not None else None,
                   0.9, now, now, now, now, 1, stance_val, supersedes))
        return tid

    def _best_active_match(self, user_id, slot, vec, claim, s_in, ev):
        """[VM-LOCAL-015] 在**活跃**（未被取代）行里找最佳匹配并裁决。

        返回 ``(trait_id, target_stance, supersede: bool)``；无候选返回
        None。裁决依据（全部确定性，0 LLM）：

        1. 候选只在取代带以上才被视为「同一话题」（比合并带更宽——
           实测反义对聚在 0.90-0.95）；带宽由预设性否定决定；
        2. 立场相反（pos↔neg）且话题词重叠且时间不倒退 → supersede；
        3. 立场一致/中立 → 相似 ≥ 0.95 才 merge（v0.8.0 行为不变）；
        4. 其余（限定/过去/不确定的异类立场、带外）→ None（SEPARATE）。

        旧行的 stance 列优先；legacy 空值按存量 claim 文本懒推断。
        """
        if vec is None:
            return None
        presup = (_stance.is_presuppositional(claim)
                  or _stance.is_presuppositional(ev.quote or ""))
        band = (SUPERSEDE_MIN_SIM_PRESUPPOSITION if presup
                else SUPERSEDE_MIN_SIM)
        best, best_sim, best_row = None, 0.0, None
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, claim, stance, embedding, superseded_by, last_seen FROM rb_traits "
                "WHERE user_id=? AND slot=? AND embedding IS NOT NULL",
                (user_id, slot)).fetchall()
        for r in rows:
            if (r["superseded_by"] or ""):
                continue          # 已取代的行不参与匹配（活跃链外）
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim, best_row = r["id"], sim, r
        if best is None or best_sim < band:
            return None

        target_claim = str(best_row["claim"] or "")
        s_tgt = str(best_row["stance"] or "")
        if not s_tgt:
            s_tgt = _stance.classify_stance(target_claim)   # legacy 懒推断

        flip = {s_in, s_tgt} == {"pos", "neg"}
        if flip:
            if not (best_sim >= band
                    and _stance.topic_overlaps(claim, target_claim)):
                return None           # 异义但不满足取代条件 → SEPARATE
            if self._is_stale_replay(ev, best_row):
                # 倒退重放：既不翻转也不强化——调用方会找链上的历史行
                # 挂证据（找不到才 SEPARATE）。
                return (best, s_tgt, False, True)
            return (best, s_tgt, True, False)

        # 同立场/中立：v0.8.0 合并语义（阈值不变）。
        agrees = (s_in == s_tgt or s_in == ""
                  or (s_tgt == "" and s_in == "pos"))
        if best_sim >= MERGE_THRESHOLD and agrees:
            return (best, s_tgt, False, False)
        return None

    def _superseded_agreeing_match(self, c, user_id, slot, vec, s_in):
        """[VM-LOCAL-015] 链上**已被取代**的同立场最佳匹配（≥ 合并阈值）。

        重放的老陈述找到它原本所属的历史行：证据挂过去，行保持冻结。
        返回 trait_id 或 None（调用方降级为 SEPARATE）。
        """
        if vec is None:
            return None
        best, best_sim = None, 0.0
        rows = c.execute(
            "SELECT id, claim, stance, embedding FROM rb_traits "
            "WHERE user_id=? AND slot=? AND embedding IS NOT NULL "
            "AND superseded_by != ''",
            (user_id, slot)).fetchall()
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            row_stance = str(r["stance"] or "")
            if not row_stance:
                row_stance = _stance.classify_stance(str(r["claim"] or ""))
            if s_in and row_stance and s_in != row_stance:
                continue          # 只要同立场（中立行任意立场都可挂）
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim = r["id"], sim
        return best if best_sim >= MERGE_THRESHOLD else None

    def _is_stale_replay(self, ev: Evidence, row) -> bool:
        """[VM-LOCAL-015] 倒退守卫：新观察的事件时间比目标行最近一次
        观察还旧 → 这是历史重放（老录音重捈、历史回放），不允许翻转
        当前状态，也不允许强化（→ 历史行挂证据或 SEPARATE）。

        比较的是**事件时间对事件时间**：目标行最近一次可解析的证据
        时间戳（evidence.created_at 用的就是 ev.at）；没有可解析的
        事件时间才退回写入墙钟 last_seen（混合时钟是已记录的限制：
        纯回填/无时间戳数据下，守卫对时间 fail-open——立场门才是硬
        安全；实时管道每轮都带 observed_at）。
        """
        incoming = _parse_iso(str(ev.at or ""))
        if incoming is None:
            return False
        target = None
        try:
            with self._conn() as c:
                latest = c.execute(
                    "SELECT MAX(created_at) FROM rb_evidence WHERE trait_id=?",
                    (row["id"],)).fetchone()
                target = _parse_iso(str(latest[0] or ""))
        except Exception:
            target = None
        if target is None:
            target = _parse_iso(str(row["last_seen"] or ""))
        if target is None:
            return False
        return incoming < target

    def _vec(self, text: str):
        # [CONTROLLED FORK - patch VM-LOCAL-004]
        # Upstream swallowed EVERY exception here and returned None — a
        # failing embedder produced NULL rb_traits.embedding rows silently
        # (the v0.4.x field defect: traits visible in the UI, semantic
        # retrieval + merge dead). The failure is still non-fatal by design
        # (a trait must never block ingest), but it is now LOUD: one
        # warning per process names the embedder error so the operator can
        # fix it (scripts/assess_trait_embeddings.py measures the damage).
        try:
            v = np.asarray(self._embed(text), dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n else v
        except Exception as exc:
            global _WARNED_VEC
            if not _WARNED_VEC:
                _WARNED_VEC.add(1)
                import logging
                logging.getLogger(__name__).warning(
                    "rb_traits embedding FAILED (%s): trait rows will be "
                    "written with NULL embeddings — semantic trait retrieval "
                    "and merge are degraded until this is fixed (run "
                    "scripts/assess_trait_embeddings.py to measure)", exc)
            return None

    def _find_similar(self, user_id: str, slot: str, vec) -> str | None:
        if vec is None:
            return None
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding, superseded_by FROM rb_traits "
                             "WHERE user_id=? AND slot=? AND embedding IS NOT NULL",
                             (user_id, slot)).fetchall()
        # [VM-LOCAL-015] 已被取代的行不参与合并匹配（v0.8.x 会把已
        # 取代的旧状态再次强化——取代链要求只有活跃行可以被强化）。
        best, best_sim = None, 0.0
        for r in rows:
            if (r["superseded_by"] if "superseded_by" in set(r.keys()) else ""):
                continue
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim = r["id"], sim
        return best if best_sim >= MERGE_THRESHOLD else None

    # ── 读 ────────────────────────────────────────────────────────────────────

    def all(self, user_id: str, *, per_slot: int = 8) -> list[Trait]:
        """给脑图用：每个 slot 取证据最多的前几条 + 最近新增的几条。"""
        out: list[Trait] = []
        with self._conn() as c:
            for slot in SLOTS:
                rows = c.execute(
                    """SELECT t.*, COUNT(e.id) n FROM rb_traits t
                       LEFT JOIN rb_evidence e ON e.trait_id = t.id
                       WHERE t.user_id=? AND t.slot=? GROUP BY t.id
                       HAVING n > 0""", (user_id, slot)).fetchall()
                by_ev = sorted(rows, key=lambda r: -r["n"])
                by_new = sorted(rows, key=lambda r: r["updated_at"], reverse=True)
                fresh = max(1, per_slot // 2)
                picked, seen = [], set()
                # 一半给最近新增的（刚说的那句要能立刻看见），一半给证据最多的
                for r in by_new[:fresh] + by_ev:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    picked.append(r)
                    if len(picked) >= per_slot:
                        break
                for r in picked:
                    out.append(self._to_trait(c, r))
        return out

    def search(self, user_id: str, query: str, *, top_k: int = 5) -> list[Trait]:
        """按语义查判断。

        原来的右脑只能按情绪锚点匹配，所以每轮返回的总是同样那几条静态画像。
        claim 有了向量之后这里才是真正的检索。
        """
        return [t for t, _ in self.search_scored(user_id, query, top_k=top_k)]

    def search_scored(self, user_id: str, query: str, *, top_k: int = 5
                      ) -> list[tuple[Trait, float]]:
        """同 :meth:`search`，但带上余弦相似度。

        检索侧要用它当 priority——判断跟这句话有多相关，直接决定它该不该占
        top-N 的位置，固定 priority 会让不相关的判断挤掉真正相关的。
        """
        q = self._vec(query)
        if q is None:
            return []
        # [ports upstream f535f9d] 检索侧据此选门槛——阈值跟 embedder 绑，
        # 见 brain.trait_min_sim。
        self.last_query_dim = int(q.shape[0])
        with self._conn() as c:
            rows = c.execute("SELECT * FROM rb_traits WHERE user_id=? AND embedding IS NOT NULL",
                             (user_id,)).fetchall()
            scored, stale = [], 0
            for r in rows:
                v = np.frombuffer(r["embedding"], dtype=np.float32)
                if v.shape != q.shape:
                    # [ports upstream 91d2e42] 换过 embedder，老向量维度对不上
                    stale += 1
                    continue
                scored.append((float(v @ q), r))
            if stale and not _WARNED_DIM:
                _WARNED_DIM.add(1)
                # [CONTROLLED FORK] upstream 用 print；这里走 logging。
                import logging
                logging.getLogger(__name__).warning(
                    "rb_traits: %d trait vectors have a dimension that does "
                    "not match the current embedder — skipped. After an "
                    "embedder change old vectors are stale; re-embed them "
                    "(dry-run first: scripts/assess_trait_embeddings.py)",
                    stale)
            # [VM-LOCAL-015] v0.9.0 检索语义：当前状态优先（Phase 8——
            # 「现状查询不得让已取代的行排在当前行前面」，与左脑 VM-LOCAL-008
            # 的 (not superseded_by, base_score) 排序同构）。已取代的行保持
            # 可检索（历史可恢复），只是排到所有活跃命中之后。
            scored.sort(key=lambda t: (bool(t[1]["superseded_by"]
                                            if "superseded_by" in set(t[1].keys()) else ""),
                                       -t[0]))
            return [(self._to_trait(c, r), s) for s, r in scored[:top_k]]

    def _to_trait(self, c, r) -> Trait:
        evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=? ORDER BY created_at DESC",
                        (r["id"],)).fetchall()
        keys = set(r.keys())
        return Trait(
            id=r["id"], slot=r["slot"], claim=r["claim"],
            confidence=r["confidence"], updated_at=r["updated_at"],
            # [VM-LOCAL-013] 观察记账字段（旧库迁移后默认：空/1）。
            first_seen=(r["first_seen"] if "first_seen" in keys else "") or "",
            last_seen=(r["last_seen"] if "last_seen" in keys else "") or "",
            occurrence_count=(int(r["occurrence_count"]) if "occurrence_count" in keys else 1) or 1,
            # [VM-LOCAL-015] 语义状态字段（旧库迁移后默认：空）。
            stance=(r["stance"] if "stance" in keys else "") or "",
            supersedes=(r["supersedes"] if "supersedes" in keys else "") or "",
            superseded_by=(r["superseded_by"] if "superseded_by" in keys else "") or "",
            superseded_at=(r["superseded_at"] if "superseded_at" in keys else "") or "",
            evidence=[Evidence(quote=e["quote"], emotion=e["emotion"],
                               cause=e["cause"], cause_id=e["cause_id"],
                               at=e["created_at"]) for e in evs],
        )

    def counts(self, user_id: str) -> tuple[int, int]:
        with self._conn() as c:
            t = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id=?", (user_id,)).fetchone()[0]
            e = c.execute("SELECT COUNT(*) FROM rb_evidence WHERE user_id=?", (user_id,)).fetchone()[0]
        return t, e
