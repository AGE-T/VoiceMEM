"""把「抽事实 / 打标签 / 右脑标签」三次 LLM 调用并成一次。

实测一次 ingest 打 12 次 chat，其中这三样吃的是**同一句话**、输出互不依赖：

    extract_facts_openai   抽出 fact 文本
    CognitiveAnnotator     给 fact 打 slot / entity / relation
    _extract_rb_traits     从原话看出「这人什么样」

合成一次调用要解决一个次序问题：annotator 的输入是 extractor 的输出，看起来必须
串行。做法是让 extractor 一次把标注也吐出来，暂存在这里；annotator 和
_extract_rb_traits 先查暂存，查得到就直接用，查不到才走自己那次 LLM 调用。

这样不用改任何函数签名，而且**坏了会自动退回旧路径**——合并输出缺字段、解析失败、
或者模型没按格式来，下游照旧各自调用一次，行为跟合并前一致。

``VOICEMEM_MERGED_EXTRACTION=0`` 关掉合并。
"""
from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_annotations: dict[str, dict] = {}      # fact 文本 -> {slot, entities, relations}
_traits: dict[str, list] = {}           # 原话      -> [(slot, label), ...]
_MAX = 512


def enabled() -> bool:
    return os.environ.get("VOICEMEM_MERGED_EXTRACTION", "1") != "0"


def _put(store: dict, key: str, value) -> None:
    if not key:
        return
    with _lock:
        if len(store) >= _MAX:
            store.clear()
        store[key] = value


def _take(store: dict, key: str):
    """取出即删：一条 fact 只会被 annotate 一次，留着只会占内存。"""
    with _lock:
        return store.pop(key, None)


def put_annotation(fact_text: str, ann: dict) -> None:
    _put(_annotations, (fact_text or "").strip(), ann)


def take_annotation(fact_text: str) -> dict | None:
    return _take(_annotations, (fact_text or "").strip())


# 情绪和特质是**整段**的，不挂在某条 fact 上，而且抽取和右脑写入是同一轮、同一
# 个线程里前后脚发生的。原来按原话做 key，结果取不到——抽取拿到的原话带着
# "Speaker 0: " 前缀，跟右脑那边收到的 text 对不上。改成线程局部的「上一次」：
# 同一线程内前后脚匹配，多线程（评测并发跑）之间互不干扰。
_local = threading.local()


def put_emotion(utterance: str, emo: str) -> None:
    _local.emotion = emo


def take_emotion(utterance: str = "") -> str | None:
    v = getattr(_local, "emotion", None)
    _local.emotion = None          # 取出即清，别让上一轮的漏到下一轮
    return v


def put_traits(utterance: str, traits: list) -> None:
    _local.traits = traits


def take_traits(utterance: str = "") -> list | None:
    v = getattr(_local, "traits", None)
    _local.traits = None
    return v


# 追加到抽取 prompt 后面的那一段。刻意写得短——这段每次 ingest 都要发一遍，
# 而且原来的抽取 prompt 已经很长了。字段名跟 annotator 的输出格式保持一致，
# 下游解析代码一行都不用改。
PROMPT_ADDENDUM = """

Additionally, for EACH item in "memory", include these three fields:
- "slot": one of [{slots}]
- "entities": [{{"name": "...", "entity_type": "user|person|project|task|knowledge|preference|place|routine|asset|organization|event", "role": "subject|object|context|owner"}}]
- "relations": [{{"from": "...", "to": "...", "relation_type": "...", "confidence": 0.9}}]
  Relation direction must match reality: a boss manages the user, not the reverse.

And add ONE top-level field "traits": subjective things this utterance reveals about
the speaker. Each item {{"slot": "...", "label": "<3-8 words>"}}, slot is one of
EXACTLY these five values (copy the value verbatim - it is an enum):

  情绪        WHEN they feel WHAT - the situation plus the feeling it triggers.
              e.g. "gets nervous before reviews", "irritated when interrupted"
  应对方式     what they DO about a feeling, or how they want to be treated.
              e.g. "wants comfort when stressed", "prefers to be alone when upset"
  表达风格     habits of speaking and communicating.
              e.g. "gives examples before conclusions"
  思维模式     how they think, weigh things, decide.
              e.g. "weighs every option before deciding"
  喜好与厌恶   what they like or dislike.
              e.g. "dislikes long meetings"

情绪 vs 应对方式 is the one people get wrong: "irritated when interrupted" is
情绪 (a feeling appearing), "walks away when interrupted" is 应对方式 (an
action taken). If the label has no verb of doing or wanting in it, it is 情绪.

For "情绪" the label must read as **a pattern, not a bare feeling word**:
"gets nervous before reviews" - NOT "anxious" / "happy".
It becomes the title of a node on a graph; a bare word tells the user nothing.

When the utterance states a RECURRING tendency about the speaker - "always",
"every time", "never", "I am the kind of person who ...", or any habit or
reaction that clearly holds beyond this one moment - a trait is REQUIRED. "I
always drift off in long meetings" is 喜好与厌恶 "dislikes long meetings";
"I never sleep before a presentation" is 情绪 "sleepless before presentations".
Do not skip it just because the same content also went into "memory":
"memory" records WHAT HAPPENED, "traits" records WHAT THIS PERSON IS LIKE,
and one sentence very often carries both.

Outside that case, only include a category the utterance clearly shows -
for a one-off event or a plain question, "traits": [] is the right answer.

Each label becomes the TITLE of a node on a graph, so write it as a short
pattern IN ENGLISH (British spelling) - 3 to 8 words, no subject, no full
stop:
  good: hates being interrupted / wants comfort when stressed / conclusions before explanations
  bad: The user tends to plan in detail. (a full sentence with a subject)
  bad: I study computer science (copied from the utterance)

Also add ONE top-level field "emotion": how the speaker feels, as a single
English word (happy / calm / anxious / sad / wronged / angry / surprised /
tired / disappointed ...).
**Judge from what they actually say.** "I love strawberries" is happy, not
sad; "I am so angry" is angry, not anxious. If the utterance carries no clear
feeling (a plain fact, a question), return "" - an empty string is the right
answer far more often than a guess. A wrong label is worse than none: it gets
shown to the user as [label] next to their own words.

Never invent entities, traits or feelings that are not in the text.

Keep one-off requests OUT of "memory": asking for a recommendation, asking
what you remember about them, asking you to do something right now. When the
same sentence ALSO states a lasting fact, write only the lasting half - never
both in one item. "I have a GRE exam next week, any book recommendations?"
gives exactly one memory, "the user takes the GRE exam next week", and
nothing about the book request.

OUTPUT SHAPE — your JSON object must have EXACTLY these three top-level keys:
{{"memory": [...], "emotion": "...", "traits": [...]}}
The prompt above describes only the "memory" key. "emotion" and "traits" are
REQUIRED as well; omitting them is an error. Use "" and [] when there is nothing."""


def prompt_addendum() -> str:
    """追加到**用户消息**末尾的那一段。

    注意是用户消息，不是 system。放 system 末尾时模型只认里面的 per-item 字段
    （slot/entities 出得来），顶层的 emotion/traits 一律丢掉——用户消息里那份输出
    格式说明写死了顶层只有 "memory"，模型严格照做。于是右脑每轮拿到的都是
    emotion="" + traits=[]，write() 直接早退，脑图一个节点都不长。
    最后那段 OUTPUT SHAPE 就是为此显式重申顶层结构，别删。
    """
    from voicemem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES
    return PROMPT_ADDENDUM.format(slots=", ".join(ALL_SLOT_V2_VALUES))
