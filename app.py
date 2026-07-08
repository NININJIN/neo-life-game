from PIL import Image, ImageDraw
import numpy as np
import streamlit as st
import time
import pandas as pd
from functools import lru_cache

try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False

st.set_page_config(page_title="ネオライフゲーム", layout="wide")
st.title("ネオライフゲーム")
st.caption("哲学的な行動方針を持つ個体が、資源・密度・交尾制約・捕食・争奪・チーム配置の中でどのようにコピーを残すかを観察する進化シミュレーションです。")

st.markdown("""
<style>
/* ===== 全体：上下余白を削る（ただし上は欠け防止で確保） ===== */
.block-container{
  padding-top: 3.2rem;     /* タイトル欠けるなら 3.6rem に */
  padding-bottom: 0.5rem;
  padding-left: 1.0rem;
  padding-right: 1.0rem;
}

/* ===== 見出し・テキストの上下マージンを削る（サイズは変えない） ===== */
h1{ margin: 0.1rem 0 0.5rem 0; line-height: 1.12; }
h2{ margin: 0.45rem 0 0.2rem 0; }
h3{ margin: 0.35rem 0 0.15rem 0; }
p, .stCaption{ margin: 0.15rem 0; }

/* ===== カラム間のスキマを削る（横に詰める） ===== */
div[data-testid="stHorizontalBlock"]{
  gap: 0.6rem;             /* 0.4〜0.8で好み */
  align-items: center;
}

/* ===== ボタンの高さをスリム化（文字サイズはそのまま） ===== */
div.stButton > button{
  padding: 0.35rem 0.8rem; /* 高さを下げる */
}

/* ===== トグル・チェックボックス・ラジオの上下を詰める ===== */
div[data-testid="stToggle"], 
div[data-testid="stCheckbox"],
div[data-testid="stRadio"]{
  padding-top: 0.05rem;
  padding-bottom: 0.05rem;
  margin-top: 0.05rem;
  margin-bottom: 0.05rem;
}

/* ラジオ（横並び）の項目間も詰める */
div[role="radiogroup"]{ gap: 0.35rem; }

/* ===== スライダーの上下を詰める ===== */
div[data-testid="stSlider"]{
  padding-top: 0.05rem;
  padding-bottom: 0.05rem;
  margin-top: 0.05rem;
  margin-bottom: 0.05rem;
}

/* ===== metric の上下余白を詰める（表示サイズは維持） ===== */
div[data-testid="stMetric"]{
  padding: 0.05rem 0;
  margin: 0.05rem 0;
}
div[data-testid="stMetricLabel"]{ margin-bottom: 0.05rem; }
div[data-testid="stMetricValue"]{ margin-top: 0.05rem; }

/* ===== 画像の上下余白を削る ===== */
div[data-testid="stImage"]{
  margin-top: 0.08rem;
  margin-bottom: 0.08rem;
}

/* ===== 研究UI：カード・コントロールの視認性 ===== */
.neo-card{
  border:1px solid rgba(128,128,128,.22);
  border-radius:14px;
  padding:0.75rem 0.9rem;
  margin:0.35rem 0;
  background:rgba(127,127,127,.055);
}
.neo-soft{ opacity:.82; font-size:0.92rem; line-height:1.55; }
.neo-title{ font-weight:700; font-size:1.02rem; margin-bottom:.2rem; }
.neo-badge{
  display:inline-block; padding:.16rem .48rem; border-radius:999px;
  border:1px solid rgba(128,128,128,.28); background:rgba(127,127,127,.08);
  font-size:.82rem; margin-right:.25rem; margin-bottom:.2rem;
}
.neo-alert{ border-left:4px solid rgba(255,180,0,.85); padding-left:.7rem; }

</style>
""", unsafe_allow_html=True)

PHASES = ["① 発生", "② 認識", "③ 思考", "④ 行動", "⑤ 生死"]

# 哲学的個体＝行動方針を持つビークル、として自然淘汰にかけるための遺伝子。
# ここでいう「哲学」は思想家本人の内面や倫理を完全再現するものではなく、
# 自然淘汰にさらすための「行動評価関数」への操作的変換である。
# 0 ヒューム型：経験・局所観察・目の前の証拠を重視する。
# 1 ストア型：外部条件に過剰反応せず、危険下でも自己保存を安定させる。
# 2 デカルト型：疑わしいものを避け、明確な利得・安全性・自己保存を重視する。
# 3 カント型：短期利得だけでなく、規則性・非搾取・持続可能性を重視する。
NORMAL_PHILO_VALUE = 4
PHILO_TYPE_COUNT = 5
PHILOSOPHY_VALUES = (0, 1, 2, 3)

PHILO_LABELS = {
    0: "ヒューム型",
    1: "ストア型",
    2: "デカルト型",
    3: "カント型",
    4: "通常個体",
}

PHILO_THEORY = {
    0: "ヒューム型（Hume）：経験・観察・局所的な証拠を重く見る。資源や実際に見えている情報への反応が強い。",
    1: "ストア型（Epictetus/Marcus Aurelius）：外部環境の変動に振り回されず、危険回避・自己制御・生存安定を重視する。",
    2: "デカルト型（Descartes）：方法的懐疑の操作的モデル。曖昧な期待値より、明確な利得・安全性・自己保存を重視する。",
    3: "カント型（Kant）：規則性・普遍化可能性の操作的モデル。短期的な搾取より、非搾取・安定繁殖・持続可能な行動を重視する。",
    4: "通常個体：哲学的な補正を持たない中立個体。哲学型が有利なのか、単に中立行動と同等なのかを比較する対照群。",
}

def philo_active_values():
    """サイドバーでONになっている哲学型の値を返す。全OFFならヒューム型だけを使う。"""
    flags = {
        0: bool(globals().get('philo_enable_hume', True)),
        1: bool(globals().get('philo_enable_stoic', True)),
        2: bool(globals().get('philo_enable_descartes', True)),
        3: bool(globals().get('philo_enable_kant', True)),
    }
    vals = [k for k, v in flags.items() if v]
    return vals if vals else [0]

def philo_choice_values_probs():
    """初期生成・古い世界の補修用。ONの型だけから、重みに従って選ぶ。"""
    vals = philo_active_values()
    weight_map = {
        0: int(globals().get('philo_weight_hume', 25)),
        1: int(globals().get('philo_weight_stoic', 25)),
        2: int(globals().get('philo_weight_descartes', 25)),
        3: int(globals().get('philo_weight_kant', 25)),
    }
    weights = np.array([max(0, weight_map.get(v, 0)) for v in vals], dtype=np.float64)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(vals), dtype=np.float64)
    probs = weights / weights.sum()
    return np.array(vals, dtype=np.int8), probs


def philo_index(value):
    """統計配列用の安全な哲学/通常個体インデックス。"""
    try:
        v = int(value)
    except Exception:
        return int(NORMAL_PHILO_VALUE)
    if 0 <= v < int(PHILO_TYPE_COUNT):
        return v
    return int(NORMAL_PHILO_VALUE)


def make_initial_philo_array(rng, n):
    """通常個体と哲学個体を、初期割合に従って厳密に生成する。"""
    n = int(n)
    if n <= 0:
        return np.array([], dtype=np.int8)
    vals, probs = philo_choice_values_probs()
    arr = rng.choice(vals, size=n, replace=True, p=probs).astype(np.int8)
    normal_pct = float(globals().get('initial_normal_pct', 80))
    normal_n = int(round(n * np.clip(normal_pct, 0.0, 100.0) / 100.0))
    normal_n = max(0, min(n, normal_n))
    if normal_n > 0:
        normal_idx = rng.choice(n, size=normal_n, replace=False)
        arr[normal_idx] = int(NORMAL_PHILO_VALUE)
    return arr.astype(np.int8)

PHILO_STAT_KEYS = [
    "stat_philo_birth_reserved", "stat_philo_birth_real", "stat_philo_death",
    "stat_philo_gather_gain", "stat_philo_move_cost", "stat_philo_upkeep_cost",
    "stat_philo_mate_attempt", "stat_philo_mate_success",
    "stat_philo_predation_attempt", "stat_philo_predation_success", "stat_philo_predation_fail",
    "stat_philo_predation_gain", "stat_philo_battle_gain", "stat_philo_battle_cost",
    # v19：親としてどれだけ子の発生に参加したか。子として増えた数とは分けて見る。
    "stat_philo_parent_offspring_reserved", "stat_philo_parent_offspring_real",
]

# ③思考で選ばれた行動。
# 「なぜその遺伝子が増えたか」を見るには、出生・死亡だけでなく行動選択の偏りが必要。
PHILO_ACTION_LABELS = {
    0: "待機",
    1: "移動",
    2: "採取",
    3: "戦闘",
    4: "回避",
    5: "交尾",
    6: "捕食",
}

# v19：親子の遺伝子フロー用。
# pair は親組み合わせ、parent_to_child は親の型がどの子型を生んだか、source_to_child は実際に子へコピーされた型。
PHILO_MATRIX_STAT_KEYS = [
    "stat_philo_pair_reserved",
    "stat_philo_pair_real",
    "stat_philo_parent_to_child_reserved",
    "stat_philo_parent_to_child_real",
    "stat_philo_source_to_child_reserved",
    "stat_philo_source_to_child_real",
]

# -------------------------
# 研究本体：ネオライフ（空間）だけを残す
# -------------------------
st.caption("モード整理済み：タカハトESSの独立デモ2種は削除し、研究本体の空間生態系モデルだけを表示します。")

def get_biome_palette(k: int):
    if k == 2:
        items = [
            ("バイオーム0（白）", "#ffffff", (1.00, 1.00, 1.00)),
            ("バイオーム1（黄）", "#fad233", (0.98, 0.82, 0.20)),
        ]
    elif k == 3:
        items = [
            ("バイオーム0（白）", "#ffffff", (1.00, 1.00, 1.00)),
            ("バイオーム1（緑）", "#33d940", (0.20, 0.85, 0.25)),
            ("バイオーム2（黄）", "#fad233", (0.98, 0.82, 0.20)),
        ]
    else:
        items = [
            ("バイオーム0（白）", "#ffffff", (1.00, 1.00, 1.00)),
            ("バイオーム1（緑）", "#33d940", (0.20, 0.85, 0.25)),
            ("バイオーム2（青）", "#267fff", (0.15, 0.50, 1.00)),
            ("バイオーム3（黄）", "#fad233", (0.98, 0.82, 0.20)),
        ]

    colors = np.array([rgb for _, _, rgb in items], dtype=np.float32)
    labels = [lab for lab, _, _ in items]
    hexes  = [hx for _, hx, _ in items]
    return colors, labels, hexes

def swatch_line(hex_color: str, text: str):
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <span style="width:14px;height:14px;border-radius:3px;background:{hex_color};
                   display:inline-block;border:1px solid rgba(255,255,255,0.2);"></span>
      <span style="font-size:13px;opacity:0.95;">{text}</span>
    </div>
    """

# -------------------------
# サイドバー（実験パラメータ）
# -------------------------
with st.sidebar:
    st.subheader("実験パラメータ")

    seed = st.number_input("乱数シード（seed）", value=0, step=1)

    st.divider()
    st.subheader("盤面・表示")
    # 「総ピクセル数を半分くらい」→ 既定を小さめにして scale を上げる
    H = st.selectbox("高さ（セル）", [60, 80, 100, 120], index=1)   # 既定80
    W = st.selectbox("幅（セル）", [80, 100, 120, 160], index=2)    # 既定120
    scale = st.slider("セル拡大倍率（大きいほど見やすい）", 8, 60, 24, 1)
    grid_line = st.slider("格子線の明るさ（0=黒,255=白）", 0, 255, 70, 1)
    grid_thickness = st.slider("格子線の太さ（px）", 1, 3, 1, 1)

    st.divider()
    st.subheader("バイオーム")
    # 色数を減らして分かりやすく
    biome_k = st.selectbox("バイオーム数（少ないほど分かりやすい）", [2, 3, 4], index=1)
    show_biome_edges = st.checkbox("境界を強調する", value=False)

    st.divider()
    st.subheader("資源（最重要）")
    init_res_cover_pct = st.slider("初期：資源があるマス割合（%）", 0.0, 60.0, 20.0, 0.5)
    init_res_cover = init_res_cover_pct / 100.0
    init_res_amount = st.slider("初期：資源量（1マス）", 0, 20, 3, 1)
    res_max = st.slider("資源上限（1マス）", 1, 30, 10, 1)

    res_spawn_rate_pct = st.slider("自然発生率（%/世代）", 0.0, 10.0, 3.0, 0.1)
    res_spawn_rate = res_spawn_rate_pct / 100.0
    res_spawn_amount = st.slider("自然発生量（発生時）", 0, 10, 2, 1)

    resource_alpha = st.slider("資源の濃さ（強すぎると地形が潰れる）", 0.0, 1.0, 0.35, 0.05)

    st.divider()
    st.subheader("生態系モデル補正")
    enable_density_dependence = st.checkbox("密度依存を有効にする", value=True)
    enable_local_resource_regen = st.checkbox("資源再生の局所性を有効にする", value=True)
    enable_kin_avoidance = st.checkbox("近親交配回避を有効にする", value=True)
    density_radius = st.slider("密度を見る半径（マス）", 1, 5, 2, 1)
    density_birth_capacity = st.slider("出生が苦しくなり始める近傍個体数", 1, 30, 10, 1)
    density_birth_penalty = st.slider("過密による出生抑制", 0.0, 1.0, 0.45, 0.05)
    local_resource_bonus = st.slider("資源再生：近くの資源による増えやすさ", 0.0, 3.0, 1.2, 0.1)
    density_resource_penalty = st.slider("資源再生：過密による減りやすさ", 0.0, 1.0, 0.35, 0.05)
    kin_avoid_strength = st.slider("近親交配回避の強さ", 0.0, 1.0, 0.80, 0.05)
    kin_avoid_threshold = st.slider("近親とみなす血縁度", 0.0, 1.0, 0.50, 0.05)

    st.divider()
    st.subheader("捕食/被食")
    enable_predation = st.checkbox("捕食/被食を有効にする", value=True)
    predation_gene_init_pct = st.slider("初期：捕食傾向遺伝子（%）", 0, 100, 5, 1)
    predation_hunger_threshold = st.slider("捕食を考え始める所持資源しきい値", 0, 50, 3, 1)
    predation_gain_rate = st.slider("捕食成功時に奪う資源割合（%）", 0, 100, 60, 5) / 100.0
    predation_fail_cost = st.slider("捕食失敗コスト", 0, 20, 1, 1)

    st.divider()
    st.subheader("争奪（資源の取り合い / タカ・ハト）")
    enable_contest = st.checkbox("資源争奪（contest）を有効にする", value=True)

    hawk_init_pct = st.slider("初期：タカ比率（%）", 0, 100, 50, 1)
    hawk_init = hawk_init_pct / 100.0

    contest_mu_pct = st.slider("出生時：戦略遺伝子の突然変異率（%）", 0.0, 20.0, 1.0, 0.1)
    contest_mu = contest_mu_pct / 100.0

    contest_C_base = st.slider("コスト C（固定）", 0, 200, 10, 1)          # 整数
    contest_C_perV = st.slider("コスト C（Vあたり）", 0, 50, 1, 1)          # 整数

    st.divider()
    st.subheader("初期個体")
    n0 = st.slider("初期個体数（体）", 0, 3000, 250, 10)
    init_bag_min = st.slider("初期：所持資源（最小）", 0, 50, 8, 1)
    init_bag_max = st.slider("初期：所持資源（最大）", 0, 50, 18, 1)

    st.divider()
    st.subheader("個体パラメータ")
    upkeep_cost = st.slider("維持コスト（⑤）", 0, 20, 1, 1)         # 整数
    move_cost = st.slider("移動コスト（④）", 0, 20, 0, 1)           # 整数。維持コストと二重計上しやすいので既定0
    gather_amount = st.slider("採取量（④）", 0, 20, 3, 1)            # 整数。1だと維持+移動を上回れず絶滅しやすい
    move_radius = st.slider("移動半径（マス）", 1, 6, 2, 1)
    max_age = st.slider("寿命（世代）", 1, 300, 80, 1)

    st.divider()
    st.subheader("認識（個体ごと）")
    vision_min = st.slider("認識半径（最小）", 0, 12, 3, 1)
    vision_max = st.slider("認識半径（最大）", 0, 12, 5, 1)
    vision_mutate_pct = st.slider("出生時：認識半径の変化確率（%）", 0, 100, 20, 5)
    vision_mutate = vision_mutate_pct / 100.0

    st.divider()
    st.subheader("肉体強度（戦闘で使用）")
    str_min = st.slider("初期：肉体強度（最小）", 1, 50, 8, 1)
    str_max = st.slider("初期：肉体強度（最大）", 1, 50, 15, 1)
    str_mutate_pct = st.slider("出生時：肉体強度の変化確率（%）", 0, 100, 20, 5)
    str_mutate = str_mutate_pct / 100.0

    st.divider()
    st.subheader("出生（①）")
    enable_birth = st.checkbox("出生を有効にする", value=True)
    birth_fee = st.slider("出生コスト（親が必ず支払う）", 0, 50, 2, 1)  # 整数
    child_share_pct = st.slider("子へ分配する所持資源（親の%）", 0, 100, 25, 5)
    child_bag_cap = st.slider("子の所持資源 上限（安全弁）", 0, 100, 30, 1)

    birth_ready_bag = st.slider("繁殖を考える所持資源しきい値", 0, 50, 6, 1)
    birth_k = st.slider("出生確率の伸び方（K）", 1, 100, 12, 1)  # 閾値付近での増え方
    mate_search_bonus = st.slider("交尾相手探索の強さ", 0.0, 30.0, 13.0, 0.5)

    st.divider()
    st.subheader("戦闘（④）")
    enable_battle = st.checkbox("戦闘を有効にする", value=True)
    upset_pct = st.slider("弱い方が勝つ確率（%）", 0, 20, 5, 1)  # 指定：5%
    upset_p = upset_pct / 100.0

    st.divider()
    st.subheader("思考（③）点数化パラメータ")
    think_w_cell = st.slider("資源：足元の重み", 0.0, 10.0, 3.0, 0.1)
    think_w_nei  = st.slider("資源：近傍(3×3)の重み", 0.0, 5.0, 1.2, 0.1)
    think_w_biome = st.slider("バイオームの重み", 0.0, 5.0, 0.6, 0.1)

    think_w_danger = st.slider("危険（強い敵が近い）の重み", 0.0, 10.0, 3.0, 0.1)

    think_battle_bias = st.slider("戦闘の基本優先度（+で戦いが起きやすい）", -50.0, 50.0, 8.0, 0.5)
    think_risk = st.slider("戦闘リスク回避（大きいほど慎重）", 0.0, 5.0, 1.2, 0.1)

    explore_pct = st.slider("探索行動（見えてる資源が少ない時にランダム移動する確率%）", 0, 30, 5, 1)
    explore_p = explore_pct / 100.0

    st.divider()
    st.subheader("哲学遺伝子（行動評価関数）")
    enable_philo_gene = st.checkbox("哲学遺伝子を行動に反映する", value=True)
    philo_effect = st.slider("哲学遺伝子の影響強度", 0.0, 2.0, 1.0, 0.05)
    initial_normal_pct = st.slider("初期：通常個体割合（%）", 0, 100, 80, 1)
    st.caption("通常個体も遺伝子フロー表・比較実験に入ります。80%なら、哲学型は20%ぶんの中で初期重みに従って発生します。")
    st.caption("通常個体は哲学的な行動補正を持たない中立対照群です。初期値は80%です。")

    with st.expander("哲学型のON/OFF・初期重み", expanded=True):
        st.caption("OFFにした型は初期個体にも、古い個体群の補修にも使われません。全OFFなら安全のためヒューム型だけにします。")
        philo_enable_hume = st.checkbox("ヒューム型を使う", value=True)
        philo_weight_hume = st.slider("初期重み：ヒューム型", 0, 100, 25, 1)
        philo_enable_stoic = st.checkbox("ストア型を使う", value=True)
        philo_weight_stoic = st.slider("初期重み：ストア型", 0, 100, 25, 1)
        philo_enable_descartes = st.checkbox("デカルト型を使う", value=True)
        philo_weight_descartes = st.slider("初期重み：デカルト型", 0, 100, 25, 1)
        philo_enable_kant = st.checkbox("カント型を使う", value=True)
        philo_weight_kant = st.slider("初期重み：カント型", 0, 100, 25, 1)

    with st.expander("4型の操作的定義", expanded=False):
        for _k, _v in PHILO_THEORY.items():
            st.markdown(f"- **{PHILO_LABELS[_k]}**：{_v}")
        st.caption("思想家本人の完全再現ではなく、自然淘汰にさらすために、行動評価関数へ変換したモデルです。")

    st.divider()
    st.subheader("可視化")
    ui_preset = st.radio("表示プリセット", ["観察", "軽量", "詳細"], index=0, horizontal=True)
    st.caption("軽量は描画情報を減らします。モデル計算そのものは変えません。")

    _default_resource = True
    _default_agents = True
    _default_perception = (ui_preset == "詳細")
    _default_thinking = (ui_preset == "詳細")

    show_resource = st.checkbox("資源を表示", value=_default_resource)
    show_agents = st.checkbox("個体を表示", value=_default_agents)
    show_perception = st.checkbox("②認識ヒートを重ねる", value=_default_perception)
    show_thinking = st.checkbox("③/④の矢印・行動マークを描く", value=_default_thinking)
    show_quick_interpretation = st.checkbox("状況の自動読み取りメモを表示", value=True)

    st.divider()
    st.subheader("高速化")
    fast_thinking = st.checkbox("思考計算を高速化する", value=True)
    mate_search_radius_cap = st.slider("配偶者探索の最大半径（高速化用）", 1, 10, 4, 1)
    max_history_keep = st.slider("統計履歴の保存上限（世代）", 100, 1500, 300, 100)
    st.caption("高速化ONでは、行動ルールはほぼ同じまま、配偶者探索を全個体総当たりから近傍探索へ切り替えます。")

    st.divider()
    st.subheader("表示テンポ")
    smooth_auto_run = st.checkbox("自動実行は1世代ごとに描画する", value=True)
    auto_generations_per_refresh = st.slider("1回の再描画で進める世代数", 1, 20, 1, 1)
    auto_lock_environment_view = st.checkbox("自動実行中は環境ビューに固定する", value=True)
    st.caption("ONにすると、①〜⑤の各フェーズで毎回画面を描き直さず、1世代ぶん計算してから一気に表示します。暗転やカクつきがかなり減ります。")

    st.divider()
    with st.expander("凡例（色・矢印）", expanded=False):
        st.markdown("**矢印（紫）・マークの意味**")
        st.markdown("""
- **紫矢印**：位置の変化（③思考=「意図」 / ④行動=「確定」）
- **□** 待機 **○** 採取 **×** 戦闘 **△** 回避 **◇** 交尾  
  ※出生は「④で相互に交尾を選択＆隣接」→ 次世代①で発生
""")

        st.markdown("---")
        st.markdown("**ピクセル色（重なり）の意味**")

        _, biome_labels, biome_hexes = get_biome_palette(int(biome_k))

        html = "<div>"
        for lab, hx in zip(biome_labels, biome_hexes):
            html += swatch_line(hx, f"{lab} = 地形（バイオーム）")
        html += swatch_line("#00ffa6", "ミント = 資源（濃いほど多い / 上限=res_max）")
        html += swatch_line("#ff3333", "赤 = 個体（チーム0）")
        html += swatch_line("#338cff", "青 = 個体（チーム1）")
        html += swatch_line("#ffff00", "黄ヒート = 認識のヒート（②：見られた回数が多いほど濃い）")
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)
        st.caption("重なり順：バイオーム → 資源 → 個体 → 認識ヒート（ON時）")


# -------------------------
# 上部UI（進行は上に）
# -------------------------
top = st.container()
with top:
    st.markdown(
        """
        <div class="neo-card">
          <div class="neo-title">操作パネル</div>
          <div class="neo-soft">普段は <b>1世代</b> か <b>自動実行</b> を使うとテンポよく観察できます。
          <span class="neo-badge">①発生</span><span class="neo-badge">②認識</span><span class="neo-badge">③思考</span><span class="neo-badge">④行動</span><span class="neo-badge">⑤生死</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c, d, e, f, g = st.columns([1.0, 1.05, 1.0, 1.0, 0.9, 0.9, 2.0])
    reset_btn = a.button("↻ リセット", use_container_width=True)
    step_gen_btn = b.button("▶ 1世代", use_container_width=True)
    step_10_btn = c.button("▶▶ 10世代", use_container_width=True)
    step_50_btn = d.button("▶▶▶ 50世代", use_container_width=True)
    step_btn = e.button("1フェーズ", use_container_width=True)
    running = f.toggle("自動", value=False)
    speed_ms = g.slider("更新間隔（ms）", 0, 2000, 80, 10)  # 0なら最速

# -------------------------
# セッション初期化
# -------------------------
if "gen" not in st.session_state:
    st.session_state.gen = 0
if "phase" not in st.session_state:
    st.session_state.phase = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "world" not in st.session_state:
    st.session_state.world = None
if "last_phase_executed" not in st.session_state:
    st.session_state.last_phase_executed = None

# -------------------------
# バイオーム生成
# -------------------------
def make_biome(h, w, seed_i, n_seeds=18, blur_iters=14):
    rng = np.random.default_rng(seed_i)
    field = np.zeros((h, w), dtype=np.float32)
    ys = rng.integers(0, h, size=n_seeds)
    xs = rng.integers(0, w, size=n_seeds)
    vals = rng.random(n_seeds).astype(np.float32)
    for y, x, v in zip(ys, xs, vals):
        field[y, x] = v
    for _ in range(blur_iters):
        up = np.roll(field, -1, axis=0)
        dn = np.roll(field, 1, axis=0)
        lf = np.roll(field, -1, axis=1)
        rt = np.roll(field, 1, axis=1)
        field = (field * 2 + up + dn + lf + rt) / 6.0
    mn, mx = float(field.min()), float(field.max())
    if mx - mn > 1e-9:
        field = (field - mn) / (mx - mn)
    return field

def discretize_biome(biome01, k):
    bins = np.linspace(0, 1, k + 1)[1:-1]
    return np.digitize(biome01, bins=bins)  # 0..k-1

def biome_edges(biome_id):
    return (biome_id != np.roll(biome_id, 1, axis=0)) | (biome_id != np.roll(biome_id, 1, axis=1))

@lru_cache(maxsize=None)
def vision_offsets(r: int, d: int):
    """
    半径rの「前方半円」(dy,dx)を返す。
    d: 0=上, 1=右, 2=下, 3=左
    """
    r = int(r)
    d = int(d)

    if r <= 0:
        return []

    # 前方向ベクトル
    if d == 0:
        fy, fx = -1, 0
    elif d == 1:
        fy, fx = 0, 1
    elif d == 2:
        fy, fx = 1, 0
    else:
        fy, fx = 0, -1

    out = []
    rr = r * r
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # 円
            if dy * dy + dx * dx > rr:
                continue
            # 前方半円（内積>=0）
            if dy * fy + dx * fx >= 0:
                out.append((dy, dx))
    return out

# -------------------------
# 出生確率（80%〜99%）
# -------------------------
def birth_probability(bag_amount, K):
    """
    出生確率。
    旧版の「最低80%」は、資源が乏しくても高確率で子を作るため生物学的に不自然だった。
    ここでは2親の合計所持資源が birth_ready_bag×2 を超えるほど出生しやすくする。
    """
    bag_amount = np.maximum(bag_amount, 0).astype(np.float32)
    K = max(float(K), 1.0)
    threshold = float(birth_ready_bag) * 2.0
    z = (bag_amount - threshold) / K
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(0.05 + 0.90 * p, 0.05, 0.95)

def torus_density_map(ys, xs, h, w, radius=2):
    """各セル周辺（Moore近傍）の個体密度。中心セルも含む。"""
    occ = np.zeros((int(h), int(w)), dtype=np.int32)
    if len(xs) > 0:
        occ[ys.astype(np.int32), xs.astype(np.int32)] = 1
    r = int(max(0, radius))
    if r <= 0:
        return occ
    out = np.zeros_like(occ, dtype=np.int32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += np.roll(np.roll(occ, dy, axis=0), dx, axis=1)
    return out

def local_resource_map(resource, radius=1):
    """近くに資源があるほど再生しやすくするための局所資源量。"""
    r = int(max(0, radius))
    res = resource.astype(np.float32)
    if r <= 0:
        return res
    out = np.zeros_like(res, dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += np.roll(np.roll(res, dy, axis=0), dx, axis=1)
    return out

def torus_delta(a, b, size):
    """トーラス上で a から b へ向かう最短の符号付き差分。"""
    size = int(size)
    d = int(b) - int(a)
    half = size // 2
    if d > half:
        d -= size
    elif d < -half:
        d += size
    return int(d)


@lru_cache(maxsize=None)
def move_offsets(radius):
    """移動半径内の全候補。Chebyshev距離で半径以内ならどこへ止まってもよい。"""
    r = int(max(1, radius))
    return tuple((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if max(abs(dy), abs(dx)) <= r)


def ensure_ecology_arrays(w):
    """古いセッションでも新しい遺伝子・血縁配列が必ず存在するようにする。"""
    n = int(len(w.get("xs", [])))

    if "uid" not in w or len(w.get("uid", [])) != n:
        w["uid"] = np.arange(n, dtype=np.int32)
        w["next_uid"] = int(n)
    else:
        w["uid"] = w["uid"].astype(np.int32)
        w["next_uid"] = int(max(w.get("next_uid", n), int(w["uid"].max()) + 1 if n > 0 else 0))

    for key in ["parent_a", "parent_b"]:
        if key not in w or len(w.get(key, [])) != n:
            w[key] = np.full(n, -1, dtype=np.int32)
        else:
            w[key] = w[key].astype(np.int32)

    if "lineage" not in w or len(w.get("lineage", [])) != n:
        w["lineage"] = w["uid"].copy().astype(np.int32)
    else:
        w["lineage"] = w["lineage"].astype(np.int32)

    if "gene_predation" not in w or len(w.get("gene_predation", [])) != n:
        w["gene_predation"] = np.zeros(n, dtype=np.int8)
    else:
        w["gene_predation"] = w["gene_predation"].astype(np.int8)

    if "gene_philo" not in w or len(w.get("gene_philo", [])) != n:
        # 古い世界の場合は、通常個体割合を含めて補う。
        rng_fix = np.random.default_rng(12345 + n)
        w["gene_philo"] = make_initial_philo_array(rng_fix, n)
    else:
        arr = w["gene_philo"].astype(np.int8)
        valid_values = set(range(int(PHILO_TYPE_COUNT)))
        invalid = np.array([int(x) not in valid_values for x in arr], dtype=bool)
        if invalid.any():
            rng_fix = np.random.default_rng(54321 + n)
            arr[invalid] = make_initial_philo_array(rng_fix, int(invalid.sum()))
        w["gene_philo"] = arr

    if "prev_contest_counts" not in w:
        w["prev_contest_counts"] = np.bincount(w.get("gene_contest", np.array([], dtype=np.int8)).astype(np.int32), minlength=2).astype(np.int32)
    if "prev_predation_counts" not in w:
        w["prev_predation_counts"] = np.bincount(w.get("gene_predation", np.array([], dtype=np.int8)).astype(np.int32), minlength=2).astype(np.int32)
    if "prev_philo_counts" not in w:
        w["prev_philo_counts"] = np.bincount(w.get("gene_philo", np.array([], dtype=np.int8)).astype(np.int32), minlength=PHILO_TYPE_COUNT).astype(np.int32)
    if "prev_pop_count_for_W" not in w:
        w["prev_pop_count_for_W"] = int(n)

    # 追加統計キー。存在しなければ0で作る。
    for key in [
        "stat_local_resource_spawned", "stat_density_spawn_block",
        "stat_birth_density_block", "stat_kin_avoided",
        "stat_predation_attempt", "stat_predation_success", "stat_predation_fail",
        "stat_predation_gain", "stat_predation_kill",
    ]:
        if key not in w:
            w[key] = 0

    # 哲学遺伝子別のイベント統計。4型ぶんのベクトルで保持する。
    for key in PHILO_STAT_KEYS:
        if key not in w or np.asarray(w.get(key, [])).size != PHILO_TYPE_COUNT:
            w[key] = np.zeros(PHILO_TYPE_COUNT, dtype=np.int32)
        else:
            w[key] = np.asarray(w[key], dtype=np.int32)

    # v19：親子フロー・行動選択の世代内統計。
    for key in PHILO_MATRIX_STAT_KEYS:
        arr = np.asarray(w.get(key, []))
        if key not in w or arr.shape != (PHILO_TYPE_COUNT, PHILO_TYPE_COUNT):
            w[key] = np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)
        else:
            w[key] = arr.astype(np.int32)

    arr = np.asarray(w.get("stat_philo_action_counts", []))
    if "stat_philo_action_counts" not in w or arr.shape != (PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)):
        w["stat_philo_action_counts"] = np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32)
    else:
        w["stat_philo_action_counts"] = arr.astype(np.int32)

def kinship_score(w, a, b):
    """0=ほぼ無関係, 0.5=同系統, 0.75以上=親子・きょうだい相当の近縁。"""
    a = int(a); b = int(b)
    if a == b:
        return 1.0

    uid = w.get("uid")
    pa = w.get("parent_a")
    pb = w.get("parent_b")
    lineage = w.get("lineage")
    if uid is None or pa is None or pb is None or lineage is None:
        return 0.0

    ua = int(uid[a]); ub = int(uid[b])
    parents_a = {int(pa[a]), int(pb[a])}
    parents_b = {int(pa[b]), int(pb[b])}
    parents_a.discard(-1)
    parents_b.discard(-1)

    # 親子
    if ua in parents_b or ub in parents_a:
        return 0.75
    # きょうだい・半きょうだい
    if len(parents_a.intersection(parents_b)) > 0:
        return 0.75
    # 同じ創始系統。完全禁止ではなく「避けやすい」程度。
    if int(lineage[a]) == int(lineage[b]):
        return 0.50
    return 0.0

def predation_success_probability(str_i, str_j, bag_i, bag_j):
    """捕食成功確率。強さ差と空腹度に依存。極端な0/1を避ける。"""
    hunger_bonus = max(0.0, float(predation_hunger_threshold) - float(bag_i)) / max(float(predation_hunger_threshold), 1.0)
    x = (float(str_i) - float(str_j)) / 8.0 + 0.6 * hunger_bonus
    p = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(p, 0.05, 0.95))


def _safe_counts(arr, minlength):
    arr = np.asarray(arr, dtype=np.int32)
    if arr.size == 0:
        return np.zeros(int(minlength), dtype=np.int32)
    return np.bincount(arr, minlength=int(minlength)).astype(np.int32)


def _safe_W(now_counts, prev_counts):
    now_counts = np.asarray(now_counts, dtype=np.float64)
    prev_counts = np.asarray(prev_counts, dtype=np.float64)
    return now_counts / np.maximum(prev_counts, 1.0)


def _gene_diversity_from_counts(counts):
    """Simpson多様度 1-sum(p^2)。0なら単一遺伝子だけ、値が大きいほど多様。"""
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    return float(1.0 - np.sum(p * p))


def philo_action_modifiers(philo_value):
    """哲学的行動遺伝子が、同じ環境入力をどう評価するかを変える。

    戻り値: resource, danger, battle, mate, predation, escape
    中立値は (1, 1, 0, 0, 0, 0)。
    philo_effect により、中立値からどれだけ離すかを調整できる。
    """
    neutral = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    if not bool(globals().get('enable_philo_gene', True)):
        return neutral

    p = philo_index(philo_value)
    if p == int(NORMAL_PHILO_VALUE):
        return neutral

    if p == 0:
        # ヒューム型：経験・局所観察を重視。見えている資源には敏感だが、抽象的リスクで過剰に行動を止めない。
        raw = (1.35, 0.95, -1.0, 0.0, -1.0, -0.2)
    elif p == 1:
        # ストア型：外部要因への過剰反応を抑え、生存安定と危険回避を重視する。
        raw = (0.95, 1.45, -4.0, -0.5, -4.0, 3.5)
    elif p == 2:
        # デカルト型：疑わしい選択を抑え、明確な利得・安全性・自己保存を重視する。
        raw = (1.05, 1.30, -2.0, -0.5, -2.5, 2.0)
    elif p == 3:
        # カント型：短期的搾取より、規則性・非搾取・持続可能な繁殖を重視する。
        raw = (1.00, 1.15, -5.0, 3.0, -5.0, 1.0)
    else:
        return neutral

    e = float(globals().get('philo_effect', 1.0))
    return tuple(float(neutral[i]) + e * (float(raw[i]) - float(neutral[i])) for i in range(6))

# -------------------------
# 世界初期化
# -------------------------
def reset_world():
    rng = np.random.default_rng(int(seed))

    b01 = make_biome(H, W, int(seed))
    biome_id = discretize_biome(b01, int(biome_k))

    # 資源：初期（バイオームで偏りを持たせる）
    resource = np.zeros((H, W), dtype=np.int32)
    k = int(biome_k)
    biome_factor = np.linspace(0.6, 1.6, k).astype(np.float32)  # 不毛→豊穣
    prob_map = init_res_cover * biome_factor[biome_id]
    spawn = rng.random((H, W)) < np.clip(prob_map, 0.0, 1.0)
    resource[spawn] = np.minimum(int(init_res_amount), int(res_max))

    # 個体
    n = min(int(n0), H * W)
    idx = rng.choice(H * W, size=n, replace=False)
    ys = (idx // W).astype(np.int32)
    xs = (idx % W).astype(np.int32)

    team = rng.integers(0, 2, size=n).astype(np.int8)  # 0赤 / 1青

    # 争奪戦略遺伝子（0=タカ, 1=ハト）
    gene_contest = (rng.random(n) < float(hawk_init)).astype(np.int8)
    gene_contest = np.where(gene_contest == 1, 0, 1).astype(np.int8)

    # 捕食傾向遺伝子（0=通常採食寄り, 1=捕食も選びやすい）
    gene_predation = (rng.random(n) < (float(predation_gene_init_pct) / 100.0)).astype(np.int8)

    # 哲学的行動遺伝子 + 通常個体：通常個体は中立行動の対照群として一定割合だけ初期生成する。
    gene_philo = make_initial_philo_array(rng, n)

    lo, hi = int(init_bag_min), int(init_bag_max)
    if hi < lo:
        hi = lo
    bag = rng.integers(lo, hi + 1, size=n).astype(np.int32)

    vmin, vmax = int(vision_min), int(vision_max)
    if vmax < vmin:
        vmax = vmin
    vision = rng.integers(vmin, vmax + 1, size=n).astype(np.int8)

    smin, smax = int(str_min), int(str_max)
    if smax < smin:
        smax = smin
    strength = rng.integers(smin, smax + 1, size=n).astype(np.int32)

    age = np.zeros(n, dtype=np.int32)

    # 向き（0上/1右/2下/3左）…生誕時にランダム
    direction = rng.integers(0, 4, size=n).astype(np.int8)
    sex = rng.integers(0, 2, size=n).astype(np.int8)  # 0/1（性別）

    # 血縁・個体ID。初期個体は全員別創始系統。
    uid = np.arange(n, dtype=np.int32)
    parent_a = np.full(n, -1, dtype=np.int32)
    parent_b = np.full(n, -1, dtype=np.int32)
    lineage = uid.copy().astype(np.int32)

    st.session_state.world = {
        "biome_id": biome_id,
        "resource": resource,
        "ys": ys, "xs": xs,
        "team": team,
        "bag": bag,
        "vision": vision,
        "strength": strength,
        "age": age,
        "dir": direction,
        "sex": sex,
        "gene_contest": gene_contest,
        "gene_predation": gene_predation,
        "gene_philo": gene_philo,
        "uid": uid,
        "next_uid": int(n),
        "parent_a": parent_a,
        "parent_b": parent_b,
        "lineage": lineage,

        # 遺伝子頻度の世代間変化・適応度Wを計算するための前世代コピー数
        "prev_contest_counts": np.bincount(gene_contest.astype(np.int32), minlength=2).astype(np.int32),
        "prev_predation_counts": np.bincount(gene_predation.astype(np.int32), minlength=2).astype(np.int32),
        "prev_philo_counts": np.bincount(gene_philo.astype(np.int32), minlength=PHILO_TYPE_COUNT).astype(np.int32),
        "prev_pop_count_for_W": int(n),

        "pending_births": [],
        "stat_contest_cells": 0,
        "stat_contest_events": 0,
        "stat_contest_hawk_win": 0,
        "stat_contest_hh_events": 0,
        "stat_contest_cost_paid": 0,
        "stat_contest_v_paid": 0,

        # 可視化用
        "perception_count": np.zeros((H, W), dtype=np.int32),
        "intent_y": ys.copy(),
        "intent_x": xs.copy(),
        "intent_act": np.zeros(n, dtype=np.int8),  # 0待機 1移動 2採取 3戦闘 4回避 5交尾 6捕食
        "last_prev": (ys.copy(), xs.copy()),
        "last_act": np.zeros(n, dtype=np.int8),

        # イベント
        "evt_birth": [],
        "evt_res_spawn": np.zeros((H, W), dtype=bool),
        "evt_battle_cells": [],
        "evt_death": [],
        "evt_mate_pairs": [],
        "evt_predation_cells": [],

        # 既存カウンタ
        "births": 0,
        "battles": 0,
        "deaths": 0,

        # ---- 統計用カウンタ（世代内で加算して⑤で確定）----
        "stat_gathered": 0,
        "stat_move_intent": 0,
        "stat_move_actual": 0,
        "stat_move_dist_sum": 0.0,
        "stat_collision": 0,
        "stat_mate_attempt": 0,
        "stat_mate_success": 0,
        "stat_battle_transfer": 0,
        "stat_battle_upset": 0,
        "stat_res_spawn_cells": 0,
        "stat_res_spawned": 0,
        "stat_move_cost_paid": 0,
        "stat_upkeep_paid": 0,
        "stat_birth_fee_paid": 0,
        "stat_child_resource_given": 0,
        "stat_birth_reserved_hawk": 0,
        "stat_birth_reserved_dove": 0,
        "stat_birth_real_hawk": 0,
        "stat_birth_real_dove": 0,
        "stat_death_hawk": 0,
        "stat_death_dove": 0,
        "stat_pop_hawk_before_death": 0,
        "stat_pop_dove_before_death": 0,
        "stat_gain_gather_hawk": 0,
        "stat_gain_gather_dove": 0,
        "stat_gain_contest_hawk": 0,
        "stat_gain_contest_dove": 0,
        "stat_cost_contest_hawk": 0,
        "stat_cost_contest_dove": 0,
        "stat_gain_battle_hawk": 0,
        "stat_gain_battle_dove": 0,
        "stat_cost_battle_hawk": 0,
        "stat_cost_battle_dove": 0,
        "stat_cost_move_hawk": 0,
        "stat_cost_move_dove": 0,
        "stat_cost_upkeep_hawk": 0,
        "stat_cost_upkeep_dove": 0,
        "stat_cost_birthfee_hawk": 0,
        "stat_cost_birthfee_dove": 0,
        "stat_cost_childshare_hawk": 0,
        "stat_cost_childshare_dove": 0,
        "stat_birth_failed_space": 0,

        # ---- 生態系モデル補正 ----
        "stat_local_resource_spawned": 0,
        "stat_density_spawn_block": 0,
        "stat_birth_density_block": 0,
        "stat_kin_avoided": 0,
        "stat_predation_attempt": 0,
        "stat_predation_success": 0,
        "stat_predation_fail": 0,
        "stat_predation_gain": 0,
        "stat_predation_kill": 0,
        **{key: np.zeros(PHILO_TYPE_COUNT, dtype=np.int32) for key in PHILO_STAT_KEYS},
        **{key: np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32) for key in PHILO_MATRIX_STAT_KEYS},
        "stat_philo_action_counts": np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32),
    }

    st.session_state.gen = 0
    st.session_state.phase = 0
    st.session_state.history = []
    st.session_state.last_phase_executed = None
       
def signature():
    return (
        int(seed), H, W, int(n0),
        int(biome_k),
        float(init_res_cover), int(init_res_amount), int(res_max),
        float(res_spawn_rate), int(res_spawn_amount),
        int(init_bag_min), int(init_bag_max),
        int(vision_min), int(vision_max),
        int(str_min), int(str_max),
        int(move_radius), int(birth_ready_bag), float(mate_search_bonus),
        bool(enable_density_dependence), bool(enable_local_resource_regen), bool(enable_kin_avoidance),
        int(density_radius), int(density_birth_capacity), float(density_birth_penalty),
        float(local_resource_bonus), float(density_resource_penalty),
        float(kin_avoid_strength), float(kin_avoid_threshold),
        bool(enable_philo_gene), float(philo_effect), int(initial_normal_pct),
        bool(philo_enable_hume), bool(philo_enable_stoic), bool(philo_enable_descartes), bool(philo_enable_kant),
        int(philo_weight_hume), int(philo_weight_stoic), int(philo_weight_descartes), int(philo_weight_kant),
        bool(enable_predation), int(predation_gene_init_pct),
        int(predation_hunger_threshold), float(predation_gain_rate), int(predation_fail_cost),
    )

if "sig" not in st.session_state:
    st.session_state.sig = signature()
    reset_world()
else:
    if st.session_state.sig != signature():
        st.session_state.sig = signature()
        reset_world()

if reset_btn:
    reset_world()
    st.rerun()

# -------------------------
# フェーズ① 発生（出生・資源発生）
# -------------------------
def phase1_spawn_and_birth():
    w = st.session_state.world
    ensure_ecology_arrays(w)
    rng = np.random.default_rng(int(seed) + st.session_state.gen * 1009 + 1)

    # 表示用イベント初期化
    w["evt_birth"] = []
    w["births"] = 0

    # ----------------
    # 資源自然発生：局所再生 + バイオーム + 過密抑制
    # ----------------
    biome_id = w["biome_id"]
    resource = w["resource"]
    k = int(biome_k)
    biome_factor = np.linspace(0.6, 1.6, k).astype(np.float32)

    local_res = local_resource_map(resource, radius=1)
    local_res01 = np.clip(local_res / max(float(res_max) * 9.0, 1.0), 0.0, 1.0)

    density = torus_density_map(w["ys"], w["xs"], H, W, radius=int(density_radius)).astype(np.float32)
    density01 = np.clip(density / max(float(density_birth_capacity), 1.0), 0.0, 2.0)

    # 何もない場所にも基礎一次生産を残し、資源が近い場所では局所再生しやすくする。
    # ON/OFFで、局所性・密度依存を切り離して検証できる。
    resource_deficit = np.clip((float(res_max) - resource.astype(np.float32)) / max(float(res_max), 1.0), 0.0, 1.0)
    if bool(enable_local_resource_regen):
        local_bonus = 0.55 + float(local_resource_bonus) * local_res01
    else:
        local_bonus = np.ones_like(resource_deficit, dtype=np.float32)

    if bool(enable_density_dependence):
        density_factor = np.clip(1.0 - float(density_resource_penalty) * np.maximum(0.0, density01 - 0.5), 0.08, 1.0)
    else:
        density_factor = np.ones_like(resource_deficit, dtype=np.float32)

    prob_map = float(res_spawn_rate) * biome_factor[biome_id] * local_bonus * density_factor * resource_deficit
    prob_map = np.clip(prob_map, 0.0, 1.0)
    spawn = (rng.random((H, W)) < prob_map) & (resource < int(res_max))
    w["evt_res_spawn"] = spawn.copy()

    w["stat_res_spawn_cells"] += int(spawn.sum())
    # 過密で「発生しなかった可能性」を粗く記録
    w["stat_density_spawn_block"] += int(((density_factor < 0.99) & (resource < int(res_max))).sum())

    add = int(res_spawn_amount)
    if add > 0:
        cap = int(res_max)
        can_add = cap - resource[spawn]
        can_add = np.clip(can_add, 0, None)
        actual_add = int(np.minimum(can_add, add).sum())
        resource[spawn] = np.minimum(resource[spawn] + add, cap)
        w["stat_res_spawned"] += actual_add
        w["stat_local_resource_spawned"] += actual_add

    w["resource"] = resource

    # ----------------
    # 出生（前世代④で成立した「予約」だけを出す）
    # 密度依存：過密な場所ほど出生しにくい。ただし完全ゼロにはしない。
    # ----------------
    if not enable_birth:
        w["pending_births"] = []
        return

    pending = w.get("pending_births", [])
    w["pending_births"] = []

    if not pending:
        return

    ys = w["ys"]; xs = w["xs"]
    team = w["team"]; bag = w["bag"]
    vision = w["vision"]; strength = w["strength"]
    age = w["age"]; direction = w["dir"]; sex = w["sex"]
    gene = w["gene_contest"]
    gene_pred = w["gene_predation"]
    gene_philo = w["gene_philo"]
    uid = w["uid"]; parent_a = w["parent_a"]; parent_b = w["parent_b"]; lineage = w["lineage"]

    n = len(xs)
    occ = np.full(H * W, -1, dtype=np.int32)
    if n > 0:
        occ[ys * W + xs] = np.arange(n, dtype=np.int32)

    local_density = torus_density_map(ys, xs, H, W, radius=int(density_radius))
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    new_y, new_x = [], []
    new_team, new_bag = [], []
    new_vis, new_str = [], []
    new_age, new_dir = [], []
    new_sex, new_gene, new_gene_pred, new_gene_philo = [], [], [], []
    new_uid, new_pa, new_pb, new_lineage = [], [], [], []

    for child in pending:
        py = int(child.get("py", 0)); px = int(child.get("px", 0))
        rng.shuffle(offsets)
        placed = False

        for dy, dx in offsets:
            ny = (py + dy) % H
            nx = (px + dx) % W
            idx = ny * W + nx
            if occ[idx] != -1:
                continue

            if bool(enable_density_dependence):
                den = int(local_density[ny, nx])
                over = max(0, den - int(density_birth_capacity))
                p_place = 1.0 - float(density_birth_penalty) * (over / max(float(density_birth_capacity), 1.0))
                p_place = float(np.clip(p_place, 0.10, 1.0))
                if rng.random() > p_place:
                    w["stat_birth_density_block"] += 1
                    continue

            child_uid = int(w.get("next_uid", 0))
            w["next_uid"] = child_uid + 1

            new_y.append(ny); new_x.append(nx)
            new_team.append(int(child.get("team", 0)))
            new_bag.append(int(child.get("bag", 0)))
            new_vis.append(int(child.get("vision", 1)))
            new_str.append(int(child.get("strength", 1)))
            new_age.append(0)
            new_dir.append(int(child.get("dir", 0)))
            new_sex.append(int(child.get("sex", 0)))
            new_gene.append(int(child.get("gene", 1)))
            new_gene_pred.append(int(child.get("gene_predation", 0)))
            new_gene_philo.append(philo_index(child.get("gene_philo", NORMAL_PHILO_VALUE)))
            new_uid.append(child_uid)
            new_pa.append(int(child.get("parent_a", -1)))
            new_pb.append(int(child.get("parent_b", -1)))
            new_lineage.append(int(child.get("lineage", child_uid)))

            occ[idx] = 999999
            local_density[ny, nx] += 1
            w["evt_birth"].append((ny, nx))
            w["births"] += 1
            child_philo = philo_index(child.get("gene_philo", NORMAL_PHILO_VALUE))
            w["stat_philo_birth_real"][child_philo] += 1

            # v19：予約だけでなく、実際に空きマスへ置かれた出生の親子フローも記録する。
            ph_a_child = philo_index(child.get("parent_philo_a", NORMAL_PHILO_VALUE))
            ph_b_child = philo_index(child.get("parent_philo_b", NORMAL_PHILO_VALUE))
            ph_source_child = philo_index(child.get("source_philo", child_philo))
            w["stat_philo_parent_offspring_real"][ph_a_child] += 1
            w["stat_philo_parent_offspring_real"][ph_b_child] += 1
            pair_lo, pair_hi = sorted((int(ph_a_child), int(ph_b_child)))
            w["stat_philo_pair_real"][pair_lo, pair_hi] += 1
            w["stat_philo_parent_to_child_real"][ph_a_child, child_philo] += 1
            w["stat_philo_parent_to_child_real"][ph_b_child, child_philo] += 1
            w["stat_philo_source_to_child_real"][ph_source_child, child_philo] += 1

            if int(child.get("gene", 1)) == 0:
                w["stat_birth_real_hawk"] += 1
            else:
                w["stat_birth_real_dove"] += 1

            placed = True
            break

        if not placed:
            w["stat_birth_failed_space"] += 1
            continue

    if new_y:
        w["ys"] = np.concatenate([ys, np.array(new_y, dtype=np.int32)])
        w["xs"] = np.concatenate([xs, np.array(new_x, dtype=np.int32)])
        w["team"] = np.concatenate([team, np.array(new_team, dtype=np.int8)])
        w["bag"] = np.concatenate([bag, np.array(new_bag, dtype=np.int32)])
        w["vision"] = np.concatenate([vision, np.array(new_vis, dtype=np.int8)])
        w["strength"] = np.concatenate([strength, np.array(new_str, dtype=np.int32)])
        w["age"] = np.concatenate([age, np.array(new_age, dtype=np.int32)])
        w["dir"] = np.concatenate([direction, np.array(new_dir, dtype=np.int8)])
        w["sex"] = np.concatenate([sex, np.array(new_sex, dtype=np.int8)])
        w["gene_contest"] = np.concatenate([gene, np.array(new_gene, dtype=np.int8)])
        w["gene_predation"] = np.concatenate([gene_pred, np.array(new_gene_pred, dtype=np.int8)])
        w["gene_philo"] = np.concatenate([w["gene_philo"], np.array(new_gene_philo, dtype=np.int8)])
        w["uid"] = np.concatenate([uid, np.array(new_uid, dtype=np.int32)])
        w["parent_a"] = np.concatenate([parent_a, np.array(new_pa, dtype=np.int32)])
        w["parent_b"] = np.concatenate([parent_b, np.array(new_pb, dtype=np.int32)])
        w["lineage"] = np.concatenate([lineage, np.array(new_lineage, dtype=np.int32)])

# -------------------------
# フェーズ② 認識（個体ごと半径）
# -------------------------
def phase2_perception():
    w = st.session_state.world
    ys = w["ys"]
    xs = w["xs"]
    vision = w["vision"]
    direction = w["dir"]
    sex = w["sex"]

    n = len(xs)
    count = np.zeros((H, W), dtype=np.int32)
    if n == 0:
        w["perception_count"] = count
        return

    # 個体ごと：半径が違う + 向きが違う + 前方半円
    for i in range(n):
        r = int(vision[i])
        if r <= 0:
            continue
        d = int(direction[i])
        y0, x0 = int(ys[i]), int(xs[i])

        for dy, dx in vision_offsets(r, d):
            yy = (y0 + dy) % H
            xx = (x0 + dx) % W
            count[yy, xx] += 1

    w["perception_count"] = count

# -------------------------
# フェーズ③ 思考（紫矢印＋マーク）
# 0=待機 □ / 1=移動 → / 2=採取 ○ / 3=戦闘 × / 4=回避 △ / 5=交尾 ◇
# -------------------------
def phase3_thinking():
    w = st.session_state.world
    ensure_ecology_arrays(w)
    ys, xs = w["ys"], w["xs"]
    team, bag, strength, vision = w["team"], w["bag"], w["strength"], w["vision"]
    sex = w["sex"]
    direction = w["dir"]
    gene_predation = w.get("gene_predation", np.zeros(len(xs), dtype=np.int8))
    gene_philo = w.get("gene_philo", np.full(len(xs), NORMAL_PHILO_VALUE, dtype=np.int8))
    resource = w["resource"]
    biome_id = w["biome_id"]

    n = len(xs)
    if n == 0:
        return

    # 位置→個体index（1セル1個体前提）。配偶者探索・危険判定・捕食判定で使う。
    pos_to_idx = {(int(ys[i]), int(xs[i])): int(i) for i in range(n)}

    k = int(biome_k)
    biome_bonus = np.linspace(0.0, 1.0, k).astype(np.float32)

    moves = move_offsets(int(move_radius))
    neigh1 = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

    intent_y = ys.copy()
    intent_x = xs.copy()
    intent_act = np.zeros(n, dtype=np.int8)  # 0待機 / 1移動 / 2採取 / 3戦闘 / 4回避 / 5交尾 / 6捕食

    rng_think = np.random.default_rng(int(seed) + int(st.session_state.gen) * 1009 + 3)

    # 高速化：3×3近傍資源は個体ごと・候補ごとに毎回sumせず、世代ごとに1回だけ作る。
    # 旧処理とほぼ同じ意味で、「候補セル周辺にどれだけ資源があるか」を返す。
    try:
        use_fast = bool(fast_thinking)
    except NameError:
        use_fast = True

    if use_fast:
        resource_nei3 = local_resource_map(resource, radius=1).astype(np.float32)
    else:
        resource_nei3 = None

    for i in range(n):
        y0, x0 = int(ys[i]), int(xs[i])
        r = int(vision[i])
        my_team = int(team[i])
        my_bag = int(bag[i])
        my_str = int(strength[i])
        my_philo = int(gene_philo[i])
        res_mult, danger_mult, battle_bonus, mate_bonus, pred_bonus, escape_bonus = philo_action_modifiers(my_philo)
        hunger01 = max(0.0, float(birth_ready_bag) - float(my_bag)) / max(float(birth_ready_bag), 1.0)

        # 見えている範囲に資源がないときだけ探索を許す。
        # 高速化ONでは、正方形視界の資源量を軽いループで数える。
        visible_total_res = 0
        if r > 0:
            for vy in range(-r, r + 1):
                yy = (y0 + vy) % H
                for vx in range(-r, r + 1):
                    visible_total_res += int(resource[yy, (x0 + vx) % W])
        else:
            visible_total_res = int(resource[y0, x0])

        do_explore = (visible_total_res <= 0) and (rng_think.random() < float(explore_p))

        best_u = -1e18
        best_y, best_x, best_a = y0, x0, 0

        best_escape_u = -1e18
        best_ey, best_ex = y0, x0

        for dy, dx in moves:
            ty = (y0 + int(dy)) % H
            tx = (x0 + int(dx)) % W
            moved = (dy != 0 or dx != 0)
            cost_move = int(move_cost) if moved else 0

            # 候補セルが視界内なら資源・バイオーム取得（それ以外は未知=0）
            if r >= max(abs(int(dy)), abs(int(dx))):
                cell_res = int(resource[ty, tx])
                cell_bio = float(biome_bonus[int(biome_id[ty, tx])])
                if use_fast:
                    nei_sum = int(resource_nei3[ty, tx])
                else:
                    # 精密モード：旧版と同じく、視界パッチ内だけの近傍合計。
                    ys_r = (y0 + np.arange(-r, r + 1)) % H
                    xs_r = (x0 + np.arange(-r, r + 1)) % W
                    patch_res = resource[np.ix_(ys_r, xs_r)]
                    py = r + int(dy)
                    px = r + int(dx)
                    y1 = max(0, py - 1); y2 = min(2 * r, py + 1)
                    x1 = max(0, px - 1); x2 = min(2 * r, px + 1)
                    nei_sum = int(patch_res[y1:y2+1, x1:x2+1].sum())
            else:
                cell_res = 0
                cell_bio = 0.0
                nei_sum = 0

            # 危険度：候補セルの周囲1マスに敵がいる（強いほど危険）
            danger = 0
            local_targets = []
            for oy, ox in neigh1:
                yy = (ty + oy) % H
                xx = (tx + ox) % W
                j = pos_to_idx.get((yy, xx))
                if j is None or j == i:
                    continue
                if int(team[j]) == my_team:
                    continue
                local_targets.append(j)
                opp_str = int(strength[j])
                danger += max(1, opp_str - my_str)

            danger_pen = float(think_w_danger) * float(danger)

            # 資源スコア（資源最重要）
            res_score = (
                float(think_w_cell) * float(cell_res)
                + float(think_w_nei) * float(nei_sum)
                + float(think_w_biome) * float(cell_bio)
            )

            # 空腹時は資源探索を強める。哲学遺伝子は「同じ見え方」の評価重みを変える。
            res_score = res_score * float(res_mult) * (1.0 + 0.70 * float(hunger01))
            danger_pen = danger_pen * float(danger_mult)

            # ① 待機
            u_wait = -danger_pen
            if (not moved) and (u_wait > best_u):
                best_u = u_wait
                best_y, best_x, best_a = ty, tx, 0

            # ② 移動
            u_move = res_score - float(cost_move) - danger_pen
            if moved and (u_move > best_u):
                best_u = u_move
                best_y, best_x, best_a = ty, tx, 1

            # ③ 採取
            if int(gather_amount) > 0 and cell_res > 0:
                gain = int(min(int(gather_amount), cell_res))
                u_gather = res_score + float(gain) * (1.0 + 1.20 * float(hunger01)) - float(cost_move) - danger_pen
                if u_gather > best_u:
                    best_u = u_gather
                    best_y, best_x, best_a = ty, tx, 2

            # ④ 戦闘（期待値 − リスク）
            if enable_battle and local_targets:
                exp_total = 0.0
                var_total = 0.0
                lose_pay = float(my_bag // 2)

                for j in local_targets:
                    opp_bag = int(bag[j])
                    opp_str = int(strength[j])
                    win_get = float(opp_bag // 2)

                    if my_str > opp_str:
                        p_win = 1.0 - float(upset_p)
                    elif my_str < opp_str:
                        p_win = float(upset_p)
                    else:
                        p_win = 0.5

                    exp = p_win * win_get + (1.0 - p_win) * (-lose_pay)
                    exp_total += exp

                    a = win_get
                    b = -lose_pay
                    var = p_win * (a - exp) ** 2 + (1.0 - p_win) * (b - exp) ** 2
                    var_total += var

                u_battle = (
                    float(think_battle_bias)
                    + float(battle_bonus)
                    + exp_total
                    - float(cost_move)
                    - danger_pen
                    - float(think_risk) * var_total
                )

                if u_battle > best_u:
                    best_u = u_battle
                    best_y, best_x, best_a = ty, tx, 3

            # ⑤ 回避（逃げ）
            u_escape = (res_score * 0.3) + float(escape_bonus) - float(cost_move) - danger_pen
            if u_escape > best_escape_u:
                best_escape_u = u_escape
                best_ey, best_ex = ty, tx

        # ⑥ 交尾・配偶者探索
        # 高速化の本丸：旧版は全個体総当たりだったが、ここでは視界内のセルだけを見る。
        # 1セル1個体なので、近傍セル走査で候補を十分に拾える。
        if enable_birth and my_bag >= int(birth_ready_bag):
            best_mate_u = -1e18
            best_mate_y, best_mate_x, best_mate_a = y0, x0, 0
            try:
                search_r = min(max(1, r), int(mate_search_radius_cap))
            except NameError:
                search_r = min(max(1, r), 5)

            for dy_m0, dx_m0 in move_offsets(search_r):
                if dy_m0 == 0 and dx_m0 == 0:
                    continue
                yy = (y0 + int(dy_m0)) % H
                xx = (x0 + int(dx_m0)) % W
                j = pos_to_idx.get((yy, xx))
                if j is None or j == i:
                    continue
                if int(team[j]) != my_team:
                    continue
                if int(sex[j]) == int(sex[i]):
                    continue
                if int(bag[j]) < int(birth_ready_bag):
                    continue

                dist = max(abs(int(dy_m0)), abs(int(dx_m0)))
                if dist <= 0:
                    continue

                rel = kinship_score(w, i, j) if bool(enable_kin_avoidance) else 0.0
                kin_penalty = 12.0 * rel
                mate_quality = min(float(my_bag), float(bag[j])) / max(float(birth_ready_bag), 1.0)

                if dist <= 1:
                    u_mate = float(mate_search_bonus) + float(mate_bonus) + 4.0 * mate_quality - kin_penalty
                    if u_mate > best_mate_u:
                        best_mate_u = u_mate
                        best_mate_y, best_mate_x, best_mate_a = y0, x0, 5
                else:
                    step_y = int(np.clip(int(dy_m0), -int(move_radius), int(move_radius)))
                    step_x = int(np.clip(int(dx_m0), -int(move_radius), int(move_radius)))
                    ty_m = (y0 + step_y) % H
                    tx_m = (x0 + step_x) % W
                    u_seek = float(mate_search_bonus) + float(mate_bonus) + 2.0 * mate_quality - 0.7 * float(dist) - kin_penalty - float(move_cost)
                    if u_seek > best_mate_u:
                        best_mate_u = u_seek
                        best_mate_y, best_mate_x, best_mate_a = int(ty_m), int(tx_m), 1

            if best_mate_u > best_u and my_bag >= int(birth_ready_bag):
                best_u = best_mate_u
                best_y, best_x, best_a = best_mate_y, best_mate_x, best_mate_a

        # ⑦ 捕食/被食（空腹または捕食傾向遺伝子を持つ個体だけが強めに検討）
        if enable_predation:
            wants_predation = (my_bag <= int(predation_hunger_threshold)) or (int(gene_predation[i]) == 1 and my_bag <= int(birth_ready_bag))
            if wants_predation:
                best_prey = None
                best_pred_u = -1e18
                for oy, ox in neigh1:
                    yy = (y0 + oy) % H
                    xx = (x0 + ox) % W
                    j = pos_to_idx.get((yy, xx))
                    if j is None or j == i:
                        continue
                    if int(team[j]) == my_team:
                        continue
                    if bool(enable_kin_avoidance) and kinship_score(w, i, j) >= float(kin_avoid_threshold):
                        continue
                    p_win = predation_success_probability(my_str, int(strength[j]), my_bag, int(bag[j]))
                    gain_est = max(1.0, float(bag[j]) * float(predation_gain_rate))
                    u_pred = 2.0 + float(pred_bonus) + p_win * gain_est - (1.0 - p_win) * float(predation_fail_cost) - danger_pen
                    if u_pred > best_pred_u:
                        best_pred_u = u_pred
                        best_prey = (yy, xx)
                if best_prey is not None and best_pred_u > best_u:
                    best_u = best_pred_u
                    best_y, best_x, best_a = int(best_prey[0]), int(best_prey[1]), 6

        # 探索
        if do_explore:
            dy, dx = moves[int(rng_think.integers(0, len(moves)))]
            intent_y[i] = (y0 + int(dy)) % H
            intent_x[i] = (x0 + int(dx)) % W
            intent_act[i] = 4 if (dy != 0 or dx != 0) else 0
            continue

        # 回避採用
        if best_a in (1, 2) and best_escape_u > best_u and best_escape_u > -5:
            intent_y[i] = int(best_ey)
            intent_x[i] = int(best_ex)
            intent_act[i] = 4
        else:
            intent_y[i] = int(best_y)
            intent_x[i] = int(best_x)
            intent_act[i] = int(best_a)

    # v19：哲学/通常型ごとの「思考結果として選ばれた行動」を記録する。
    # これにより、型の増減を出生・死亡だけでなく、採取/交尾/回避/捕食などの行動偏りから説明できる。
    action_counts = np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32)
    gp_safe = np.array([philo_index(v) for v in gene_philo], dtype=np.int32)
    for ph_i in range(PHILO_TYPE_COUNT):
        mask = gp_safe == int(ph_i)
        if mask.any():
            action_counts[ph_i] = np.bincount(intent_act[mask].astype(np.int32), minlength=len(PHILO_ACTION_LABELS)).astype(np.int32)[:len(PHILO_ACTION_LABELS)]
    w["stat_philo_action_counts"] = action_counts

    w["intent_y"] = intent_y.astype(np.int32)
    w["intent_x"] = intent_x.astype(np.int32)
    w["intent_act"] = intent_act.astype(np.int8)

# -------------------------
# 戦闘（仕様そのまま）
# -------------------------
def battle_pair(i, j, w, rng):
    strength = w["strength"]
    bag = w["bag"]

    si = int(strength[i]); sj = int(strength[j])

    # upset が起きたか
    upset_happened = False

    if si == sj:
        i_wins = (rng.random() < 0.5)
    else:
        strong_is_i = (si > sj)
        if rng.random() < float(upset_p):
            i_wins = (not strong_is_i)
            upset_happened = True
        else:
            i_wins = strong_is_i

    if upset_happened:
        w["stat_battle_upset"] += 1

    winner, loser = (i, j) if i_wins else (j, i)

    transfer = int(bag[loser] // 2)
    if transfer > 0:
        bag[loser] -= transfer
        bag[winner] += transfer
        w["stat_battle_transfer"] += transfer
    
    # 遺伝子別：戦闘の資源移転（得た/失った）
    gene = w["gene_contest"]
    if int(gene[winner]) == 0:
        w["stat_gain_battle_hawk"] += int(transfer)
    else:
        w["stat_gain_battle_dove"] += int(transfer)

    if int(gene[loser]) == 0:
        w["stat_cost_battle_hawk"] += int(transfer)
    else:
        w["stat_cost_battle_dove"] += int(transfer)

    gene_philo = w.get("gene_philo", None)
    if gene_philo is not None and int(transfer) > 0:
        ph_w = philo_index(gene_philo[winner])
        ph_l = philo_index(gene_philo[loser])
        w["stat_philo_battle_gain"][ph_w] += int(transfer)
        w["stat_philo_battle_cost"][ph_l] += int(transfer)

def contest_cost(V_cell: int, C_base: int, C_perV: int) -> int:
    # すべて整数
    return int(C_base) + int(C_perV) * int(V_cell)

def apply_resource_contest(w, intent_y, intent_x, intent_act, rng):
    """
    資源セルの取り合い（contest）を先に解決して、
    - 資源(resource)と所持資源(bag)を更新
    - 「勝者をそのターゲットセルに必ず置く」ための forced_targets を返す
    - contestで採取処理を済ませた個体を skip できるように mask を返す
    """
    ys = w["ys"]; xs = w["xs"]
    bag = w["bag"]
    resource = w["resource"]
    gene = w["gene_contest"]

    n = len(xs)
    forced_targets = {}  # (ty,tx) -> winner_index
    gathered_done = np.zeros(n, dtype=bool)

    if (not enable_contest) or n == 0:
        return forced_targets, gathered_done

    # gather(=2) だけ対象
    gatherers = np.where(intent_act == 2)[0]
    if len(gatherers) <= 1:
        return forced_targets, gathered_done

    # ターゲットセルごとに候補を束ねる
    cell_to_ids = {}
    for i in gatherers:
        ty = int(intent_y[i]); tx = int(intent_x[i])
        cell_to_ids.setdefault((ty, tx), []).append(int(i))

    for (ty, tx), ids in cell_to_ids.items():
        if len(ids) <= 1:
            continue

        V_cell = int(min(int(resource[ty, tx]), int(gather_amount)))
        if V_cell <= 0:
            continue

        hawks = [i for i in ids if int(gene[i]) == 0]
        doves = [i for i in ids if int(gene[i]) == 1]

        w["stat_contest_cells"] += 1
        w["stat_contest_events"] += 1

        if len(hawks) >= 1:
            # ハトは譲る（0）、タカ同士で争う → タカ勝者がV獲得、タカ敗者がC支払い
            winner = int(hawks[int(rng.integers(0, len(hawks)))])
            forced_targets[(ty, tx)] = winner

            bag[winner] += V_cell
            w["stat_contest_v_paid"] += V_cell
            w["stat_contest_hawk_win"] += 1

            # 遺伝子別：contestのV獲得（勝者）
            if int(gene[winner]) == 0:
                w["stat_gain_contest_hawk"] += int(V_cell)
            else:
                w["stat_gain_contest_dove"] += int(V_cell)

            losers = []
            if len(hawks) >= 2:
                w["stat_contest_hh_events"] += 1
                C_loss = contest_cost(V_cell, int(contest_C_base), int(contest_C_perV))
                losers = [i for i in hawks if i != winner]
                for lo in losers:
                    bag[lo] -= C_loss
                w["stat_contest_cost_paid"] += int(C_loss) * int(len(losers))

            for lo in losers:
                if int(gene[lo]) == 0:
                    w["stat_cost_contest_hawk"] += int(C_loss)
                else:
                    w["stat_cost_contest_dove"] += int(C_loss)

            resource[ty, tx] -= V_cell

            # contestで採取完了扱い（勝者だけが採取したことにする）
            gathered_done[winner] = True

        else:
            # 全員ハト：分け合う（Vを等分、余りはランダム配布）
            forced = int(doves[int(rng.integers(0, len(doves)))])
            forced_targets[(ty, tx)] = forced

            q = V_cell // len(doves)
            r = V_cell - q * len(doves)

            for i in doves:
                bag[i] += q
                gathered_done[i] = True
            
            # 遺伝子別：contestのV獲得（分配）
            w["stat_gain_contest_dove"] += int(q) * int(len(doves))

            if r > 0:
                picks = rng.choice(doves, size=int(r), replace=False if r <= len(doves) else True)
                for i in picks:
                    bag[int(i)] += 1
            w["stat_gain_contest_dove"] += int(r)

            resource[ty, tx] -= V_cell
            w["stat_contest_v_paid"] += V_cell

    w["bag"] = bag
    w["resource"] = resource
    
    return forced_targets, gathered_done

    # ターゲットセルごとに候補を束ねる
    cell_to_ids = {}
    for i in gatherers:
        ty = int(intent_y[i]); tx = int(intent_x[i])
        cell_to_ids.setdefault((ty, tx), []).append(int(i))

    for (ty, tx), ids in cell_to_ids.items():
        if len(ids) <= 1:
            continue

        V_cell = int(min(int(resource[ty, tx]), int(gather_amount)))
        if V_cell <= 0:
            continue

        hawks = [i for i in ids if int(gene[i]) == 0]
        doves = [i for i in ids if int(gene[i]) == 1]

        w["stat_contest_cells"] += 1
        w["stat_contest_events"] += 1

        if len(hawks) >= 1:
            # ハトは譲る（0）、タカ同士で争う → タカ勝者がV獲得、タカ敗者がC支払い
            winner = int(hawks[int(rng.integers(0, len(hawks)))])
            forced_targets[(ty, tx)] = winner

            bag[winner] += V_cell
            w["stat_contest_v_paid"] += V_cell
            w["stat_contest_hawk_win"] += 1

            if len(hawks) >= 2:
                w["stat_contest_hh_events"] += 1
                C_loss = contest_cost(V_cell, int(contest_C_base), int(contest_C_perV))
                losers = [i for i in hawks if i != winner]
                for lo in losers:
                    bag[lo] -= C_loss
                w["stat_contest_cost_paid"] += int(C_loss) * int(len(losers))

            resource[ty, tx] -= V_cell

            # contestで採取完了扱い（勝者だけが採取したことにする）
            gathered_done[winner] = True

        else:
            # 全員ハト：分け合う（Vを等分、余りはランダム配布）
            forced = int(doves[int(rng.integers(0, len(doves)))])
            forced_targets[(ty, tx)] = forced

            q = V_cell // len(doves)
            r = V_cell - q * len(doves)

            for i in doves:
                bag[i] += q
                gathered_done[i] = True

            if r > 0:
                picks = rng.choice(doves, size=int(r), replace=False if r <= len(doves) else True)
                for i in picks:
                    bag[int(i)] += 1

            resource[ty, tx] -= V_cell
            w["stat_contest_v_paid"] += V_cell

    w["bag"] = bag
    w["resource"] = resource
    return forced_targets, gathered_done

def resolve_collisions(prev_y, prev_x, tgt_y, tgt_x, H, W, rng, search_r=2, forced_targets=None):
    """
    衝突解消（1セル1個体）。
    forced_targets={(ty,tx): winner_index} があれば
    その個体を先に (ty,tx) に確定配置する。
    """
    n = len(tgt_x)
    out_y = prev_y.copy()
    out_x = prev_x.copy()

    occ = np.zeros((H, W), dtype=bool)
    fixed = np.zeros(n, dtype=bool)

    def ring_positions(by, bx, r):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue
                yy = (by + dy) % H
                xx = (bx + dx) % W
                yield yy, xx

    def try_place(i, by, bx):
        by = int(by); bx = int(bx)
        if not occ[by, bx]:
            occ[by, bx] = True
            out_y[i], out_x[i] = by, bx
            return True

        for rr in range(1, int(search_r) + 1):
            for yy, xx in ring_positions(by, bx, rr):
                if not occ[yy, xx]:
                    occ[yy, xx] = True
                    out_y[i], out_x[i] = yy, xx
                    return True
        return False

    # ---- forced を先に固定 ----
    if forced_targets:
        for (ty, tx), wi in forced_targets.items():
            wi = int(wi)
            ty = int(ty); tx = int(tx)
            if 0 <= wi < n and (not fixed[wi]) and (not occ[ty, tx]):
                occ[ty, tx] = True
                out_y[wi], out_x[wi] = ty, tx
                fixed[wi] = True

    # ---- 残りを通常処理 ----
    order = rng.permutation(n)
    for i in order:
        i = int(i)
        if fixed[i]:
            continue
        if not try_place(i, tgt_y[i], tgt_x[i]):
            if not try_place(i, prev_y[i], prev_x[i]):
                yy, xx = np.where(~occ)
                if len(yy) > 0:
                    k = int(rng.integers(0, len(yy)))
                    occ[int(yy[k]), int(xx[k])] = True
                    out_y[i], out_x[i] = int(yy[k]), int(xx[k])

    return out_y.astype(np.int32), out_x.astype(np.int32)

    # ---- forced を先に固定 ----
    if forced_targets:
        for (ty, tx), wi in forced_targets.items():
            wi = int(wi)
            if 0 <= wi < n and (not fixed[wi]) and (not occ[int(ty), int(tx)]):
                occ[int(ty), int(tx)] = True
                out_y[wi], out_x[wi] = int(ty), int(tx)
                fixed[wi] = True

    # ---- 残りを通常処理 ----
    order = rng.permutation(n)
    for i in order:
        i = int(i)
        if fixed[i]:
            continue
        if not try_place(i, tgt_y[i], tgt_x[i]):
            if not try_place(i, prev_y[i], prev_x[i]):
                yy, xx = np.where(~occ)
                if len(yy) > 0:
                    k = int(rng.integers(0, len(yy)))
                    occ[yy[k], xx[k]] = True
                    out_y[i], out_x[i] = int(yy[k]), int(xx[k])

    return out_y.astype(np.int32), out_x.astype(np.int32)

# -------------------------
# フェーズ④ 行動（移動→採取→戦闘）
# -------------------------
def phase4_action():
    w = st.session_state.world
    ensure_ecology_arrays(w)
    rng = np.random.default_rng(int(seed) + st.session_state.gen * 1009 + 4)

    w["evt_battle_cells"] = []
    w["evt_predation_cells"] = []
    w["battles"] = 0

    ys = w["ys"]
    xs = w["xs"]
    team = w["team"]
    bag = w["bag"]
    resource = w["resource"]
    direction = w["dir"]
    sex = w["sex"]

    n = len(xs)
    if n == 0:
        return

    # 行動前の位置（可視化用）
    w["last_prev"] = (ys.copy(), xs.copy())
    intent_y = w["intent_y"]
    intent_x = w["intent_x"]
    intent_act = w["intent_act"]
    forced_targets, gathered_done = apply_resource_contest(w, intent_y, intent_x, intent_act, rng)

    # ---- 移動コスト（「動こうとした」個体から引く：衝突で止まってもコストは払う）----
    moved_intent = (intent_y != ys) | (intent_x != xs)
    bag[moved_intent] -= int(move_cost)
    w["stat_move_cost_paid"] += int(move_cost) * int(moved_intent.sum())
    gene_philo_move = w.get("gene_philo", np.full(n, NORMAL_PHILO_VALUE, dtype=np.int8))
    if int(move_cost) > 0 and n > 0:
        w["stat_philo_move_cost"] += (
            np.bincount(gene_philo_move[moved_intent].astype(np.int32), minlength=PHILO_TYPE_COUNT).astype(np.int32)
            * int(move_cost)
        )

    gene = w["gene_contest"]
    if n > 0:
        w["stat_cost_move_hawk"] += int(int(move_cost) * int(((gene == 0) & moved_intent).sum()))
        w["stat_cost_move_dove"] += int(int(move_cost) * int(((gene == 1) & moved_intent).sum()))

    # ---- 向き更新（「動こうとした向き」を向く）----
    dy = (intent_y - ys).astype(np.int32)
    dx = (intent_x - xs).astype(np.int32)
    # 移動半径が2以上でも正しく向きを決めるため、トーラス上の最短差分に直す。
    dy = np.where(dy > H // 2, dy - H, dy)
    dy = np.where(dy < -H // 2, dy + H, dy)
    dx = np.where(dx > W // 2, dx - W, dx)
    dx = np.where(dx < -W // 2, dx + W, dx)

    for i in np.where(moved_intent)[0]:
        ddy = int(dy[i]); ddx = int(dx[i])
        if abs(ddx) > abs(ddy):
            direction[i] = 1 if ddx > 0 else 3
        else:
            direction[i] = 2 if ddy > 0 else 0

    # ---- 衝突解消（1セル1個体）----
    ys2, xs2 = resolve_collisions(ys, xs, intent_y, intent_x, H, W, rng, search_r=2, forced_targets=forced_targets)
    ys = ys2
    xs = xs2

    # ---- 統計：移動/衝突 ----
    w["stat_move_intent"] += int(moved_intent.sum())
    prev_y, prev_x = w["last_prev"]
    diff_from_intent = ((ys2 != intent_y) | (xs2 != intent_x)) & moved_intent
    w["stat_collision"] += int(diff_from_intent.sum())
    moved_actual = (ys2 != prev_y) | (xs2 != prev_x)
    w["stat_move_actual"] += int(moved_actual.sum())
    dy = np.abs(ys2 - prev_y)
    dy = np.minimum(dy, H - dy)
    dx = np.abs(xs2 - prev_x)
    dx = np.minimum(dx, W - dx)
    w["stat_move_dist_sum"] += float((dy + dx).sum())

    neighbor8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # ---- 捕食/被食（phase3で捕食を選んだ個体が、近傍の他チーム個体を狙う）----
    if enable_predation:
        pos_to_i = {(int(ys[i]), int(xs[i])): i for i in range(n)}
        predators = list(np.where(intent_act == 6)[0])
        already_targeted = set()

        for i in predators:
            i = int(i)
            if int(bag[i]) <= 0:
                continue
            yi, xi = int(ys[i]), int(xs[i])

            candidates = []
            # intent先を最優先候補にしつつ、周囲8近傍も見る
            target_pos = (int(intent_y[i]), int(intent_x[i]))
            j0 = pos_to_i.get(target_pos)
            if j0 is not None:
                candidates.append(int(j0))
            for oy, ox in neighbor8:
                yy = (yi + oy) % H
                xx = (xi + ox) % W
                j = pos_to_i.get((yy, xx))
                if j is not None:
                    candidates.append(int(j))

            # 重複除去
            seen = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]

            best_j = None
            best_score = -1e18
            for j in candidates:
                if j == i or j in already_targeted:
                    continue
                if int(team[j]) == int(team[i]):
                    continue
                if bool(enable_kin_avoidance) and kinship_score(w, i, j) >= float(kin_avoid_threshold):
                    continue
                p_win = predation_success_probability(int(w["strength"][i]), int(w["strength"][j]), int(bag[i]), int(bag[j]))
                expected_gain = p_win * max(1.0, float(bag[j]) * float(predation_gain_rate))
                score = expected_gain - (1.0 - p_win) * float(predation_fail_cost)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                continue

            j = int(best_j)
            already_targeted.add(j)
            w["stat_predation_attempt"] += 1
            ph_i = philo_index(w.get("gene_philo", np.full(n, NORMAL_PHILO_VALUE, dtype=np.int8))[i])
            w["stat_philo_predation_attempt"][ph_i] += 1
            p_win = predation_success_probability(int(w["strength"][i]), int(w["strength"][j]), int(bag[i]), int(bag[j]))

            if rng.random() < p_win:
                take = int(max(1, min(int(bag[j]), int(round(float(bag[j]) * float(predation_gain_rate))))))
                if take > 0:
                    bag[j] -= take
                    bag[i] += take
                    w["stat_predation_gain"] += int(take)
                    w["stat_philo_predation_gain"][ph_i] += int(take)
                # 捕食成功はしばしば致死的。ただし資源が残る場合は逃げ延びる余地を残す。
                if int(bag[j]) <= 0:
                    w["stat_predation_kill"] += 1
                w["stat_predation_success"] += 1
                w["stat_philo_predation_success"][ph_i] += 1
                w["evt_predation_cells"].append((int(ys[j]), int(xs[j])))
            else:
                bag[i] -= int(predation_fail_cost)
                w["stat_predation_fail"] += 1
                w["stat_philo_predation_fail"][ph_i] += 1

    # ---- 交尾（互いに選択＆隣接 → 次世代①で出生予約）----
    w["evt_mate_pairs"] = []
    if enable_birth and n > 1:
        sex = w["sex"]
        vision = w["vision"]
        strength = w["strength"]
        gene = w["gene_contest"]
        gene_pred = w["gene_predation"]
        gene_philo = w["gene_philo"]
        uid = w["uid"]
        lineage = w["lineage"]

        pos_to_i = {(int(ys[i]), int(xs[i])): i for i in range(n)}
        mators = set(np.where(intent_act == 5)[0])
        done_pairs = set()
        mated = set()

        for i in list(mators):
            i = int(i)
            if i in mated or int(bag[i]) <= 0:
                continue
            yi, xi = int(ys[i]), int(xs[i])

            candidates = []
            for oy, ox in neighbor8:
                yj = (yi + oy) % H
                xj = (xi + ox) % W
                j = pos_to_i.get((yj, xj))
                if j is None or j == i:
                    continue
                j = int(j)
                if j not in mators or j in mated:
                    continue
                if int(team[j]) != int(team[i]):
                    continue
                if int(sex[j]) == int(sex[i]):
                    continue
                if int(bag[j]) <= 0:
                    continue
                candidates.append(j)

            if not candidates:
                continue

            # 血縁度が低い候補を優先。近縁でも環境次第で完全禁止にはしない。
            candidates.sort(key=lambda j: kinship_score(w, i, j) if bool(enable_kin_avoidance) else 0.0)
            for j in candidates:
                a, b = (i, j) if i < j else (j, i)
                pair_key = (int(a), int(b))
                if pair_key in done_pairs:
                    continue

                rel = kinship_score(w, a, b) if bool(enable_kin_avoidance) else 0.0
                if bool(enable_kin_avoidance) and rel >= float(kin_avoid_threshold) and rng.random() < float(kin_avoid_strength):
                    w["stat_kin_avoided"] += 1
                    continue

                done_pairs.add(pair_key)
                w["stat_mate_attempt"] += 1
                ph_a = philo_index(gene_philo[a])
                ph_b = philo_index(gene_philo[b])
                w["stat_philo_mate_attempt"][ph_a] += 1
                w["stat_philo_mate_attempt"][ph_b] += 1

                total_bag = int(bag[a]) + int(bag[b])
                p = float(birth_probability(np.array([total_bag], dtype=np.int32), birth_k)[0])
                if rng.random() > p:
                    continue

                fee = int(birth_fee)
                fee_a = fee // 2
                fee_b = fee - fee_a
                if int(bag[a]) < fee_a or int(bag[b]) < fee_b:
                    continue

                share_a_plan = int(int(bag[a]) * float(child_share_pct) / 100.0)
                share_b_plan = int(int(bag[b]) * float(child_share_pct) / 100.0)
                invest_a_max = max(0, int(bag[a]) - fee_a)
                invest_b_max = max(0, int(bag[b]) - fee_b)
                pay_a = min(share_a_plan, invest_a_max)
                pay_b = min(share_b_plan, invest_b_max)
                child_bag = int(min(int(pay_a + pay_b), int(child_bag_cap)))

                bag[a] -= int(fee_a + pay_a)
                bag[b] -= int(fee_b + pay_b)

                w["stat_birth_fee_paid"] += int(fee_a + fee_b)
                w["stat_child_resource_given"] += int(pay_a + pay_b)

                if int(gene[a]) == 0:
                    w["stat_cost_birthfee_hawk"] += int(fee_a)
                    w["stat_cost_childshare_hawk"] += int(pay_a)
                else:
                    w["stat_cost_birthfee_dove"] += int(fee_a)
                    w["stat_cost_childshare_dove"] += int(pay_a)

                if int(gene[b]) == 0:
                    w["stat_cost_birthfee_hawk"] += int(fee_b)
                    w["stat_cost_childshare_hawk"] += int(pay_b)
                else:
                    w["stat_cost_birthfee_dove"] += int(fee_b)
                    w["stat_cost_childshare_dove"] += int(pay_b)

                child_vis = int(round((int(vision[a]) + int(vision[b])) / 2))
                if rng.random() < float(vision_mutate):
                    child_vis += int(rng.integers(-1, 2))
                child_vis = int(np.clip(child_vis, 0, 12))

                child_str = int(round((int(strength[a]) + int(strength[b])) / 2))
                if rng.random() < float(str_mutate):
                    child_str += int(rng.integers(-1, 2))
                child_str = int(max(1, child_str))

                child_dir = int(rng.integers(0, 4))
                child_sex = int(rng.integers(0, 2))
                child_team = int(team[a])

                child_gene = int(gene[a]) if (rng.random() < 0.5) else int(gene[b])
                if rng.random() < float(contest_mu):
                    child_gene = 1 - int(child_gene)

                # 捕食傾向は別遺伝子として親のどちらかから継承。突然変異分布変更はまだ入れない。
                child_gene_pred = int(gene_pred[a]) if (rng.random() < 0.5) else int(gene_pred[b])

                # 哲学的行動遺伝子も、親のどちらかからそのまま継承する。
                # v19では「どちらの親型から子型へコピーされたか」を記録する。
                # 突然変異の分布変更はまだ入れないため、ここでは新しい哲学型は生成しない。
                if rng.random() < 0.5:
                    child_gene_philo = int(gene_philo[a])
                    child_source_philo = philo_index(gene_philo[a])
                else:
                    child_gene_philo = int(gene_philo[b])
                    child_source_philo = philo_index(gene_philo[b])

                mother = a if int(sex[a]) == 1 else (b if int(sex[b]) == 1 else a)
                py, px = int(ys[mother]), int(xs[mother])
                # 子のlineageは母系/父系どちらかを引き継ぐ。近親回避判定用の粗い系統情報。
                child_lineage = int(lineage[mother])

                w["pending_births"].append({
                    "py": py, "px": px,
                    "team": child_team,
                    "bag": child_bag,
                    "vision": child_vis,
                    "strength": child_str,
                    "dir": child_dir,
                    "sex": child_sex,
                    "gene": int(child_gene),
                    "gene_predation": int(child_gene_pred),
                    "gene_philo": int(child_gene_philo),
                    "parent_a": int(uid[a]),
                    "parent_b": int(uid[b]),
                    "parent_philo_a": int(ph_a),
                    "parent_philo_b": int(ph_b),
                    "source_philo": int(child_source_philo),
                    "lineage": int(child_lineage),
                })

                if int(child_gene) == 0:
                    w["stat_birth_reserved_hawk"] += 1
                else:
                    w["stat_birth_reserved_dove"] += 1

                child_philo_idx = philo_index(child_gene_philo)
                w["stat_philo_birth_reserved"][child_philo_idx] += 1

                # v19：親子フロー。
                # 「子として増えた型」と「親として子を残した型」を分けることで、
                # 比率上昇の原因が親側の繁殖成功か、死亡回避か、相対的残存かを読みやすくする。
                w["stat_philo_parent_offspring_reserved"][ph_a] += 1
                w["stat_philo_parent_offspring_reserved"][ph_b] += 1
                pair_lo, pair_hi = sorted((int(ph_a), int(ph_b)))
                w["stat_philo_pair_reserved"][pair_lo, pair_hi] += 1
                w["stat_philo_parent_to_child_reserved"][ph_a, child_philo_idx] += 1
                w["stat_philo_parent_to_child_reserved"][ph_b, child_philo_idx] += 1
                w["stat_philo_source_to_child_reserved"][child_source_philo, child_philo_idx] += 1

                w["stat_mate_success"] += 1
                w["stat_philo_mate_success"][ph_a] += 1
                w["stat_philo_mate_success"][ph_b] += 1
                w["evt_mate_pairs"].append(((yi, xi), (int(ys[j]), int(xs[j]))))
                mated.add(i)
                mated.add(j)
                break

    # ---- 採取（同時）----
    if int(gather_amount) > 0:
        gatherers = np.where((intent_act == 2) & (~gathered_done))[0]
        gene = w["gene_contest"]

        for i in gatherers:
            i = int(i)
            if int(bag[i]) <= 0:
                continue
            y = int(ys[i]); x = int(xs[i])
            take = 0
            if resource[y, x] > 0:
                take = min(int(gather_amount), int(resource[y, x]))
                resource[y, x] -= take
                bag[i] += take

            w["stat_gathered"] += int(take)
            ph_i = philo_index(w.get("gene_philo", np.full(n, NORMAL_PHILO_VALUE, dtype=np.int8))[i])
            w["stat_philo_gather_gain"][ph_i] += int(take)
            if int(gene[i]) == 0:
                w["stat_gain_gather_hawk"] += int(take)
            else:
                w["stat_gain_gather_dove"] += int(take)

    # ---- 戦闘（仕様そのまま）----
    if enable_battle:
        pos_map = {}
        for i in range(n):
            pos_map.setdefault((int(ys[i]), int(xs[i])), []).append(i)

        attackers = np.where(intent_act == 3)[0]
        done = set()

        for i in attackers:
            i = int(i)
            if int(bag[i]) <= 0:
                continue
            cy, cx = int(ys[i]), int(xs[i])
            for ddy in (-1, 0, 1):
                for ddx in (-1, 0, 1):
                    if ddy == 0 and ddx == 0:
                        continue
                    yy = (cy + ddy) % H
                    xx = (cx + ddx) % W
                    js = pos_map.get((yy, xx), [])
                    for j in js:
                        if i == j:
                            continue
                        if int(bag[j]) <= 0:
                            continue
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) in done:
                            continue
                        done.add((a, b))
                        battle_pair(i, j, w, rng)
                        w["battles"] += 1
                        w["evt_battle_cells"].append((yy, xx))

    # ---- 反映 ----
    w["ys"] = ys.astype(np.int32)
    w["xs"] = xs.astype(np.int32)
    w["bag"] = bag.astype(np.int32)
    w["resource"] = resource
    w["dir"] = direction.astype(np.int8)
    w["last_act"] = intent_act.copy()

# -------------------------
# フェーズ⑤ 生死（維持コスト・寿命）
# -------------------------
def log_generation():
    w = st.session_state.world
    ensure_ecology_arrays(w)
    n = int(len(w["xs"]))

    resource = w["resource"]
    res_total = int(resource.sum())
    res_cells = int((resource > 0).sum())
    res_cell_ratio = float(res_cells) / float(H * W)

    ys = w["ys"]
    xs = w["xs"]
    bag = w["bag"]
    strength = w["strength"]
    vision = w["vision"]
    age = w["age"]
    team = w["team"]
    gene = w["gene_contest"]
    gene_pred = w.get("gene_predation", np.zeros(n, dtype=np.int8))
    gene_philo = w.get("gene_philo", np.full(n, NORMAL_PHILO_VALUE, dtype=np.int8))

    contest_counts = _safe_counts(gene, 2)
    pred_counts = _safe_counts(gene_pred, 2)
    philo_counts = _safe_counts(gene_philo, PHILO_TYPE_COUNT)

    # v20.2 bugfix: 通常個体を追加した後、log_generation内で
    # normal_count / philosophy_count / philo_only_counts を定義していなかったため、
    # Streamlit Cloudの自動実行中にNameErrorで停止していた。
    normal_count = int(philo_counts[int(NORMAL_PHILO_VALUE)]) if len(philo_counts) > int(NORMAL_PHILO_VALUE) else 0
    philosophy_count = max(int(n) - normal_count, 0)
    philo_only_counts = np.asarray([philo_counts[i] for i in PHILOSOPHY_VALUES], dtype=np.int32)

    prev_contest = np.asarray(w.get("prev_contest_counts", np.maximum(contest_counts, 1)), dtype=np.int32)
    prev_pred = np.asarray(w.get("prev_predation_counts", np.maximum(pred_counts, 1)), dtype=np.int32)
    prev_philo = np.asarray(w.get("prev_philo_counts", np.maximum(philo_counts, 1)), dtype=np.int32)
    prev_pop = int(w.get("prev_pop_count_for_W", max(n, 1)))

    W_contest = _safe_W(contest_counts, prev_contest)
    W_pred = _safe_W(pred_counts, prev_pred)
    W_philo = _safe_W(philo_counts, prev_philo)
    W_pop = float(n) / max(float(prev_pop), 1.0)

    density_now = torus_density_map(w["ys"], w["xs"], H, W, radius=int(density_radius))
    density_on_agents = density_now[w["ys"], w["xs"]] if n > 0 else np.array([], dtype=np.int32)

    def _mean(x):
        return float(x.mean()) if len(x) > 0 else 0.0

    def _pctl(x, p):
        return float(np.percentile(x, p)) if len(x) > 0 else 0.0

    # チーム別
    if n > 0:
        m0 = (team == 0)
        m1 = (team == 1)
        n0 = int(m0.sum()); n1 = int(m1.sum())
    else:
        n0 = n1 = 0
        m0 = m1 = np.array([], dtype=bool)

    row = {
        "世代（回）": int(st.session_state.gen),

        # 基本
        "個体数（体）": n,
        "赤個体数（体）": n0,
        "青個体数（体）": n1,
        "設定:密度依存ON": int(bool(enable_density_dependence)),
        "設定:局所資源再生ON": int(bool(enable_local_resource_regen)),
        "設定:近親回避ON": int(bool(enable_kin_avoidance)),
        "設定:哲学遺伝子ON": int(bool(enable_philo_gene)),
        "設定:捕食ON": int(bool(enable_predation)),

        # 資源
        "資源総量（単位）": res_total,
        "資源マス数（マス）": res_cells,
        "資源マス割合（0-1）": res_cell_ratio,

        # 個体状態（分布寄り）
        "平均所持資源（単位/体）": _mean(bag),
        "所持資源p10（単位）": _pctl(bag, 10),
        "所持資源中央値（単位）": _pctl(bag, 50),
        "所持資源p90（単位）": _pctl(bag, 90),
        "資源格差Gini（0-1）": gini(bag) if n > 0 else 0.0,

        "平均肉体強度（値/体）": _mean(strength),
        "平均認識半径（マス）": _mean(vision),

        "平均年齢（世代）": _mean(age),
        "年齢p90（世代）": _pctl(age, 90),
        "最大年齢（世代）": int(age.max()) if n > 0 else 0,

        # 争奪（タカ/ハト）遺伝子の構成
        "タカ数（体）": int((gene == 0).sum()) if n > 0 else 0,
        "ハト数（体）": int((gene == 1).sum()) if n > 0 else 0,
        "タカ比率（0-1）": float((gene == 0).mean()) if n > 0 else 0.0,

        # ===== 遺伝子の世代ごとの流れ：コピー数・頻度・増殖率W =====
        "個体群全体W（増殖率）": W_pop,
        "タカ 適応度W（コピー増殖率）": float(W_contest[0]),
        "ハト 適応度W（コピー増殖率）": float(W_contest[1]),
        "非捕食 適応度W（コピー増殖率）": float(W_pred[0]),
        "捕食 適応度W（コピー増殖率）": float(W_pred[1]),
        "ヒューム型 数（体）": int(philo_counts[0]),
        "ストア型 数（体）": int(philo_counts[1]),
        "デカルト型 数（体）": int(philo_counts[2]),
        "カント型 数（体）": int(philo_counts[3]),
        "ヒューム型 比率（0-1）": float(philo_counts[0]) / max(n, 1),
        "ストア型 比率（0-1）": float(philo_counts[1]) / max(n, 1),
        "デカルト型 比率（0-1）": float(philo_counts[2]) / max(n, 1),
        "カント型 比率（0-1）": float(philo_counts[3]) / max(n, 1),
        "ヒューム型 W": float(W_philo[0]),
        "ストア型 W": float(W_philo[1]),
        "デカルト型 W": float(W_philo[2]),
        "カント型 W": float(W_philo[3]),
        "争奪遺伝子多様度（Simpson）": _gene_diversity_from_counts(contest_counts),
        "捕食遺伝子多様度（Simpson）": _gene_diversity_from_counts(pred_counts),
        "通常個体数（体）": normal_count,
        "哲学個体数（体）": philosophy_count,
        "通常個体割合（0-1）": float(normal_count) / max(int(n), 1),
        "哲学個体割合（0-1）": float(philosophy_count) / max(int(n), 1),
        "行動型多様度（通常含むSimpson）": _gene_diversity_from_counts(philo_counts),
        "哲学遺伝子多様度（Simpson）": _gene_diversity_from_counts(philo_only_counts),

        # ===== 遺伝子別（タカ/ハト） =====
        "タカ 平均所持資源（単位/体）": _mean(bag[gene == 0]),
        "ハト 平均所持資源（単位/体）": _mean(bag[gene == 1]),
        "タカ 資源格差Gini（0-1）": gini(bag[gene == 0]) if int((gene == 0).sum()) > 1 else 0.0,
        "ハト 資源格差Gini（0-1）": gini(bag[gene == 1]) if int((gene == 1).sum()) > 1 else 0.0,

        "タカ 平均肉体強度（値/体）": _mean(strength[gene == 0]),
        "ハト 平均肉体強度（値/体）": _mean(strength[gene == 1]),
        "タカ 平均認識半径（マス）": _mean(vision[gene == 0]),
        "ハト 平均認識半径（マス）": _mean(vision[gene == 1]),
        "タカ 平均年齢（世代）": _mean(age[gene == 0]),
        "ハト 平均年齢（世代）": _mean(age[gene == 1]),

        # ===== チーム内の遺伝子構成（赤/青の中でタカがどれくらいか） =====
        "赤タカ数（体）": int(((team == 0) & (gene == 0)).sum()) if n > 0 else 0,
        "赤ハト数（体）": int(((team == 0) & (gene == 1)).sum()) if n > 0 else 0,
        "青タカ数（体）": int(((team == 1) & (gene == 0)).sum()) if n > 0 else 0,
        "青ハト数（体）": int(((team == 1) & (gene == 1)).sum()) if n > 0 else 0,
        "赤タカ比率（0-1）": float(((team == 0) & (gene == 0)).sum()) / max(int((team == 0).sum()), 1),
        "青タカ比率（0-1）": float(((team == 1) & (gene == 0)).sum()) / max(int((team == 1).sum()), 1),

        # 争奪（contest）イベント統計
        "争奪セル数（マス/世代）": int(w.get("stat_contest_cells", 0)),
        "争奪イベント数（回/世代）": int(w.get("stat_contest_events", 0)),
        "タカ勝利数（回/世代）": int(w.get("stat_contest_hawk_win", 0)),
        "タカ同士争い（回/世代）": int(w.get("stat_contest_hh_events", 0)),
        "争奪で得たV合計（単位/世代）": int(w.get("stat_contest_v_paid", 0)),
        "争奪で支払ったC合計（単位/世代）": int(w.get("stat_contest_cost_paid", 0)),


        # 世代イベント（既存 + 追加）
        "出生数（体/世代）": int(w.get("births", 0)),
        "死亡数（体/世代）": int(w.get("deaths", 0)),
        
        # ===== 遺伝子別（④成立→①出生の成功率）=====
        "タカ 出生予約数（回/世代）": int(w.get("stat_birth_reserved_hawk", 0)),
        "ハト 出生予約数（回/世代）": int(w.get("stat_birth_reserved_dove", 0)),
        "タカ 実出生数（体/世代）": int(w.get("stat_birth_real_hawk", 0)),
        "ハト 実出生数（体/世代）": int(w.get("stat_birth_real_dove", 0)),
        "タカ 予約→出生成功率（0-1）": float(w.get("stat_birth_real_hawk", 0)) / max(int(w.get("stat_birth_reserved_hawk", 0)), 1),
        "ハト 予約→出生成功率（0-1）": float(w.get("stat_birth_real_dove", 0)) / max(int(w.get("stat_birth_reserved_dove", 0)), 1),

        # ===== 遺伝子別（死亡率）=====
        "タカ 死亡数（体/世代）": int(w.get("stat_death_hawk", 0)),
        "ハト 死亡数（体/世代）": int(w.get("stat_death_dove", 0)),
        "タカ 死亡率（0-1/世代）": float(w.get("stat_death_hawk", 0)) / max(int(w.get("stat_pop_hawk_before_death", 0)), 1),
        "ハト 死亡率（0-1/世代）": float(w.get("stat_death_dove", 0)) / max(int(w.get("stat_pop_dove_before_death", 0)), 1),

        # ===== 遺伝子別：資源収支（単位/世代）=====
        "タカ 獲得:採取（単位/世代）": int(w.get("stat_gain_gather_hawk", 0)),
        "ハト 獲得:採取（単位/世代）": int(w.get("stat_gain_gather_dove", 0)),

        "タカ 獲得:争奪V（単位/世代）": int(w.get("stat_gain_contest_hawk", 0)),
        "ハト 獲得:争奪V（単位/世代）": int(w.get("stat_gain_contest_dove", 0)),
        "タカ 支払:争奪C（単位/世代）": int(w.get("stat_cost_contest_hawk", 0)),
        "ハト 支払:争奪C（単位/世代）": int(w.get("stat_cost_contest_dove", 0)),

        "タカ 獲得:戦闘（単位/世代）": int(w.get("stat_gain_battle_hawk", 0)),
        "ハト 獲得:戦闘（単位/世代）": int(w.get("stat_gain_battle_dove", 0)),
        "タカ 支払:戦闘（単位/世代）": int(w.get("stat_cost_battle_hawk", 0)),
        "ハト 支払:戦闘（単位/世代）": int(w.get("stat_cost_battle_dove", 0)),

        "タカ 支払:移動（単位/世代）": int(w.get("stat_cost_move_hawk", 0)),
        "ハト 支払:移動（単位/世代）": int(w.get("stat_cost_move_dove", 0)),
        "タカ 支払:維持（単位/世代）": int(w.get("stat_cost_upkeep_hawk", 0)),
        "ハト 支払:維持（単位/世代）": int(w.get("stat_cost_upkeep_dove", 0)),

        "タカ 支払:出生コスト（単位/世代）": int(w.get("stat_cost_birthfee_hawk", 0)),
        "ハト 支払:出生コスト（単位/世代）": int(w.get("stat_cost_birthfee_dove", 0)),
        "タカ 支払:子への分配（単位/世代）": int(w.get("stat_cost_childshare_hawk", 0)),
        "ハト 支払:子への分配（単位/世代）": int(w.get("stat_cost_childshare_dove", 0)),

        # ネット（獲得−支払）
        "タカ 資源収支ネット（単位/世代）": (
            int(w.get("stat_gain_gather_hawk", 0))
            + int(w.get("stat_gain_contest_hawk", 0))
            + int(w.get("stat_gain_battle_hawk", 0))
            - int(w.get("stat_cost_contest_hawk", 0))
            - int(w.get("stat_cost_battle_hawk", 0))
            - int(w.get("stat_cost_move_hawk", 0))
            - int(w.get("stat_cost_upkeep_hawk", 0))
            - int(w.get("stat_cost_birthfee_hawk", 0))
            - int(w.get("stat_cost_childshare_hawk", 0))
        ),
        "ハト 資源収支ネット（単位/世代）": (
            int(w.get("stat_gain_gather_dove", 0))
            + int(w.get("stat_gain_contest_dove", 0))
            + int(w.get("stat_gain_battle_dove", 0))
            - int(w.get("stat_cost_contest_dove", 0))
            - int(w.get("stat_cost_battle_dove", 0))
            - int(w.get("stat_cost_move_dove", 0))
            - int(w.get("stat_cost_upkeep_dove", 0))
            - int(w.get("stat_cost_birthfee_dove", 0))
            - int(w.get("stat_cost_childshare_dove", 0))
        ),

        "戦闘回数（回/世代）": int(w.get("battles", 0)),

        "採取総量（単位/世代）": int(w.get("stat_gathered", 0)),

        "移動意図（体/世代）": int(w.get("stat_move_intent", 0)),
        "移動実行（体/世代）": int(w.get("stat_move_actual", 0)),
        "衝突（体/世代）": int(w.get("stat_collision", 0)),
        "移動成功率（0-1）": float(w.get("stat_move_actual", 0)) / max(int(w.get("stat_move_intent", 0)), 1),

        "交尾試行（回/世代）": int(w.get("stat_mate_attempt", 0)),
        "交尾成立（回/世代）": int(w.get("stat_mate_success", 0)),
        "交尾成立率（0-1）": float(w.get("stat_mate_success", 0)) / max(int(w.get("stat_mate_attempt", 0)), 1),

        "戦闘移転資源（単位/世代）": int(w.get("stat_battle_transfer", 0)),
        "弱者勝利（回/世代）": int(w.get("stat_battle_upset", 0)),
        "弱者勝率（0-1）": float(w.get("stat_battle_upset", 0)) / max(int(w.get("battles", 0)), 1),

        # 追加：コスト・資源発生
        "資源自然発生マス数（マス/世代）": int(w.get("stat_res_spawn_cells", 0)),
        "資源自然発生総量（単位/世代）": int(w.get("stat_res_spawned", 0)),
        "移動コスト支払総量（単位/世代）": int(w.get("stat_move_cost_paid", 0)),
        "維持コスト支払総量（単位/世代）": int(w.get("stat_upkeep_paid", 0)),
        "出生コスト支払総量（単位/世代）": int(w.get("stat_birth_fee_paid", 0)),
        "子へ分配した資源（単位/世代）": int(w.get("stat_child_resource_given", 0)),
        "出生失敗（空きなし）（体/世代）": int(w.get("stat_birth_failed_space", 0)),

        # 生態系モデル補正
        "平均局所密度（体/近傍）": _mean(density_on_agents),
        "最大局所密度（体/近傍）": int(density_on_agents.max()) if n > 0 else 0,
        "過密で抑制された出生候補（回/世代）": int(w.get("stat_birth_density_block", 0)),
        "過密で抑制された資源再生候補（マス/世代）": int(w.get("stat_density_spawn_block", 0)),
        "局所再生による資源発生量（単位/世代）": int(w.get("stat_local_resource_spawned", 0)),
        "近親交配回避（回/世代）": int(w.get("stat_kin_avoided", 0)),
        "捕食傾向個体数（体）": int((gene_pred == 1).sum()) if n > 0 else 0,
        "捕食傾向比率（0-1）": float((gene_pred == 1).mean()) if n > 0 else 0.0,
        "捕食試行（回/世代）": int(w.get("stat_predation_attempt", 0)),
        "捕食成功（回/世代）": int(w.get("stat_predation_success", 0)),
        "捕食失敗（回/世代）": int(w.get("stat_predation_fail", 0)),
        "捕食成功率（0-1）": float(w.get("stat_predation_success", 0)) / max(int(w.get("stat_predation_attempt", 0)), 1),
        "捕食獲得資源（単位/世代）": int(w.get("stat_predation_gain", 0)),
        "捕食で致死的被害（体/世代）": int(w.get("stat_predation_kill", 0)),
    }

    # 現在のPHILO_LABELSに合わせた、哲学遺伝子のコピー数・頻度・W。
    # 以前のラベル名が履歴に残っていても、新しい分析グラフはここを見る。
    for ph_i, ph_label in PHILO_LABELS.items():
        row[f"{ph_label} 数（体）"] = int(philo_counts[int(ph_i)])
        row[f"{ph_label} 比率（0-1）"] = float(philo_counts[int(ph_i)]) / max(int(n), 1)
        row[f"{ph_label} W"] = float(W_philo[int(ph_i)])

    # チーム別の格差も入れる（列は増えるけど強い）
    if n0 > 0:
        row["赤 平均所持資源"] = _mean(bag[m0])
        row["赤 Gini"] = gini(bag[m0])
    else:
        row["赤 平均所持資源"] = 0.0
        row["赤 Gini"] = 0.0

    if n1 > 0:
        row["青 平均所持資源"] = _mean(bag[m1])
        row["青 Gini"] = gini(bag[m1])
    else:
        row["青 平均所持資源"] = 0.0
        row["青 Gini"] = 0.0

    # ===== 哲学遺伝子別：状態・行動・資源収支の詳細 =====
    resource_underfoot = resource[ys, xs] if n > 0 else np.array([], dtype=np.int32)
    hunger_mask = bag <= int(birth_ready_bag) if n > 0 else np.array([], dtype=bool)
    for ph_i, ph_label in PHILO_LABELS.items():
        mask = (gene_philo == int(ph_i)) if n > 0 else np.array([], dtype=bool)
        row[f"{ph_label} 平均所持資源（単位/体）"] = _mean(bag[mask])
        row[f"{ph_label} 所持資源中央値（単位）"] = _pctl(bag[mask], 50)
        row[f"{ph_label} 平均年齢（世代）"] = _mean(age[mask])
        row[f"{ph_label} 平均肉体強度（値/体）"] = _mean(strength[mask])
        row[f"{ph_label} 平均認識半径（マス）"] = _mean(vision[mask])
        row[f"{ph_label} 平均足元資源（単位/マス）"] = _mean(resource_underfoot[mask]) if n > 0 else 0.0
        row[f"{ph_label} 平均局所密度（体/近傍）"] = _mean(density_on_agents[mask]) if n > 0 else 0.0
        row[f"{ph_label} 空腹個体比率（0-1）"] = float(hunger_mask[mask].mean()) if int(mask.sum()) > 0 else 0.0
        row[f"{ph_label} タカ数（体）"] = int((gene[mask] == 0).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} ハト数（体）"] = int((gene[mask] == 1).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} タカ比率（0-1）"] = float((gene[mask] == 0).mean()) if int(mask.sum()) > 0 else 0.0
        row[f"{ph_label} 非捕食数（体）"] = int((gene_pred[mask] == 0).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} 捕食傾向数（体）"] = int((gene_pred[mask] == 1).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} 捕食傾向比率（0-1）"] = float((gene_pred[mask] == 1).mean()) if int(mask.sum()) > 0 else 0.0
        row[f"{ph_label} 赤内数（体）"] = int(((team == 0) & mask).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} 青内数（体）"] = int(((team == 1) & mask).sum()) if int(mask.sum()) > 0 else 0
        row[f"{ph_label} 赤比率（型内0-1）"] = float(((team == 0) & mask).sum()) / max(int(mask.sum()), 1) if int(mask.sum()) > 0 else 0.0
        row[f"{ph_label} 青比率（型内0-1）"] = float(((team == 1) & mask).sum()) / max(int(mask.sum()), 1) if int(mask.sum()) > 0 else 0.0

        row[f"{ph_label} 出生予約（体/世代）"] = int(w["stat_philo_birth_reserved"][ph_i])
        row[f"{ph_label} 実出生（体/世代）"] = int(w["stat_philo_birth_real"][ph_i])
        row[f"{ph_label} 死亡（体/世代）"] = int(w["stat_philo_death"][ph_i])
        row[f"{ph_label} 採取獲得（単位/世代）"] = int(w["stat_philo_gather_gain"][ph_i])
        row[f"{ph_label} 移動支払（単位/世代）"] = int(w["stat_philo_move_cost"][ph_i])
        row[f"{ph_label} 維持支払（単位/世代）"] = int(w["stat_philo_upkeep_cost"][ph_i])
        row[f"{ph_label} 交尾試行参加（回/世代）"] = int(w["stat_philo_mate_attempt"][ph_i])
        row[f"{ph_label} 交尾成功参加（回/世代）"] = int(w["stat_philo_mate_success"][ph_i])
        row[f"{ph_label} 捕食試行（回/世代）"] = int(w["stat_philo_predation_attempt"][ph_i])
        row[f"{ph_label} 捕食成功（回/世代）"] = int(w["stat_philo_predation_success"][ph_i])
        row[f"{ph_label} 捕食失敗（回/世代）"] = int(w["stat_philo_predation_fail"][ph_i])
        row[f"{ph_label} 捕食獲得（単位/世代）"] = int(w["stat_philo_predation_gain"][ph_i])
        row[f"{ph_label} 戦闘獲得（単位/世代）"] = int(w["stat_philo_battle_gain"][ph_i])
        row[f"{ph_label} 戦闘損失（単位/世代）"] = int(w["stat_philo_battle_cost"][ph_i])
        row[f"{ph_label} 資源収支ネット（単位/世代）"] = (
            int(w["stat_philo_gather_gain"][ph_i])
            + int(w["stat_philo_predation_gain"][ph_i])
            + int(w["stat_philo_battle_gain"][ph_i])
            - int(w["stat_philo_move_cost"][ph_i])
            - int(w["stat_philo_upkeep_cost"][ph_i])
            - int(w["stat_philo_battle_cost"][ph_i])
        )

    # ===== v19/v20：親子遺伝子フロー・行動選択・チーム内分布 =====
    action_counts = np.asarray(w.get("stat_philo_action_counts", np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32)), dtype=np.int32)
    if action_counts.shape != (PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)):
        action_counts = np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32)

    pair_reserved = np.asarray(w.get("stat_philo_pair_reserved", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)
    pair_real = np.asarray(w.get("stat_philo_pair_real", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)
    parent_child_reserved = np.asarray(w.get("stat_philo_parent_to_child_reserved", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)
    parent_child_real = np.asarray(w.get("stat_philo_parent_to_child_real", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)
    source_child_reserved = np.asarray(w.get("stat_philo_source_to_child_reserved", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)
    source_child_real = np.asarray(w.get("stat_philo_source_to_child_real", np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)), dtype=np.int32)

    for ph_i, ph_label in PHILO_LABELS.items():
        ph_i = int(ph_i)
        row[f"{ph_label} 親参加:出生予約（回/世代）"] = int(w["stat_philo_parent_offspring_reserved"][ph_i])
        row[f"{ph_label} 親参加:実出生（回/世代）"] = int(w["stat_philo_parent_offspring_real"][ph_i])
        denom_ph = max(int(philo_counts[ph_i]), 1)
        for act_i, act_label in PHILO_ACTION_LABELS.items():
            act_i = int(act_i)
            cnt_act = int(action_counts[ph_i, act_i])
            row[f"{ph_label} 行動:{act_label}（体/世代）"] = cnt_act
            row[f"{ph_label} 行動率:{act_label}（0-1）"] = float(cnt_act) / float(denom_ph)

        # チーム内でどの型が多いか。赤青差がチーム色の効果なのか、哲学型の偏りなのかを分ける。
        if n > 0:
            red_mask_ph = (team == 0) & (gene_philo == ph_i)
            blue_mask_ph = (team == 1) & (gene_philo == ph_i)
            row[f"赤×{ph_label} 数（体）"] = int(red_mask_ph.sum())
            row[f"青×{ph_label} 数（体）"] = int(blue_mask_ph.sum())
            row[f"赤×{ph_label} 比率（赤内0-1）"] = float(red_mask_ph.sum()) / max(int((team == 0).sum()), 1)
            row[f"青×{ph_label} 比率（青内0-1）"] = float(blue_mask_ph.sum()) / max(int((team == 1).sum()), 1)
        else:
            row[f"赤×{ph_label} 数（体）"] = 0
            row[f"青×{ph_label} 数（体）"] = 0
            row[f"赤×{ph_label} 比率（赤内0-1）"] = 0.0
            row[f"青×{ph_label} 比率（青内0-1）"] = 0.0

    for pi, plab in PHILO_LABELS.items():
        for ci, clab in PHILO_LABELS.items():
            row[f"親→子 予約:{plab}→{clab}（回/世代）"] = int(parent_child_reserved[int(pi), int(ci)])
            row[f"親→子 実出生:{plab}→{clab}（回/世代）"] = int(parent_child_real[int(pi), int(ci)])
            row[f"コピー元→子 予約:{plab}→{clab}（回/世代）"] = int(source_child_reserved[int(pi), int(ci)])
            row[f"コピー元→子 実出生:{plab}→{clab}（回/世代）"] = int(source_child_real[int(pi), int(ci)])

    for pi, plab in PHILO_LABELS.items():
        for ci, clab in PHILO_LABELS.items():
            if int(pi) <= int(ci):
                row[f"親組合せ 予約:{plab}×{clab}（回/世代）"] = int(pair_reserved[int(pi), int(ci)])
                row[f"親組合せ 実出生:{plab}×{clab}（回/世代）"] = int(pair_real[int(pi), int(ci)])

    st.session_state.history.append(row)
    if len(st.session_state.history) > int(max_history_keep):
        st.session_state.history = st.session_state.history[-int(max_history_keep):]

    # 次世代のW計算用に、現在の遺伝子コピー数を保存
    w["prev_contest_counts"] = contest_counts.astype(np.int32)
    w["prev_predation_counts"] = pred_counts.astype(np.int32)
    w["prev_philo_counts"] = philo_counts.astype(np.int32)
    w["prev_pop_count_for_W"] = int(n)

    # 世代内カウンタをリセット
    for k in [
        "stat_gathered","stat_move_intent","stat_move_actual","stat_collision",
        "stat_mate_attempt","stat_mate_success",
        "stat_battle_transfer","stat_battle_upset",
        "stat_res_spawn_cells","stat_res_spawned",
        "stat_move_cost_paid","stat_upkeep_paid",
        "stat_birth_fee_paid","stat_child_resource_given",
        "stat_birth_failed_space",
        "stat_contest_cells","stat_contest_events","stat_contest_hawk_win","stat_contest_hh_events",
        "stat_contest_cost_paid","stat_contest_v_paid",
        "stat_birth_reserved_hawk","stat_birth_reserved_dove",
        "stat_birth_real_hawk","stat_birth_real_dove",
        "stat_death_hawk","stat_death_dove",
        "stat_pop_hawk_before_death","stat_pop_dove_before_death",
        "stat_gain_gather_hawk","stat_gain_gather_dove",
        "stat_gain_contest_hawk","stat_gain_contest_dove",
        "stat_cost_contest_hawk","stat_cost_contest_dove",
        "stat_gain_battle_hawk","stat_gain_battle_dove",
        "stat_cost_battle_hawk","stat_cost_battle_dove",
        "stat_cost_move_hawk","stat_cost_move_dove",
        "stat_cost_upkeep_hawk","stat_cost_upkeep_dove",
        "stat_cost_birthfee_hawk","stat_cost_birthfee_dove",
        "stat_cost_childshare_hawk","stat_cost_childshare_dove",
        "stat_local_resource_spawned","stat_density_spawn_block",
        "stat_birth_density_block","stat_kin_avoided",
        "stat_predation_attempt","stat_predation_success","stat_predation_fail",
        "stat_predation_gain","stat_predation_kill",
    ]:
        w[k] = 0
    w["stat_move_dist_sum"] = 0.0
    for key in PHILO_STAT_KEYS:
        w[key] = np.zeros(PHILO_TYPE_COUNT, dtype=np.int32)
    for key in PHILO_MATRIX_STAT_KEYS:
        w[key] = np.zeros((PHILO_TYPE_COUNT, PHILO_TYPE_COUNT), dtype=np.int32)
    w["stat_philo_action_counts"] = np.zeros((PHILO_TYPE_COUNT, len(PHILO_ACTION_LABELS)), dtype=np.int32)

def phase5_life_death():
    w = st.session_state.world
    ensure_ecology_arrays(w)
    w["evt_death"] = []
    w["deaths"] = 0

    ys = w["ys"]
    xs = w["xs"]
    team = w["team"]
    bag = w["bag"]
    vision = w["vision"]
    strength = w["strength"]
    age = w["age"]
    direction = w["dir"]
    sex = w["sex"]
    gene = w["gene_contest"]
    gene_pred = w["gene_predation"]
    gene_philo = w["gene_philo"]
    uid = w["uid"]
    parent_a = w["parent_a"]
    parent_b = w["parent_b"]
    lineage = w["lineage"]

    n = len(xs)
    if n == 0:
        log_generation()
        st.session_state.gen += 1
        return

    # 維持コスト
    w["stat_upkeep_paid"] += int(upkeep_cost) * int(n)
    w["stat_cost_upkeep_hawk"] += int(int(upkeep_cost) * int((gene == 0).sum()))
    w["stat_cost_upkeep_dove"] += int(int(upkeep_cost) * int((gene == 1).sum()))
    if n > 0 and int(upkeep_cost) > 0:
        w["stat_philo_upkeep_cost"] += (
            np.bincount(gene_philo.astype(np.int32), minlength=PHILO_TYPE_COUNT).astype(np.int32)
            * int(upkeep_cost)
        )
    bag = bag - int(upkeep_cost)

    # 年齢増加
    age = age + 1

    # 死亡条件：資源0以下 or 寿命
    dead = (bag <= 0) | (age >= int(max_age))

    w["stat_pop_hawk_before_death"] = int((gene == 0).sum())
    w["stat_pop_dove_before_death"] = int((gene == 1).sum())
    w["stat_death_hawk"] = int(((gene == 0) & dead).sum())
    w["stat_death_dove"] = int(((gene == 1) & dead).sum())
    w["stat_philo_death"] = np.bincount(gene_philo[dead].astype(np.int32), minlength=PHILO_TYPE_COUNT).astype(np.int32)

    if dead.any():
        for y, x in zip(ys[dead], xs[dead]):
            w["evt_death"].append((int(y), int(x)))

    alive = ~dead
    w["deaths"] = int(dead.sum())
    w["ys"] = ys[alive]
    w["xs"] = xs[alive]
    w["team"] = team[alive]
    w["bag"] = bag[alive]
    w["vision"] = vision[alive]
    w["strength"] = strength[alive]
    w["age"] = age[alive]
    w["dir"] = direction[alive]
    w["sex"] = sex[alive]
    w["gene_contest"] = gene[alive]
    w["gene_predation"] = gene_pred[alive]
    w["gene_philo"] = gene_philo[alive]
    w["uid"] = uid[alive]
    w["parent_a"] = parent_a[alive]
    w["parent_b"] = parent_b[alive]
    w["lineage"] = lineage[alive]

    log_generation()
    st.session_state.gen += 1

def gini(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    if len(x) == 0:
        return 0.0
    x = np.sort(np.maximum(x, 0))
    s = float(x.sum())
    if s <= 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1, dtype=np.float32)
    return float((2 * (idx * x).sum()) / (n * s) - (n + 1) / n)
# -------------------------
# 進行
# -------------------------
def advance_one_phase():
    p = int(st.session_state.phase)

    if p == 0:
        phase1_spawn_and_birth()
    elif p == 1:
        phase2_perception()
    elif p == 2:
        phase3_thinking()
    elif p == 3:
        phase4_action()
    elif p == 4:
        phase5_life_death()

    # 「いま実行したステップ」を記録
    st.session_state.last_phase_executed = p

    # 次のステップへ
    st.session_state.phase = (p + 1) % 5


def advance_one_generation():
    """
    ①→⑤を内部で一気に進める。
    画面描画は世代末だけにするため、自動実行時の暗転・カクつきを減らす。
    途中の phase から始まっても、まず次の世代末まで進める。
    """
    steps_to_generation_end = 5 - int(st.session_state.phase)
    if steps_to_generation_end <= 0:
        steps_to_generation_end = 5

    for _ in range(int(steps_to_generation_end)):
        advance_one_phase()


def advance_generations(k: int):
    for _ in range(max(1, int(k))):
        advance_one_generation()


if step_btn:
    advance_one_phase()
    st.rerun()

if step_gen_btn:
    advance_one_generation()
    st.rerun()

if step_10_btn:
    advance_generations(10)
    st.rerun()

if step_50_btn:
    advance_generations(50)
    st.rerun()

_manual_pressed = bool(step_btn or step_gen_btn or step_10_btn or step_50_btn or reset_btn)
if running and (not _manual_pressed):
    if smooth_auto_run:
        advance_generations(int(auto_generations_per_refresh))
    else:
        advance_one_phase()

    if int(speed_ms) > 0:
        time.sleep(float(speed_ms) / 1000.0)
    st.rerun()

# -------------------------
# 描画（凡例は出さない）
# -------------------------
w = st.session_state.world
biome_id = w["biome_id"]
resource = w["resource"]
ys, xs = w["ys"], w["xs"]
team = w["team"]
bag = w["bag"]
vision = w["vision"]
strength = w["strength"]

cells = H * W
n_agents = int(len(xs))
res_total = int(resource.sum())
res_cells = int((resource > 0).sum())

biome_colors, _, _ = get_biome_palette(int(biome_k))

def render_base():
    img = biome_colors[biome_id].copy()

    if show_biome_edges:
        e = biome_edges(biome_id)
        img[e] = img[e] * 0.15 + np.array([0.15, 0.15, 0.15], dtype=np.float32) * 0.85

    if show_resource:
        res_color = np.array([0.00, 1.00, 0.65], dtype=np.float32)  # ミント（鮮やか）
        strength01 = np.clip(resource.astype(np.float32) / max(int(res_max), 1), 0.0, 1.0) * float(resource_alpha)
        img = img * (1.0 - strength01[..., None]) + res_color * strength01[..., None]

    if show_agents and n_agents > 0:
        agent_colors = np.array([
            [1.00, 0.20, 0.20],  # 赤
            [0.20, 0.55, 1.00],  # 青
        ], dtype=np.float32)
        for y, x, t in zip(ys, xs, team):
            img[y, x] = img[y, x] * 0.10 + agent_colors[int(t)] * 0.90

    return img

def upscale_with_grid(img01):
    img_u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img_u8).resize((W * scale, H * scale), resample=Image.NEAREST)
    big = np.array(pil)
    for k in range(int(grid_thickness)):
        big[k::scale, :, :] = int(grid_line)
        big[:, k::scale, :] = int(grid_line)
    return big

def overlay_perception(img01):
    cnt = w.get("perception_count", None)
    if cnt is None or cnt.max() <= 0:
        return img01
    mx = float(cnt.max())
    a = np.clip(cnt.astype(np.float32) / mx, 0, 1)
    out = img01.copy()
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    alpha = a[..., None] * 0.45
    out = out * (1 - alpha) + yellow * alpha
    return out

def draw_thinking(big_rgb, prev_y, prev_x, to_y, to_x, act):
    img = Image.fromarray(big_rgb)
    dr = ImageDraw.Draw(img)
    purple = (190, 0, 255)

    for y0, x0, y1, x1, a in zip(prev_y, prev_x, to_y, to_x, act):
        y0 = int(y0); x0 = int(x0); y1 = int(y1); x1 = int(x1)
        cx0 = x0 * scale + scale // 2
        cy0 = y0 * scale + scale // 2
        cx1 = x1 * scale + scale // 2
        cy1 = y1 * scale + scale // 2

        # 矢印（移動先が違うとき）
        if (x0, y0) != (x1, y1):
            dr.line((cx0, cy0, cx1, cy1), fill=purple, width=2)
            head = 4
            dr.polygon([(cx1, cy1), (cx1 - head, cy1 - head), (cx1 - head, cy1 + head)], fill=purple)

        # マーク（行動別）
        if int(a) == 0:  # 待機：□
            s = 4
            dr.rectangle((cx0 - s, cy0 - s, cx0 + s, cy0 + s), outline=purple, width=2)

        elif int(a) == 2:  # 採取：○
            r = 5
            dr.ellipse((cx1 - r, cy1 - r, cx1 + r, cy1 + r), outline=purple, width=2)

        elif int(a) == 3:  # 戦闘：×
            s = 6
            dr.line((cx1 - s, cy1 - s, cx1 + s, cy1 + s), fill=purple, width=2)
            dr.line((cx1 - s, cy1 + s, cx1 + s, cy1 - s), fill=purple, width=2)

        elif int(a) == 4:  # 回避：△
            s = 7
            dr.polygon([(cx1, cy1 - s), (cx1 - s, cy1 + s), (cx1 + s, cy1 + s)],
                       outline=purple, width=2)
        elif int(a) == 5:  # 交尾：◇
            s = 7
            dr.polygon([(cx1, cy1 - s), (cx1 - s, cy1), (cx1, cy1 + s), (cx1 + s, cy1)],
                       outline=purple, width=2)

        elif int(a) == 6:  # 捕食：星っぽい印
            s = 6
            dr.line((cx1 - s, cy1, cx1 + s, cy1), fill=purple, width=2)
            dr.line((cx1, cy1 - s, cx1, cy1 + s), fill=purple, width=2)
            dr.line((cx1 - s, cy1 - s, cx1 + s, cy1 + s), fill=purple, width=1)
            dr.line((cx1 - s, cy1 + s, cx1 + s, cy1 - s), fill=purple, width=1)
        elif int(a) == 6:  # 捕食：二重丸
            r = 7
            dr.ellipse((cx1 - r, cy1 - r, cx1 + r, cy1 + r), outline=purple, width=2)
            dr.ellipse((cx1 - 3, cy1 - 3, cx1 + 3, cy1 + 3), outline=purple, width=2)


    return np.array(img)

base = render_base()

# 状況（単位つき）
st.subheader("状況")
lp = st.session_state.last_phase_executed
step_label = PHASES[int(lp)] if lp is not None else "（まだ未実行）"
st.caption(f"世代：{st.session_state.gen}（回） / ステップ：{step_label}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("個体数（体）", n_agents)
c2.metric("資源総量（単位）", res_total)
c3.metric("資源マス数（マス）", res_cells)
c4.metric("平均所持資源（単位/体）", f"{(float(bag.mean()) if n_agents else 0.0):.2f}")
c5.metric("平均認識半径（マス）", f"{(float(vision.mean()) if n_agents else 0.0):.2f}")
c6.metric("平均肉体強度（値/体）", f"{(float(strength.mean()) if n_agents else 0.0):.2f}")

# ---- 状況の簡易読み取り ----
if show_quick_interpretation:
    notes = []
    if n_agents <= 0:
        notes.append("個体群は絶滅しています。資源・出生条件・維持コストの再調整が必要です。")
    elif n_agents < 20:
        notes.append("個体数がかなり少ないため、遺伝子頻度は偶然の影響を強く受けます。")

    if len(st.session_state.history) > 0:
        _last = st.session_state.history[-1]
        _births = int(_last.get("出生数（体/世代）", 0))
        _deaths = int(_last.get("死亡数（体/世代）", 0))
        _wpop = float(_last.get("個体群全体W（増殖率）", 1.0))
        if _wpop < 0.95:
            notes.append(f"個体群W={_wpop:.3f}。この世代は縮小圧が強めです。")
        elif _wpop > 1.05:
            notes.append(f"個体群W={_wpop:.3f}。この世代は増殖圧が強めです。")
        else:
            notes.append(f"個体群W={_wpop:.3f}。この世代はほぼ横ばいです。")

        if _births < _deaths:
            notes.append(f"出生{_births} < 死亡{_deaths}。長く続くと淘汰以前に個体群が縮みます。")
        elif _births > _deaths:
            notes.append(f"出生{_births} > 死亡{_deaths}。個体群は回復・拡大しやすい状態です。")

    if res_cells > 0 and n_agents > 0:
        _res_ratio = res_cells / max(cells, 1)
        if _res_ratio > 0.35 and float(bag.mean()) < 3.0:
            notes.append("資源マスは多いのに平均所持資源が低いです。探索・採取・移動コストの結合に問題がある可能性があります。")

    if len(notes) > 0:
        st.markdown(
            "<div class='neo-card neo-alert'><div class='neo-title'>現在の読み取り</div>"
            + "".join([f"<div class='neo-soft'>・{x}</div>" for x in notes[:4]])
            + "</div>",
            unsafe_allow_html=True,
        )

# ===== 凡例UI（タイトル/状況の直下に表示） =====
def _swatch(rgb, title, desc=""):
    r, g, b = [int(x) for x in rgb]
    box = f"""
    <span style="
      display:inline-block;width:14px;height:14px;border-radius:4px;
      background:rgb({r},{g},{b});
      border:1px solid rgba(255,255,255,.25);
      margin-right:8px; flex:0 0 auto;"></span>
    """
    return f"""
    <div style="display:flex;align-items:flex-start;gap:6px;margin:4px 0;line-height:1.35;">
      {box}
      <div>
        <div style="font-weight:600; margin-top:-1px;">{title}</div>
        <div style="opacity:.75; font-size:0.92em;">{desc}</div>
      </div>
    </div>
    """

# biome_colors はこの時点で定義済み（あなたのコードどおり）
biome_rgb = [(biome_colors[i] * 255).astype(int).tolist() for i in range(len(biome_colors))]

with st.expander("凡例（色・矢印）", expanded=False):
    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("#### 矢印（紫）・マークの意味")
        st.markdown(
            "- **紫矢印**：位置の変化\n"
            "  - **③思考** =「意図」 / **④行動** =「確定」\n"
            "- **マーク**（行動の種類）\n"
            "  - □ 待機 ○ 採取 × 戦闘 △ 回避 ◇ 交尾\n"
            "  - ※出生は **④で相互に交尾を選択＆隣接** → 次世代①で発生"
        )

    with colB:
        st.markdown("#### ピクセル色（重なり）の意味")

        # 地形（バイオーム）
        # ここはあなたの色定義に合わせて説明（不毛→豊穣の傾向）
        for i, rgb in enumerate(biome_rgb):
            if i == 0:
                desc = "地形（不毛寄り）"
            else:
                desc = "地形（より豊穣寄り）"
            st.markdown(_swatch(rgb, f"バイオーム {i}", desc), unsafe_allow_html=True)

        st.markdown("---")

        # 資源・個体・認識ヒートなど（あなたの描画順に対応）
        st.markdown(_swatch((0, 255, 166), "資源（ミント）", "濃いほど資源量が多い（res_max と resource_alpha に依存）"), unsafe_allow_html=True)
        st.markdown(_swatch((255, 51, 51), "個体：赤（team 0）", "セル上の個体（赤チーム）"), unsafe_allow_html=True)
        st.markdown(_swatch((51, 140, 255), "個体：青（team 1）", "セル上の個体（青チーム）"), unsafe_allow_html=True)
        st.markdown(_swatch((255, 255, 0), "②認識ヒート（黄）", "見られた回数が多いほど黄色が強い"), unsafe_allow_html=True)

        if show_biome_edges:
            st.markdown(_swatch((40, 40, 40), "バイオーム境界（強調ON時）", "境界が暗く強調される"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "**重なり順（ざっくり）**：地形 →（境界）→ 資源 → 個体 → 認識ヒート →（③/④の紫矢印・マーク）"
        )

# -------------------------
# 表示タブ（ステップ追従で自動切替）
# -------------------------
VIEW_TABS = ["環境", "①発生", "②認識", "③思考", "④行動", "⑤生死", "統計"]
PHASE_TO_VIEW = {0: "①発生", 1: "②認識", 2: "③思考", 3: "④行動", 4: "⑤生死"}

if "follow_phase" not in st.session_state:
    st.session_state.follow_phase = True
if "view_tab" not in st.session_state:
    st.session_state.view_tab = "環境"

# ステップに追従：phase に合わせて「選択中タブ」を強制更新
# 自動実行中は環境ビューに固定すると、重い統計/矢印描画への切替が減り、テンポが良くなる。
if bool(running) and bool(smooth_auto_run) and bool(auto_lock_environment_view):
    st.session_state.view_tab = "環境"
elif st.session_state.follow_phase:
    lp = st.session_state.last_phase_executed
    st.session_state.view_tab = PHASE_TO_VIEW[int(lp)] if lp is not None else "環境"

# UI（タブっぽい横並び）
left, right = st.columns([1, 8])
with left:
    st.session_state.follow_phase = st.checkbox("追従", value=st.session_state.follow_phase)
with right:
    view = st.radio(
        label="",
        options=VIEW_TABS,
        horizontal=True,
        key="view_tab",
        label_visibility="collapsed",
    )

# -------------------------
# 各ビュー表示（元の tabs 内容をそのまま移植）
# -------------------------
if view == "環境":
    st.image(upscale_with_grid(base), use_container_width=True)

elif view == "①発生":
    big = upscale_with_grid(base)
    pil = Image.fromarray(big)
    dr = ImageDraw.Draw(pil)

    # 出生：緑の＋
    for (y, x) in w.get("evt_birth", []):
        cx = int(x) * scale + scale // 2
        cy = int(y) * scale + scale // 2
        s = max(4, scale // 4)
        dr.line((cx - s, cy, cx + s, cy), fill=(0, 220, 0), width=2)
        dr.line((cx, cy - s, cx, cy + s), fill=(0, 220, 0), width=2)

    # 資源発生：緑の点（薄く）
    spawn = w.get("evt_res_spawn", None)
    if spawn is not None and spawn.any():
        ys_s, xs_s = np.where(spawn)
        step = max(1, len(ys_s) // 1200)
        for y, x in zip(ys_s[::step], xs_s[::step]):
            cx = int(x) * scale + scale // 2
            cy = int(y) * scale + scale // 2
            dr.point((cx, cy), fill=(0, 220, 0))

    st.image(np.array(pil), use_container_width=True)

elif view == "②認識":
    img = base
    if show_perception:
        img = overlay_perception(img)
    st.image(upscale_with_grid(img), use_container_width=True)

elif view == "③思考":
    big = upscale_with_grid(base)
    if n_agents > 0 and show_thinking:
        big2 = draw_thinking(
            big,
            ys, xs,
            w.get("intent_y", ys),
            w.get("intent_x", xs),
            w.get("intent_act", np.zeros(n_agents, dtype=np.int8))
        )
        st.image(big2, use_container_width=True)
    else:
        st.image(big, use_container_width=True)

elif view == "④行動":
    big = upscale_with_grid(base)
    prev_y, prev_x = w.get("last_prev", (ys, xs))
    act = w.get("last_act", np.zeros(n_agents, dtype=np.int8))
    if len(prev_x) == n_agents and n_agents > 0:
        big2 = draw_thinking(big, prev_y, prev_x, ys, xs, act)
        st.image(big2, use_container_width=True)
    else:
        st.image(big, use_container_width=True)

elif view == "⑤生死":
    big = upscale_with_grid(base)
    pil = Image.fromarray(big)
    dr = ImageDraw.Draw(pil)

    # 死亡：赤×
    for (y, x) in w.get("evt_death", []):
        cx = int(x) * scale + scale // 2
        cy = int(y) * scale + scale // 2
        s = max(5, scale // 4)
        dr.line((cx - s, cy - s, cx + s, cy + s), fill=(255, 0, 0), width=2)
        dr.line((cx - s, cy + s, cx + s, cy - s), fill=(255, 0, 0), width=2)

    st.image(np.array(pil), use_container_width=True)

elif view == "統計":
    st.subheader("統計（世代推移）")

    df = pd.DataFrame(st.session_state.history)

    if len(df) == 0:
        st.info("まだ統計がありません。『1世代（①→⑤）進める』を押してください。")
        st.stop()

    # ---- 表示範囲（直近N世代） ----
    max_n = int(len(df))
    if max_n == 1:
        last_n = 1
        st.info("統計が1世代分だけなので、グラフは次の世代から出ます。")
    else:
        min_n = 1
        default_n = min(120, max_n)
        last_n = st.slider(
            "表示する直近世代数（横軸：世代（回））",
            min_value=min_n,
            max_value=max_n,
            value=default_n,
            step=1,
            key="last_n_stats",
        )

    dff = df.tail(int(last_n)).copy()

    # ---- CSVエクスポート ----
    st.download_button(
        label="統計CSVをダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="neo_life_stats.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ---- 最新世代のサマリー ----
    last = df.iloc[-1].to_dict()
    cA, cB, cC, cD, cE, cF = st.columns(6)
    cA.metric("世代（回）", int(last.get("世代（回）", 0)))
    cB.metric("個体数（体）", int(last.get("個体数（体）", 0)))
    cC.metric("資源総量（単位）", int(last.get("資源総量（単位）", 0)))
    cD.metric("タカ比率（0-1）", f"{float(last.get('タカ比率（0-1）', 0.0)):.3f}")
    cE.metric("出生数（体/世代）", int(last.get("出生数（体/世代）", 0)))
    cF.metric("死亡数（体/世代）", int(last.get("死亡数（体/世代）", 0)))

    st.caption("※グラフは基本 Altair（軸に単位が書ける）を優先。Altairが無い環境では簡易グラフになります。")

    # -------------------------
    # Altair用の共通関数（単位付き軸）
    # -------------------------
    def plot_lines(title: str, xcol: str, cols: list, y_title: str):
        cols = [c for c in cols if c in dff.columns]
        if len(cols) == 0:
            st.warning(f"『{title}』に使う列が見つかりませんでした。")
            return

        if HAS_ALTAIR and len(dff) >= 2:
            data = dff[[xcol] + cols].melt(id_vars=[xcol], var_name="指標", value_name="値")
            ch = (
                alt.Chart(data)
                .mark_line()
                .encode(
                    x=alt.X(f"{xcol}:Q", title="世代（回）"),
                    y=alt.Y("値:Q", title=y_title),
                    color=alt.Color("指標:N", title="指標"),
                    tooltip=[alt.Tooltip(f"{xcol}:Q", title="世代（回）"), "指標:N", alt.Tooltip("値:Q", title=y_title)]
                )
                .properties(title=title, height=210)
                .interactive()
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption(f"横軸：世代（回） / 縦軸：{y_title}")
            st.line_chart(dff.set_index(xcol)[cols])

    def plot_area(title: str, xcol: str, col: str, y_title: str):
        if col not in dff.columns:
            st.warning(f"『{title}』に使う列 {col} が見つかりません。")
            return

        if HAS_ALTAIR and len(dff) >= 2:
            data = dff[[xcol, col]].copy()
            ch = (
                alt.Chart(data)
                .mark_area(opacity=0.7)
                .encode(
                    x=alt.X(f"{xcol}:Q", title="世代（回）"),
                    y=alt.Y(f"{col}:Q", title=y_title),
                    tooltip=[alt.Tooltip(f"{xcol}:Q", title="世代（回）"), alt.Tooltip(f"{col}:Q", title=y_title)]
                )
                .properties(title=title, height=210)
                .interactive()
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption(f"横軸：世代（回） / 縦軸：{y_title}")
            st.line_chart(dff.set_index(xcol)[[col]])

    def plot_hist(title: str, values: np.ndarray, x_title: str):
        values = np.asarray(values)
        if values.size == 0:
            st.info(f"{title}：データがありません。")
            return

        # 小さい値でも見えるように最低1bin確保
        vmin = float(values.min())
        vmax = float(values.max())
        if vmin == vmax:
            st.info(f"{title}：値がすべて同じです（{vmin}）。")
            return

        if HAS_ALTAIR:
            bins = max(10, int(np.sqrt(len(values))))
            hist = pd.DataFrame({ "値": values })
            ch = (
                alt.Chart(hist)
                .mark_bar()
                .encode(
                    x=alt.X("値:Q", bin=alt.Bin(maxbins=bins), title=x_title),
                    y=alt.Y("count():Q", title="個体数（体）"),
                    tooltip=[alt.Tooltip("count():Q", title="個体数（体）")]
                )
                .properties(title=title, height=200)
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            # fallback: ヒストが弱いので、要約だけ出す
            st.caption(f"{title}（{x_title}）")
            st.write({
                "min": int(values.min()),
                "p10": float(np.percentile(values, 10)),
                "median": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "max": int(values.max()),
            })

    def explain_box(title: str, body: str):
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.16); border-radius:12px; padding:12px 14px; margin:8px 0; background:rgba(255,255,255,0.035);">
              <div style="font-weight:700; margin-bottom:4px;">{title}</div>
              <div style="opacity:.86; line-height:1.55; font-size:0.94em;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _latest_num(col: str, default: float = 0.0) -> float:
        try:
            v = last.get(col, default)
            if pd.isna(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def _trend_delta(col: str) -> float:
        if col not in dff.columns or len(dff) < 2:
            return 0.0
        try:
            a = float(dff[col].iloc[0])
            b = float(dff[col].iloc[-1])
            return b - a
        except Exception:
            return 0.0

    def _fmt_delta(x: float, digits: int = 2) -> str:
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.{digits}f}"

    def plot_stacked_area(title: str, xcol: str, cols: list, y_title: str, explanation: str = ""):
        cols = [c for c in cols if c in dff.columns]
        if len(cols) == 0:
            st.warning(f"『{title}』に使う列が見つかりませんでした。")
            return
        if explanation:
            st.caption(explanation)

        if HAS_ALTAIR and len(dff) >= 2:
            data = dff[[xcol] + cols].melt(id_vars=[xcol], var_name="遺伝子/指標", value_name="値")
            ch = (
                alt.Chart(data)
                .mark_area(opacity=0.78)
                .encode(
                    x=alt.X(f"{xcol}:Q", title="世代（回）"),
                    y=alt.Y("値:Q", title=y_title, stack="zero"),
                    color=alt.Color("遺伝子/指標:N", title="遺伝子/指標"),
                    tooltip=[
                        alt.Tooltip(f"{xcol}:Q", title="世代（回）"),
                        alt.Tooltip("遺伝子/指標:N", title="遺伝子/指標"),
                        alt.Tooltip("値:Q", title=y_title, format=".4f"),
                    ],
                )
                .properties(title=title, height=260)
                .interactive()
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            st.area_chart(dff.set_index(xcol)[cols])

    def plot_latest_bar(title: str, cols: list, y_title: str, explanation: str = ""):
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            st.warning(f"『{title}』に使う列が見つかりませんでした。")
            return
        if explanation:
            st.caption(explanation)
        vals = []
        for c in cols:
            try:
                vals.append(float(df[c].iloc[-1]))
            except Exception:
                vals.append(0.0)
        data = pd.DataFrame({"指標": cols, "値": vals})
        if HAS_ALTAIR:
            base = alt.Chart(data).encode(
                y=alt.Y("指標:N", sort="-x", title=""),
                x=alt.X("値:Q", title=y_title),
                tooltip=["指標:N", alt.Tooltip("値:Q", title=y_title, format=".4f")],
            )
            bars = base.mark_bar()
            text = base.mark_text(align="left", dx=4).encode(text=alt.Text("値:Q", format=".3f"))
            st.altair_chart((bars + text).properties(title=title, height=max(180, 28 * len(cols))), use_container_width=True)
        else:
            st.bar_chart(data.set_index("指標"))

    def show_reading_guide():
        with st.expander("数値の読み方・この研究での意味", expanded=True):
            st.markdown("""
### まず見る順番
1. **個体群全体W**を見る。`1`前後なら個体群が維持、`1`を大きく下回るなら自然淘汰以前に生態系が崩壊方向です。  
2. **哲学遺伝子W**を見る。各思想型のコピーが前世代より増えたか減ったかを見ます。  
3. **頻度**を見る。頻度は相対値なので、個体群が減っているときは「勝っているように見えるだけ」のことがあります。  
4. **資源フロー**を見る。採取・捕食・戦闘獲得が、維持・移動・繁殖投資を上回っているかを確認します。  
5. **出生/死亡**を見る。遺伝子が残るには、最終的に出生へつながる必要があります。

### 重要語
**W（コピー増殖率）**：遺伝子を見るための最重要値です。`W > 1` なら前世代よりその遺伝子コピーが増え、`W = 1` なら維持、`W < 1` なら減少です。個体が強いかではなく、コピーが残ったかを見ます。

**遺伝子頻度（0-1）**：集団内でその遺伝子型が占める割合です。頻度が上がっていても、全体個体数が激減している場合は「崩壊中に相対的に残った」だけかもしれません。Wと個体群全体Wをセットで見ます。

**Simpson多様度**：0に近いほど単一遺伝子に偏り、1に近いほど多様です。急落する場合は、特定遺伝子が固定に向かっているか、個体群が小さくなりすぎています。

**Gini（資源格差）**：0に近いほど平等、1に近いほど一部個体に資源が集中しています。高すぎると、集団全体では資源があっても繁殖できる個体が限られます。

**資源ストックと資源フロー**：ストックは盤面に残っている資源量、フローは世代あたりの発生・採取・消費です。資源が増えているのに個体が減るなら、探索・移動・採取・繁殖接続が失敗しています。

**局所密度**：周辺にいる個体数です。高いほど競争・衝突・出生場所不足が起こりやすく、低すぎると交尾相手探索が失敗しやすくなります。

**哲学遺伝子**：思想家本人の完全再現ではなく、哲学的方針を行動評価関数に変換した遺伝子です。自然淘汰で問うのは「その行動方針がどの環境圧でコピーを残すか」です。
""")

        with st.expander("指標ごとの診断ポイント", expanded=False):
            guide_rows = [
                {"指標": "個体群全体W", "意味": "集団全体が増えているか維持しているか", "見方": "1付近が安定。0.9未満が続くと崩壊圧、1.1超が続くと増殖圧が強い"},
                {"指標": "哲学遺伝子W", "意味": "各思想型のコピー増殖率", "見方": "1超なら増加、1未満なら減少。頻度より優先して見る"},
                {"指標": "哲学遺伝子頻度", "意味": "集団内での割合", "見方": "増えていても個体群が減っていれば、真の適応とは限らない"},
                {"指標": "資源収支ネット", "意味": "獲得資源 − 支払資源", "見方": "正なら繁殖余剰を作りやすい。負が続く型は長期的に不利"},
                {"指標": "出生予約/実出生", "意味": "繁殖行動が子の発生まで届いたか", "見方": "予約が多く実出生が少ないなら、出生場所不足や密度依存が強い"},
                {"指標": "空腹個体比率", "意味": "資源不足個体の割合", "見方": "高い型は探索・採取・コスト面で不利な可能性"},
                {"指標": "捕食成功率", "意味": "捕食が資源戦略として成立しているか", "見方": "高すぎると被食圧で生態系が崩れ、低すぎると捕食型はコスト負けする"},
                {"指標": "Gini", "意味": "資源の偏り", "見方": "高いほど一部個体だけが繁殖できる。自然淘汰が強くなるが絶滅リスクも上がる"},
                {"指標": "Simpson多様度", "意味": "遺伝子型の多様さ", "見方": "低下は固定化かボトルネック。高すぎる場合は淘汰圧が弱い可能性"},
            ]
            st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)

    def show_philo_summary_table():
        rows = []
        for lab in PHILO_LABELS.values():
            rows.append({
                "哲学型": lab,
                "比率": _latest_num(f"{lab} 比率（0-1）"),
                "W": _latest_num(f"{lab} W"),
                "資源収支ネット": _latest_num(f"{lab} 資源収支ネット（単位/世代）"),
                "平均所持資源": _latest_num(f"{lab} 平均所持資源（単位/体）"),
                "空腹比率": _latest_num(f"{lab} 空腹個体比率（0-1）"),
                "実出生": _latest_num(f"{lab} 実出生（体/世代）"),
                "死亡": _latest_num(f"{lab} 死亡（体/世代）"),
            })
        tab = pd.DataFrame(rows)
        if len(tab) > 0:
            st.markdown("#### 最新世代：哲学遺伝子サマリー")
            st.caption("W・資源収支・出生/死亡を横に並べると、『なぜその遺伝子が増減したか』を読みやすくなります。")
            st.dataframe(
                tab.style.format({
                    "比率": "{:.3f}",
                    "W": "{:.3f}",
                    "資源収支ネット": "{:.1f}",
                    "平均所持資源": "{:.2f}",
                    "空腹比率": "{:.3f}",
                    "実出生": "{:.0f}",
                    "死亡": "{:.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    def show_auto_interpretation():
        """
        最新状態・直近10/50世代・全期間の変化を組み合わせて、
        「何が起きたか」だけでなく「なぜそう読めるか」まで文章化する。
        """
        def _has(col: str) -> bool:
            return col in df.columns

        def _num(col, default=0.0):
            return _latest_num(col, default)

        def _s(col):
            if col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce")

        def _recent_mean(col, k=10, default=0.0):
            s = _s(col).dropna()
            if len(s) == 0:
                return float(default)
            return float(s.tail(min(k, len(s))).mean())

        def _recent_delta(col, k=10, default=0.0):
            s = _s(col).dropna()
            if len(s) < 2:
                return float(default)
            block = s.tail(min(k, len(s)))
            if len(block) < 2:
                return float(default)
            return float(block.iloc[-1] - block.iloc[0])

        def _first(col, default=0.0):
            s = _s(col).dropna()
            return float(s.iloc[0]) if len(s) else float(default)

        def _mean_all(col, default=0.0):
            s = _s(col).dropna()
            return float(s.mean()) if len(s) else float(default)

        def _direction(delta: float, eps: float = 1e-9) -> str:
            if delta > eps:
                return "増加"
            if delta < -eps:
                return "減少"
            return "横ばい"

        def _why_join(parts):
            parts = [str(p) for p in parts if str(p).strip()]
            return " / ".join(parts) if parts else "根拠列が不足しているため、比率とW中心の読み取りです。"

        full_n = int(len(df))
        recent10 = int(min(10, full_n))
        recent50 = int(min(50, full_n))

        pop_now = _num("個体数（体）")
        pop_first = _first("個体数（体）", pop_now)
        pop_delta10 = _recent_delta("個体数（体）", recent10)
        pop_delta50 = _recent_delta("個体数（体）", recent50)
        pop_delta_all = pop_now - pop_first
        W_pop_now = _num("個体群全体W（増殖率）", np.nan)
        if np.isnan(W_pop_now):
            W_pop_now = _num("個体群全体W（増殖率)", 0.0)
        W_pop_10 = _recent_mean("個体群全体W（増殖率）", recent10, W_pop_now)
        W_pop_50 = _recent_mean("個体群全体W（増殖率）", recent50, W_pop_now)

        births_now = _num("出生数（体/世代）")
        deaths_now = _num("死亡数（体/世代）")
        births_10 = _recent_mean("出生数（体/世代）", recent10)
        deaths_10 = _recent_mean("死亡数（体/世代）", recent10)
        births_50 = _recent_mean("出生数（体/世代）", recent50)
        deaths_50 = _recent_mean("死亡数（体/世代）", recent50)
        bd_now = births_now / max(deaths_now, 1.0)
        bd_10 = births_10 / max(deaths_10, 1e-9)
        bd_50 = births_50 / max(deaths_50, 1e-9)

        res_now = _num("資源総量（単位）")
        res_first = _first("資源総量（単位）", res_now)
        res_delta10 = _recent_delta("資源総量（単位）", recent10)
        res_delta50 = _recent_delta("資源総量（単位）", recent50)
        mean_bag_now = _num("平均所持資源（単位/体）")
        mean_bag_delta10 = _recent_delta("平均所持資源（単位/体）", recent10)
        gather_10 = _recent_mean("採取総量（単位/世代）", recent10)
        spawn_10 = _recent_mean("資源自然発生総量（単位/世代）", recent10)
        upkeep_10 = _recent_mean("維持コスト支払総量（単位/世代）", recent10)
        move_10 = _recent_mean("移動コスト支払総量（単位/世代）", recent10)
        birth_cost_10 = _recent_mean("出生コスト支払総量（単位/世代）", recent10)
        child_share_10 = _recent_mean("子へ分配した資源（単位/世代）", recent10)
        energetic_balance_10 = gather_10 - upkeep_10 - move_10 - birth_cost_10 - child_share_10
        use_ratio_10 = gather_10 / max(spawn_10, 1e-9)

        mate_attempt_10 = _recent_mean("交尾試行（回/世代）", recent10)
        mate_success_10 = _recent_mean("交尾成立（回/世代）", recent10)
        mate_rate_10 = mate_success_10 / max(mate_attempt_10, 1e-9)
        pred_attempt_10 = _recent_mean("捕食試行（回/世代）", recent10)
        pred_success_10 = _recent_mean("捕食成功（回/世代）", recent10)
        pred_rate_10 = pred_success_10 / max(pred_attempt_10, 1e-9)
        battle_10 = _recent_mean("戦闘回数（回/世代）", recent10)
        contest_10 = _recent_mean("争奪イベント数（回/世代）", recent10)

        gini_now = _num("資源格差Gini（0-1）")
        gini_delta10 = _recent_delta("資源格差Gini（0-1）", recent10)
        philo_div = _num("哲学遺伝子多様度（Simpson）")
        philo_div_delta = philo_div - _first("哲学遺伝子多様度（Simpson）", philo_div)
        density_now = _num("平均局所密度（体/近傍）")
        density_block_10 = _recent_mean("過密で抑制された出生候補（回/世代）", recent10)
        kin_avoid_10 = _recent_mean("近親交配回避（回/世代）", recent10)

        # --- 1. 個体群レベル：なぜ増減しているか ---
        eco_notes = []
        eco_why = []
        if W_pop_10 < 0.97:
            eco_notes.append(f"直近{recent10}世代の平均Wは **{W_pop_10:.3f}** で、個体群には縮小圧があります。")
            eco_why.append("なぜなら、Wは前世代に対するコピー数比なので、1を下回る状態が続くほど出生より死亡・消失が優勢になりやすいからです。")
        elif W_pop_10 > 1.03:
            eco_notes.append(f"直近{recent10}世代の平均Wは **{W_pop_10:.3f}** で、個体群には増殖圧があります。")
            eco_why.append("なぜなら、Wが1を超える状態は、世代更新後の個体コピー数が前世代より増えていることを意味するからです。")
        else:
            eco_notes.append(f"直近{recent10}世代の平均Wは **{W_pop_10:.3f}** で、個体群は維持線付近です。")
            eco_why.append("なぜなら、Wが1付近にあると、全体個体数の増減が小さくなり、哲学遺伝子どうしの相対差を観察しやすくなるからです。")

        if bd_10 < 0.90:
            eco_notes.append(f"直近{recent10}世代では出生/死亡比が **{bd_10:.2f}** で、死亡が出生を上回っています。")
            eco_why.append("これは、コピーを増やす入口である出生が、コピーを失う出口である死亡を補えていないという意味です。")
        elif bd_10 > 1.10:
            eco_notes.append(f"直近{recent10}世代では出生/死亡比が **{bd_10:.2f}** で、出生が死亡を上回っています。")
            eco_why.append("これは、個体群が単なる生存だけでなく、遺伝子コピーを新個体へ送る段階まで到達しているという意味です。")
        else:
            eco_notes.append(f"直近{recent10}世代では出生/死亡比が **{bd_10:.2f}** で、出生と死亡はかなり近いです。")
            eco_why.append("この場合、集団全体はほぼ均衡し、小さな行動評価差が哲学遺伝子頻度の差として現れやすくなります。")

        if abs(pop_delta10) >= max(5, pop_now * 0.05):
            eco_notes.append(f"直近{recent10}世代の個体数変化は **{pop_delta10:+.0f}体** です。")
            if pop_delta10 < 0:
                eco_why.append("個体数が短期で減る場合、最新比率で増えた哲学型も『本当に増えた』のではなく『他型より減りにくかった』だけの可能性があります。")
            else:
                eco_why.append("個体数が短期で増える場合、比率上昇と実数増加が一致している型ほど、コピー増殖として解釈しやすくなります。")

        # --- 2. 資源：なぜ利用できている/できていないか ---
        resource_notes = []
        resource_why = []
        resource_notes.append(f"直近{recent10}世代の資源総量変化は **{res_delta10:+.0f}**、平均所持資源変化は **{mean_bag_delta10:+.2f}** です。")
        resource_why.append("資源総量は環境側のストック、平均所持資源は個体側へ回収された資源です。両者が同じ方向に動くなら環境と個体が接続し、逆方向なら資源配置・認識・移動・採取のどこかで接続が弱いと読めます。")
        resource_notes.append(f"直近{recent10}世代の平均採取は **{gather_10:.1f}**、自然発生は **{spawn_10:.1f}**、採取/発生比は **{use_ratio_10:.2f}** です。")
        if use_ratio_10 < 0.60 and spawn_10 > 0:
            resource_why.append("採取/発生比が低いので、資源は発生しているのに個体が十分に拾えていない可能性があります。これは認識範囲・移動コスト・資源の局所性に原因候補があります。")
        elif use_ratio_10 > 1.20:
            resource_why.append("採取/発生比が高いので、個体が過去から残った資源ストックも回収している可能性があります。ただし長く続くと資源枯渇を招きます。")
        else:
            resource_why.append("採取/発生比が極端ではないため、資源発生と資源利用は比較的つながっています。ここでは資源量そのものより、どの型が資源を持つかが重要です。")

        resource_notes.append(f"採取から主要コストを引いた簡易収支は **{energetic_balance_10:.1f}/世代** です。")
        if energetic_balance_10 < -50:
            resource_why.append("簡易収支が負なので、採取だけでは維持・移動・繁殖投資を払い切れていません。捕食・戦闘獲得・高資源個体の偏りが集団を支えている可能性があります。")
        elif energetic_balance_10 > 50:
            resource_why.append("簡易収支が正なので、環境から個体群へ余剰が入っています。次に見るべきなのは、その余剰が出生へ接続されているかです。")
        else:
            resource_why.append("簡易収支が均衡に近いので、少しの行動差や局所配置の差が、そのまま淘汰差として出やすくなります。")

        if res_delta10 > 0 and pop_delta10 < 0:
            resource_notes.append("資源は増えているのに個体数が減っています。")
            resource_why.append("これは『環境に資源はあるが、個体がそれを使えていない』状態の典型です。探索・採取・移動コスト・配偶者探索のどれかが、資源から出生への経路を切っている可能性があります。")
        elif res_delta10 < 0 and pop_delta10 > 0:
            resource_notes.append("個体数は増えていますが資源は減っています。")
            resource_why.append("これは増殖が資源ストックを先食いしている状態です。後の世代で空腹・死亡・交尾失敗が増える可能性があります。")

        if gini_now > 0.50:
            resource_notes.append(f"資源格差Giniは **{gini_now:.3f}** で高めです。")
            resource_why.append("Giniが高いと、一部個体だけが繁殖資源を持ちやすくなります。自然淘汰は強まりますが、集団全体の出生が一部個体に依存する危険も増えます。")
        elif gini_now < 0.25:
            resource_notes.append(f"資源格差Giniは **{gini_now:.3f}** で低めです。")
            resource_why.append("Giniが低いと資源差による淘汰は弱くなります。思想型の差が出るなら、資源量より行動選択や死亡回避に原因がある可能性が高いです。")
        elif abs(gini_delta10) > 0.03:
            resource_notes.append(f"直近{recent10}世代のGini変化は **{gini_delta10:+.3f}** です。")
            resource_why.append("Giniが短期で動くときは、資源の偏りが変化しているので、繁殖可能個体の偏りも変わっている可能性があります。")

        # --- 3. 行動イベント：なぜ出生/死亡へ接続したか ---
        behavior_notes = []
        behavior_why = []
        behavior_notes.append(f"直近{recent10}世代の交尾試行は平均 **{mate_attempt_10:.1f}**、交尾成立は **{mate_success_10:.1f}**、成立率は **{mate_rate_10:.2f}** です。")
        if mate_attempt_10 < 1 and pop_now > 30:
            behavior_why.append("個体数に対して交尾試行が少ないので、繁殖行動が選ばれていない、相手が認識範囲に入りにくい、または資源不足で交尾より採取・回避が優先されている可能性があります。")
        elif mate_attempt_10 > 0 and mate_rate_10 < 0.25:
            behavior_why.append("交尾試行に対して成立が少ないので、相互選択・距離・空きマス・密度依存・近親回避のどれかが繁殖ボトルネックになっている可能性があります。")
        elif mate_success_10 > 0:
            behavior_why.append("交尾成立が継続しているので、少なくとも配偶者探索から出生予約までは機能しています。次は実出生と子の生存を見る段階です。")

        if density_block_10 > 0:
            behavior_notes.append(f"過密で抑制された出生候補は直近平均 **{density_block_10:.1f}** 回です。")
            behavior_why.append("過密抑制が出ているため、出生数の低下は交尾意欲だけでなく、局所空間の詰まりによっても起きています。")
        if kin_avoid_10 > 0:
            behavior_notes.append(f"近親交配回避は直近平均 **{kin_avoid_10:.1f}** 回です。")
            behavior_why.append("近親回避は遺伝的には妥当な制約ですが、個体数が少ない局面では繁殖機会そのものを削り、Wを下げる原因にもなります。")

        behavior_notes.append(f"捕食試行は平均 **{pred_attempt_10:.1f}**、捕食成功率は **{pred_rate_10:.2f}** です。戦闘回数は **{battle_10:.1f}**、争奪イベントは **{contest_10:.1f}** です。")
        if pred_attempt_10 > 0 and pred_rate_10 < 0.20:
            behavior_why.append("捕食成功率が低いので、捕食は資源獲得ではなくリスクや機会損失になっている可能性があります。捕食傾向の高い哲学型はここで不利になりやすいです。")
        elif pred_attempt_10 > 0 and pred_rate_10 > 0.70:
            behavior_why.append("捕食成功率が高いので、捕食は資源獲得戦略として成立しています。ただし、被食圧が高すぎると集団全体の死亡圧も高まります。")
        if battle_10 > 0 or contest_10 > 0:
            behavior_why.append("戦闘・争奪が観測される場合、資源獲得は採取だけではなく個体間相互作用にも依存しています。したがって『強い型』は資源探索型ではなく、衝突回避または衝突勝利型として残っている可能性があります。")

        # --- 4. 哲学遺伝子：各型の「なぜ」 ---
        philo_rows = []
        for lab in PHILO_LABELS.values():
            ratio = _num(f"{lab} 比率（0-1）")
            count = _num(f"{lab} 数（体）")
            count_first = _first(f"{lab} 数（体）", count)
            ratio_first = _first(f"{lab} 比率（0-1）", ratio)
            W_now = _num(f"{lab} W", np.nan)
            W_recent = _recent_mean(f"{lab} W", recent10, W_now if not np.isnan(W_now) else 0.0)
            W_50 = _recent_mean(f"{lab} W", recent50, W_recent)
            net = _num(f"{lab} 資源収支ネット（単位/世代）")
            net_recent = _recent_mean(f"{lab} 資源収支ネット（単位/世代）", recent10, net)
            avg_bag = _num(f"{lab} 平均所持資源（単位/体）")
            hunger = _num(f"{lab} 空腹個体比率（0-1）")
            born = _num(f"{lab} 実出生（体/世代）")
            dead = _num(f"{lab} 死亡（体/世代）")
            born_10 = _recent_mean(f"{lab} 実出生（体/世代）", recent10, born)
            dead_10 = _recent_mean(f"{lab} 死亡（体/世代）", recent10, dead)
            birth_death_diff_10 = born_10 - dead_10
            pred = _num(f"{lab} 捕食試行（回/世代）")
            mate = _num(f"{lab} 交尾成功参加（回/世代）")
            mate_10_lab = _recent_mean(f"{lab} 交尾成功参加（回/世代）", recent10, mate)
            delta_ratio = ratio - ratio_first
            delta_count = count - count_first

            causal_parts = []
            if W_recent > 1.02:
                causal_parts.append(f"直近Wが1超（{W_recent:.3f}）でコピーが増えやすい")
            elif W_recent < 0.98:
                causal_parts.append(f"直近Wが1未満（{W_recent:.3f}）でコピーが減りやすい")
            else:
                causal_parts.append(f"直近Wが維持線付近（{W_recent:.3f}）")
            if birth_death_diff_10 > 0.5:
                causal_parts.append(f"出生が死亡を上回る（{birth_death_diff_10:+.1f}/世代）")
            elif birth_death_diff_10 < -0.5:
                causal_parts.append(f"死亡が出生を上回る（{birth_death_diff_10:+.1f}/世代）")
            if net_recent > 10:
                causal_parts.append(f"資源収支が正（{net_recent:+.1f}）")
            elif net_recent < -10:
                causal_parts.append(f"資源収支が負（{net_recent:+.1f}）")
            if hunger > 0.40:
                causal_parts.append(f"空腹比率が高い（{hunger:.2f}）")
            elif 0 < hunger < 0.15:
                causal_parts.append(f"空腹比率が低い（{hunger:.2f}）")
            if mate_10_lab > 0:
                causal_parts.append(f"交尾成功参加がある（{mate_10_lab:.1f}/世代）")
            if pred > 0:
                causal_parts.append(f"捕食試行がある（最新{pred:.0f}）")

            if W_recent > 1.02 and delta_ratio > 0.02:
                reading = "増加をかなり説明しやすい"
            elif W_recent > 1.02 and delta_count > 0:
                reading = "実数増加を伴う増加"
            elif W_recent < 0.98 and delta_ratio < -0.02:
                reading = "縮小をかなり説明しやすい"
            elif delta_ratio > 0 and delta_count <= 0:
                reading = "相対的に残った可能性"
            elif net_recent < 0 and hunger > 0.4:
                reading = "資源制約で不利の可能性"
            elif abs(delta_ratio) < 0.02:
                reading = "ほぼ中立・均衡寄り"
            else:
                reading = "複合要因"

            philo_rows.append({
                "哲学型": lab,
                "初期数": count_first,
                "最新数": count,
                "数変化": delta_count,
                "初期比率": ratio_first,
                "最新比率": ratio,
                "比率変化": delta_ratio,
                "最新W": W_now,
                f"直近{recent10}世代W": W_recent,
                f"直近{recent50}世代W": W_50,
                f"直近{recent10}世代出生-死亡": birth_death_diff_10,
                f"直近{recent10}世代資源収支": net_recent,
                "平均所持資源": avg_bag,
                "空腹比率": hunger,
                "交尾成功参加": mate,
                "捕食試行": pred,
                "読み取り": reading,
                "なぜそう読めるか": _why_join(causal_parts),
            })

        philo_tab = pd.DataFrame(philo_rows)
        gene_notes = []
        gene_why = []
        if len(philo_tab) > 0:
            best_ratio = philo_tab.sort_values("最新比率", ascending=False).iloc[0]
            best_delta = philo_tab.sort_values("比率変化", ascending=False).iloc[0]
            worst_delta = philo_tab.sort_values("比率変化", ascending=True).iloc[0]
            best_w = philo_tab.sort_values(f"直近{recent10}世代W", ascending=False).iloc[0]
            worst_w = philo_tab.sort_values(f"直近{recent10}世代W", ascending=True).iloc[0]
            gene_notes.append(f"最新比率で最も多いのは **{best_ratio['哲学型']}**（{best_ratio['最新比率']:.3f}）です。初期から最も伸びたのは **{best_delta['哲学型']}**（{best_delta['比率変化']:+.3f}）、最も縮小したのは **{worst_delta['哲学型']}**（{worst_delta['比率変化']:+.3f}）です。")
            gene_why.append("比率は相対値なので、増加した型については必ず実数変化とWも見ます。比率だけ上がって実数が減っている場合、それは『増えた』というより『他型より減りにくかった』と読むべきです。")
            gene_notes.append(f"直近{recent10}世代Wが最も高いのは **{best_w['哲学型']}**（W={best_w[f'直近{recent10}世代W']:.3f}）、最も低いのは **{worst_w['哲学型']}**（W={worst_w[f'直近{recent10}世代W']:.3f}）です。")
            gene_why.append("直近Wは現在の環境圧に対する応答です。全期間で増えた型でも、直近Wが低ければ現在は失速している可能性があります。逆に全期間で減った型でも、直近Wが高ければ回復局面かもしれません。")
            if philo_div_delta < -0.05:
                gene_notes.append(f"哲学遺伝子多様度は初期から **{philo_div_delta:+.3f}** 変化し、低下しています。")
                gene_why.append("多様度低下は、特定型への偏りが強まったか、個体数低下で偶然の偏りが増えたことを示します。選択圧とドリフトを分けるにはseed違いの比較が必要です。")
            elif philo_div_delta > 0.05:
                gene_notes.append(f"哲学遺伝子多様度は初期から **{philo_div_delta:+.3f}** 変化し、上昇しています。")
                gene_why.append("多様度上昇は、単一型への固定が弱く、複数の哲学方針が環境内で共存している可能性を示します。")
            else:
                gene_notes.append(f"哲学遺伝子多様度の初期からの変化は **{philo_div_delta:+.3f}** です。")
                gene_why.append("多様度が大きく崩れていないため、現段階では単純な一強固定ではなく、複数型の競合が続いていると読めます。")

        # --- 原因候補マトリクス ---
        cause_rows = []
        cause_rows.append({
            "観察された状態": "個体群W",
            "最新/直近値": f"今 {W_pop_now:.3f} / 直近{recent10}平均 {W_pop_10:.3f}",
            "なぜ重要か": "Wはコピー増殖率なので、1からのズレが個体群全体の増減圧を示す",
            "原因候補": "出生不足・死亡過多・資源不足・交尾失敗・過密/近親制約",
            "次に見る列": "出生数、死亡数、交尾成立、空腹比率、資源収支",
        })
        cause_rows.append({
            "観察された状態": "資源接続",
            "最新/直近値": f"資源Δ{res_delta10:+.0f} / 所持資源Δ{mean_bag_delta10:+.2f}",
            "なぜ重要か": "盤面資源があっても個体が持てなければ繁殖へ接続しない",
            "原因候補": "局所資源・認識半径・移動コスト・採取量・Gini",
            "次に見る列": "資源総量、平均所持資源、採取総量、Gini、局所密度",
        })
        cause_rows.append({
            "観察された状態": "繁殖ボトルネック",
            "最新/直近値": f"交尾成立率 {mate_rate_10:.2f} / 出生死亡比 {bd_10:.2f}",
            "なぜ重要か": "遺伝子コピーは最終的に出生しなければ次世代へ増えない",
            "原因候補": "相互選択失敗・相手不足・空きマス不足・近親回避・資源不足",
            "次に見る列": "交尾試行、交尾成立、出生予約、実出生、近親回避、過密抑制",
        })
        cause_rows.append({
            "観察された状態": "哲学遺伝子差",
            "最新/直近値": f"多様度 {philo_div:.3f} / Δ{philo_div_delta:+.3f}",
            "なぜ重要か": "多様度と型別Wから、思想型の固定・共存・ドリフトを切り分ける",
            "原因候補": "行動評価関数の差・資源収支差・死亡差・出生差・偶然の浮動",
            "次に見る列": "各哲学型W、比率、数、資源収支、空腹、出生、死亡",
        })

        with st.expander("現在の読み取り：なぜそうなっているかまで含む総合解釈", expanded=True):
            st.markdown("#### 1. 個体群レベル")
            for msg in eco_notes:
                st.markdown("- " + msg)
            st.markdown("**なぜそう読めるか**")
            for msg in eco_why:
                st.markdown("- " + msg)

            st.markdown("#### 2. 資源・エネルギー収支")
            for msg in resource_notes:
                st.markdown("- " + msg)
            st.markdown("**なぜそう読めるか**")
            for msg in resource_why:
                st.markdown("- " + msg)

            st.markdown("#### 3. 行動イベント")
            for msg in behavior_notes:
                st.markdown("- " + msg)
            st.markdown("**なぜそう読めるか**")
            for msg in behavior_why:
                st.markdown("- " + msg)

            st.markdown("#### 4. 哲学遺伝子の流れ")
            for msg in gene_notes:
                st.markdown("- " + msg)
            st.markdown("**なぜそう読めるか**")
            for msg in gene_why:
                st.markdown("- " + msg)

            st.markdown("#### 5. 原因候補マトリクス")
            st.caption("この表は、観察された数値から『どの仕組みが原因候補か』へ橋渡しするためのものです。")
            st.dataframe(pd.DataFrame(cause_rows), use_container_width=True, hide_index=True)

            if len(philo_tab) > 0:
                st.markdown("#### 6. 哲学型ごとの『なぜ』")
                st.dataframe(
                    philo_tab.style.format({
                        "初期数": "{:.0f}",
                        "最新数": "{:.0f}",
                        "数変化": "{:+.0f}",
                        "初期比率": "{:.3f}",
                        "最新比率": "{:.3f}",
                        "比率変化": "{:+.3f}",
                        "最新W": "{:.3f}",
                        f"直近{recent10}世代W": "{:.3f}",
                        f"直近{recent50}世代W": "{:.3f}",
                        f"直近{recent10}世代出生-死亡": "{:+.1f}",
                        f"直近{recent10}世代資源収支": "{:+.1f}",
                        "平均所持資源": "{:.2f}",
                        "空腹比率": "{:.3f}",
                        "交尾成功参加": "{:.0f}",
                        "捕食試行": "{:.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption("この読み取りは、数値から原因候補を出すための補助です。断定ではなく、次にモデル修正・seed比較・ON/OFF比較で検証する仮説として扱います。")


    def show_whole_run_summary():
        """
        全履歴dfを使った長めの研究サマリー。
        直近表示範囲dffではなく、保存されている全世代を対象にする。
        数値の列挙だけでなく「なぜそう読めるか」を併記する。
        """
        if len(df) < 2:
            return

        def safe_mean(col: str, data=None, default=0.0):
            data = df if data is None else data
            if col not in data.columns or len(data) == 0:
                return float(default)
            v = pd.to_numeric(data[col], errors="coerce").dropna()
            return float(v.mean()) if len(v) else float(default)

        def safe_last(col: str, default=0.0):
            if col not in df.columns:
                return float(default)
            try:
                v = pd.to_numeric(df[col], errors="coerce").iloc[-1]
                return float(v) if not pd.isna(v) else float(default)
            except Exception:
                return float(default)

        def safe_first(col: str, default=0.0):
            if col not in df.columns:
                return float(default)
            try:
                v = pd.to_numeric(df[col], errors="coerce").iloc[0]
                return float(v) if not pd.isna(v) else float(default)
            except Exception:
                return float(default)

        def safe_delta(col: str, data=None, default=0.0):
            data = df if data is None else data
            if col not in data.columns or len(data) < 2:
                return float(default)
            s = pd.to_numeric(data[col], errors="coerce").dropna()
            if len(s) < 2:
                return float(default)
            return float(s.iloc[-1] - s.iloc[0])

        def safe_minmax(col: str):
            if col not in df.columns:
                return 0.0, 0.0, None, None
            s = pd.to_numeric(df[col], errors="coerce")
            if s.dropna().empty:
                return 0.0, 0.0, None, None
            i_min = int(s.idxmin())
            i_max = int(s.idxmax())
            gen_min = int(df.loc[i_min, xcol]) if xcol in df.columns else i_min
            gen_max = int(df.loc[i_max, xcol]) if xcol in df.columns else i_max
            return float(s.min()), float(s.max()), gen_min, gen_max

        def safe_corr(a: str, b: str):
            if a not in df.columns or b not in df.columns:
                return np.nan
            aa = pd.to_numeric(df[a], errors="coerce")
            bb = pd.to_numeric(df[b], errors="coerce")
            d = pd.concat([aa, bb], axis=1).dropna()
            if len(d) < 3:
                return np.nan
            return float(d.iloc[:, 0].corr(d.iloc[:, 1]))

        def fmt_corr(v):
            return "不足" if pd.isna(v) else f"{v:+.3f}"

        full_n = int(len(df))
        recent_n = int(min(50, full_n))
        recent = df.tail(recent_n).copy()

        pop_start = safe_first("個体数（体）")
        pop_end = safe_last("個体数（体）")
        pop_min, pop_max, pop_min_gen, pop_max_gen = safe_minmax("個体数（体）")
        pop_ratio = pop_end / max(pop_start, 1.0)
        pop_W_mean = safe_mean("個体群全体W（増殖率）")
        pop_W_recent = safe_mean("個体群全体W（増殖率）", recent)
        birth_mean_recent = safe_mean("出生数（体/世代）", recent)
        death_mean_recent = safe_mean("死亡数（体/世代）", recent)
        bd_recent = birth_mean_recent / max(death_mean_recent, 1e-9)

        res_start = safe_first("資源総量（単位）")
        res_end = safe_last("資源総量（単位）")
        res_min, res_max, res_min_gen, res_max_gen = safe_minmax("資源総量（単位）")
        mean_bag_start = safe_first("平均所持資源（単位/体）")
        mean_bag_end = safe_last("平均所持資源（単位/体）")
        gini_start = safe_first("資源格差Gini（0-1）")
        gini_end = safe_last("資源格差Gini（0-1）")
        philo_div_start = safe_first("哲学遺伝子多様度（Simpson）")
        philo_div_end = safe_last("哲学遺伝子多様度（Simpson）")

        corr_pop_res = safe_corr("個体数（体）", "資源総量（単位）")
        corr_pop_bag = safe_corr("個体数（体）", "平均所持資源（単位/体）")
        corr_birth_bag = safe_corr("出生数（体/世代）", "平均所持資源（単位/体）")
        corr_death_gini = safe_corr("死亡数（体/世代）", "資源格差Gini（0-1）")

        # 状態ラベルは論文化可否ではなく、実験ランの観察性を表すだけにする。
        if pop_end >= 50 and 0.97 <= pop_W_recent <= 1.03:
            state_label = "遺伝子差を読み取りやすい維持状態"
            state_reason = "個体群が残り、直近Wも1付近なので、全体崩壊よりも型間差を読めるからです。"
        elif pop_end >= 20 and 0.93 <= pop_W_recent <= 1.05:
            state_label = "揺らぎを含む観察状態"
            state_reason = "個体群は残っていますが、直近の増減圧がややあるため、型間差と環境変動を分けて読む必要があるからです。"
        else:
            state_label = "生態圧が強く、型間差の解釈に注意が必要な状態"
            state_reason = "個体群が小さいかWが維持線から外れていると、哲学型の差が選択ではなく崩壊・増殖・偶然の浮動に見えやすいからです。"

        causal_readings = []
        if pop_W_recent < 0.97:
            causal_readings.append(f"直近{recent_n}世代のWが **{pop_W_recent:.3f}** なので、最近は縮小圧があります。なぜならW<1は、世代ごとのコピー数が前世代より少ないことを意味するからです。")
        elif pop_W_recent > 1.03:
            causal_readings.append(f"直近{recent_n}世代のWが **{pop_W_recent:.3f}** なので、最近は増殖圧があります。なぜならW>1は、世代ごとのコピー数が前世代より多いことを意味するからです。")
        else:
            causal_readings.append(f"直近{recent_n}世代のWが **{pop_W_recent:.3f}** なので、最近は維持線付近です。なぜならWが1に近いほど出生と死亡の総合結果が釣り合っているからです。")

        if res_end > res_start and mean_bag_end > mean_bag_start:
            causal_readings.append("資源総量と平均所持資源がどちらも増えています。なぜなら、環境側の資源ストックだけでなく、個体側の資源回収も増えているため、資源が個体の生存・繁殖へ接続している可能性があるからです。")
        elif res_end > res_start and mean_bag_end <= mean_bag_start:
            causal_readings.append("資源総量は増えているのに平均所持資源は増えていません。なぜなら、盤面に資源があっても、個体がそこへ到達・認識・採取できていない可能性があるからです。")
        elif res_end < res_start and mean_bag_end > mean_bag_start:
            causal_readings.append("資源総量は減っていますが平均所持資源は増えています。なぜなら、個体が環境ストックを強く回収している可能性があり、短期的には有利でも長期的には資源枯渇リスクがあるからです。")
        else:
            causal_readings.append("資源総量と平均所持資源がどちらも減っています。なぜなら、環境側の供給と個体側の回収が同時に弱くなっている可能性があり、死亡圧や繁殖抑制につながりやすいからです。")

        if philo_div_end < philo_div_start - 0.05:
            causal_readings.append(f"哲学多様度は **{philo_div_start:.3f}→{philo_div_end:.3f}** と低下しています。なぜなら、特定の哲学型が相対的に増えるか、個体群が減って偶然の偏りが強まると、Simpson多様度は下がるからです。")
        elif philo_div_end > philo_div_start + 0.05:
            causal_readings.append(f"哲学多様度は **{philo_div_start:.3f}→{philo_div_end:.3f}** と上昇しています。なぜなら、型の偏りが弱まり、複数の哲学型が共存しているほど多様度が高くなるからです。")
        else:
            causal_readings.append(f"哲学多様度は **{philo_div_start:.3f}→{philo_div_end:.3f}** で大きくは崩れていません。なぜなら、単一型への固定よりも、複数型の競合が残っている状態だからです。")

        corr_notes = []
        corr_notes.append(f"個体数と資源総量の相関：**{fmt_corr(corr_pop_res)}**。正なら資源が多い時に個体数も多く、負なら個体数増加が資源を削る関係が疑われます。")
        corr_notes.append(f"個体数と平均所持資源の相関：**{fmt_corr(corr_pop_bag)}**。正なら個体増加と個体資源が両立し、負なら密度上昇で一体あたり資源が薄まる可能性があります。")
        corr_notes.append(f"出生数と平均所持資源の相関：**{fmt_corr(corr_birth_bag)}**。正なら資源保持が繁殖に接続している可能性があり、弱いなら繁殖制約は資源以外にあります。")
        corr_notes.append(f"死亡数とGiniの相関：**{fmt_corr(corr_death_gini)}**。正なら資源格差が死亡圧と連動している可能性があります。")

        with st.expander("全体サマリー：この実験ランが示していることと、その理由", expanded=True):
            st.markdown(f"""
### 状態：**{state_label}**
**なぜそう見るか**：{state_reason}

この欄は、表示範囲ではなく**保存されている全 {full_n} 世代**を対象にした要約です。個体数だけではなく、W・頻度・多様度・資源フローを合わせて読みます。理由は、この研究の観察対象が個体の勝敗ではなく、哲学的行動方針を持つ**遺伝子コピーの増減**だからです。
""")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("初期→最新 個体数", f"{int(pop_start)}→{int(pop_end)}体", f"{pop_ratio:.2f}倍")
            c2.metric("個体数 最小/最大", f"{int(pop_min)} / {int(pop_max)}", f"min世代 {pop_min_gen} / max世代 {pop_max_gen}")
            c3.metric(f"直近{recent_n}世代 平均W", f"{pop_W_recent:.3f}", "1.0が維持線")
            c4.metric("哲学多様度 初期→最新", f"{philo_div_start:.3f}→{philo_div_end:.3f}")

            st.markdown(f"""
#### 基本数値
- 個体群全体Wの全体平均：**{pop_W_mean:.3f}**、直近{recent_n}世代平均：**{pop_W_recent:.3f}**。  
- 直近{recent_n}世代の出生/死亡比：**{bd_recent:.2f}**。  
- 資源総量：**{int(res_start)}→{int(res_end)}**、最小 **{int(res_min)}**（世代{res_min_gen}）、最大 **{int(res_max)}**（世代{res_max_gen}）。  
- 平均所持資源：**{mean_bag_start:.2f}→{mean_bag_end:.2f}**。  
- 資源格差Gini：**{gini_start:.3f}→{gini_end:.3f}**。

#### 事実に基づく考察
""")
            for msg in causal_readings:
                st.markdown("- " + msg)

            st.markdown("#### 相関から見る原因候補")
            for msg in corr_notes:
                st.markdown("- " + msg)
            st.caption("相関は原因を証明しません。ただし、どの関係を次にON/OFF比較すべきかを決める手がかりになります。")

            # 時期別サマリー
            if full_n >= 9:
                # Streamlit Cloud の pandas/numpy 環境では、np.array_split(df, 3) が
                # DataFrame ではなく ndarray を返す場合がある。
                # その場合 block.columns が存在せず AttributeError になるため、
                # DataFrame の iloc で明示的に3分割する。
                cut_points = np.linspace(0, len(df), 4, dtype=int)
                cuts = [df.iloc[cut_points[i]:cut_points[i + 1]].copy() for i in range(3)]
                names = ["前期", "中期", "後期"]
                phase_rows = []
                prev = None
                for nm, block in zip(names, cuts):
                    if len(block) == 0:
                        continue
                    dom_name = ""
                    dom_ratio = -1.0
                    for lab in PHILO_LABELS.values():
                        col = f"{lab} 比率（0-1）"
                        if col in block.columns:
                            r = safe_mean(col, block)
                            if r > dom_ratio:
                                dom_ratio = r
                                dom_name = lab
                    pop_mean = safe_mean("個体数（体）", block)
                    W_mean = safe_mean("個体群全体W（増殖率）", block)
                    res_mean = safe_mean("資源総量（単位）", block)
                    bag_mean = safe_mean("平均所持資源（単位/体）", block)
                    gini_mean = safe_mean("資源格差Gini（0-1）", block)
                    div_mean = safe_mean("哲学遺伝子多様度（Simpson）", block)
                    birth_mean = safe_mean("出生数（体/世代）", block)
                    death_mean = safe_mean("死亡数（体/世代）", block)
                    bd = birth_mean / max(death_mean, 1e-9)

                    why = []
                    if W_mean < 0.97:
                        why.append("平均Wが1未満なので縮小圧")
                    elif W_mean > 1.03:
                        why.append("平均Wが1超なので増殖圧")
                    else:
                        why.append("平均Wが1付近なので維持状態")
                    if bd < 0.9:
                        why.append("死亡が出生を上回る")
                    elif bd > 1.1:
                        why.append("出生が死亡を上回る")
                    else:
                        why.append("出生と死亡が拮抗")
                    if prev is not None:
                        if res_mean > prev["res"] and pop_mean < prev["pop"]:
                            why.append("資源は増えたが個体数は減ったため利用接続に注意")
                        elif res_mean < prev["res"] and pop_mean > prev["pop"]:
                            why.append("個体数増加が資源を消費している可能性")
                        if div_mean < prev["div"] - 0.02:
                            why.append("哲学多様度が低下し、型の偏りが強まる")
                    prev = {"pop": pop_mean, "res": res_mean, "div": div_mean}

                    phase_rows.append({
                        "区間": nm,
                        "世代範囲": f"{int(block[xcol].iloc[0])}〜{int(block[xcol].iloc[-1])}" if xcol in block.columns else "",
                        "平均個体数": pop_mean,
                        "平均W": W_mean,
                        "出生/死亡比": bd,
                        "平均資源総量": res_mean,
                        "平均所持資源": bag_mean,
                        "平均Gini": gini_mean,
                        "哲学多様度": div_mean,
                        "最多哲学型": dom_name,
                        "最多型平均比率": dom_ratio,
                        "なぜそう読めるか": " / ".join(why),
                    })
                st.markdown("#### 時期別サマリー")
                st.caption("前期・中期・後期で、個体群と遺伝子構成がどう変わったかをまとめます。最後の列で、数値をどう理由づけて読むかを示します。")
                st.dataframe(pd.DataFrame(phase_rows).style.format({
                    "平均個体数": "{:.1f}",
                    "平均W": "{:.3f}",
                    "出生/死亡比": "{:.2f}",
                    "平均資源総量": "{:.1f}",
                    "平均所持資源": "{:.2f}",
                    "平均Gini": "{:.3f}",
                    "哲学多様度": "{:.3f}",
                    "最多型平均比率": "{:.3f}",
                }), use_container_width=True, hide_index=True)


    def show_philosophy_gene_flow_summary():
        if len(df) < 2:
            return

        def _s(col):
            if col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce")

        def _first(col, default=np.nan):
            s = _s(col).dropna()
            return float(s.iloc[0]) if len(s) else default

        def _last(col, default=np.nan):
            s = _s(col).dropna()
            return float(s.iloc[-1]) if len(s) else default

        def _mean(col, data=None, default=np.nan):
            data = df if data is None else data
            if col not in data.columns:
                return default
            s = pd.to_numeric(data[col], errors="coerce").dropna()
            return float(s.mean()) if len(s) else default

        recent_n = int(min(50, len(df)))
        recent = df.tail(recent_n).copy()

        # 各世代で最多だった哲学型
        ratio_cols = {lab: f"{lab} 比率（0-1）" for lab in PHILO_LABELS.values() if f"{lab} 比率（0-1）" in df.columns}
        ratio_frame = pd.DataFrame({lab: pd.to_numeric(df[col], errors="coerce") for lab, col in ratio_cols.items()})
        dominant_series = ratio_frame.idxmax(axis=1) if len(ratio_frame.columns) else pd.Series(dtype=str)

        rows = []
        for lab in PHILO_LABELS.values():
            count_col = f"{lab} 数（体）"
            ratio_col = f"{lab} 比率（0-1）"
            W_col = f"{lab} W"
            if ratio_col not in df.columns:
                continue
            s = pd.to_numeric(df[ratio_col], errors="coerce")
            wv = pd.to_numeric(df[W_col], errors="coerce") if W_col in df.columns else pd.Series(dtype=float)
            count_s = pd.to_numeric(df[count_col], errors="coerce") if count_col in df.columns else pd.Series([np.nan] * len(df))
            max_i = int(s.idxmax()) if len(s.dropna()) else 0
            min_i = int(s.idxmin()) if len(s.dropna()) else 0
            latest_ratio = float(s.iloc[-1])
            first_ratio = float(s.iloc[0])
            latest_count = float(count_s.iloc[-1]) if len(count_s) else np.nan
            first_count = float(count_s.iloc[0]) if len(count_s) else np.nan
            recent_w = _mean(W_col, recent)
            recent_net = _mean(f"{lab} 資源収支ネット（単位/世代）", recent)
            recent_birth = _mean(f"{lab} 実出生（体/世代）", recent, 0.0)
            recent_death = _mean(f"{lab} 死亡（体/世代）", recent, 0.0)
            latest_hunger = _last(f"{lab} 空腹個体比率（0-1）", np.nan)
            latest_bag = _last(f"{lab} 平均所持資源（単位/体）", np.nan)
            latest_mate = _last(f"{lab} 交尾成功参加（回/世代）", 0.0)
            latest_pred = _last(f"{lab} 捕食試行（回/世代）", 0.0)

            why = []
            if latest_ratio > first_ratio + 0.02:
                why.append("比率が上昇")
            elif latest_ratio < first_ratio - 0.02:
                why.append("比率が低下")
            else:
                why.append("比率はほぼ維持")
            if not pd.isna(recent_w):
                if recent_w > 1.02:
                    why.append("直近Wが1超で現在も増殖寄り")
                elif recent_w < 0.98:
                    why.append("直近Wが1未満で現在は減少寄り")
                else:
                    why.append("直近Wは維持線付近")
            bd_diff = recent_birth - recent_death
            if bd_diff > 0.5:
                why.append("直近で出生が死亡を上回る")
            elif bd_diff < -0.5:
                why.append("直近で死亡が出生を上回る")
            if not pd.isna(recent_net):
                if recent_net > 10:
                    why.append("資源収支が正")
                elif recent_net < -10:
                    why.append("資源収支が負")
            if not pd.isna(latest_hunger):
                if latest_hunger > 0.40:
                    why.append("空腹個体が多い")
                elif latest_hunger < 0.15:
                    why.append("空腹個体が少ない")

            interpretation = " / ".join(why)
            if latest_ratio > first_ratio and latest_count < first_count:
                interpretation += "。ただし実数は減っているため、増加というより相対的残存の可能性があります。"
            elif latest_ratio > first_ratio and latest_count >= first_count:
                interpretation += "。比率と実数が同時に増えているため、コピー増殖として解釈しやすいです。"

            rows.append({
                "哲学型": lab,
                "初期数": first_count,
                "最新数": latest_count,
                "数変化": latest_count - first_count,
                "初期比率": first_ratio,
                "最新比率": latest_ratio,
                "比率変化": latest_ratio - first_ratio,
                "最大比率": float(s.max()),
                "最大世代": int(df.loc[max_i, xcol]) if xcol in df.columns else max_i,
                "最小比率": float(s.min()),
                "最小世代": int(df.loc[min_i, xcol]) if xcol in df.columns else min_i,
                "平均W": float(wv.mean()) if len(wv.dropna()) else np.nan,
                f"直近{recent_n}世代平均W": recent_w,
                f"直近{recent_n}世代出生-死亡": bd_diff,
                f"直近{recent_n}世代資源収支": recent_net,
                "最新平均所持資源": latest_bag,
                "最新空腹比率": latest_hunger,
                "最新交尾成功参加": latest_mate,
                "最新捕食試行": latest_pred,
                "支配世代数": int((dominant_series == lab).sum()) if len(dominant_series) else 0,
                "なぜ増減したと読めるか": interpretation,
            })
        if not rows:
            return
        tab = pd.DataFrame(rows)
        with st.expander("哲学個体遺伝子の世代ごとの流れ：なぜ増減したか", expanded=True):
            st.markdown("""
ここはこの研究の中心です。頻度の増減だけではなく、**実数・W・出生死亡差・資源収支・空腹・支配世代数**を並べます。  
なぜなら、哲学型の比率だけを見ると、個体群が減る局面で「相対的に残っただけ」の型を、誤って「増えた型」と読んでしまうからです。
""")
            st.dataframe(tab.style.format({
                "初期数": "{:.0f}", "最新数": "{:.0f}", "数変化": "{:+.0f}",
                "初期比率": "{:.3f}", "最新比率": "{:.3f}", "比率変化": "{:+.3f}",
                "最大比率": "{:.3f}", "最小比率": "{:.3f}",
                "平均W": "{:.3f}", f"直近{recent_n}世代平均W": "{:.3f}",
                f"直近{recent_n}世代出生-死亡": "{:+.1f}",
                f"直近{recent_n}世代資源収支": "{:+.1f}",
                "最新平均所持資源": "{:.2f}",
                "最新空腹比率": "{:.3f}",
                "最新交尾成功参加": "{:.0f}",
                "最新捕食試行": "{:.0f}",
                "支配世代数": "{:.0f}",
            }), use_container_width=True, hide_index=True)

            best_ratio = tab.sort_values("最新比率", ascending=False).iloc[0]
            best_delta = tab.sort_values("比率変化", ascending=False).iloc[0]
            weakest = tab.sort_values("比率変化", ascending=True).iloc[0]
            best_w = tab.sort_values(f"直近{recent_n}世代平均W", ascending=False).iloc[0]
            st.markdown(f"""
**現在の優勢型**は **{best_ratio['哲学型']}**（最新比率 {best_ratio['最新比率']:.3f}）です。  
**初期から最も比率を伸ばした型**は **{best_delta['哲学型']}**（{best_delta['比率変化']:+.3f}）です。  
**最も縮小した型**は **{weakest['哲学型']}**（{weakest['比率変化']:+.3f}）です。  
**直近で最も勢いがある型**は **{best_w['哲学型']}**（直近W {best_w[f'直近{recent_n}世代平均W']:.3f}）です。

**なぜこの4つを分けるか**：最新比率は「今多い型」、比率変化は「全期間で伸びた型」、直近Wは「今の環境で伸びている型」を表します。これらが一致すると強い解釈ができます。一致しない場合、過去の蓄積・現在の失速・偶然の残存を分けて考える必要があります。
""")



    def show_v19_lineage_flow_summary():
        """v19：親子フロー・行動フロー・チーム内偏りから、遺伝子の流れと因果候補を読む。"""
        if len(df) < 2:
            return

        def _num(col):
            if col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce")

        def _sum(col):
            s = _num(col).dropna()
            return float(s.sum()) if len(s) else 0.0

        def _mean(col):
            s = _num(col).dropna()
            return float(s.mean()) if len(s) else np.nan

        def _last(col):
            s = _num(col).dropna()
            return float(s.iloc[-1]) if len(s) else np.nan

        def _first(col):
            s = _num(col).dropna()
            return float(s.iloc[0]) if len(s) else np.nan

        def _corr(a, b):
            if a not in df.columns or b not in df.columns:
                return np.nan
            aa = _num(a); bb = _num(b)
            m = aa.notna() & bb.notna()
            if int(m.sum()) < 4:
                return np.nan
            if float(aa[m].std()) == 0.0 or float(bb[m].std()) == 0.0:
                return np.nan
            return float(aa[m].corr(bb[m]))

        st.markdown("### 親子遺伝子フロー：どの型が、どの経路で増えたか")
        st.caption("ここでは『現在多い型』ではなく、『親として子を残した型』『子として発生した型』『どの親組み合わせから生まれたか』『どの行動を選んだか』を分けて読みます。これにより、遺伝子頻度変化の原因候補をかなり細かく追えます。")

        # 1) 親としての繁殖貢献と子としての増加
        parent_rows = []
        for lab in PHILO_LABELS.values():
            count_col = f"{lab} 数（体）"
            ratio_col = f"{lab} 比率（0-1）"
            child_birth_col = f"{lab} 実出生（体/世代）"
            death_col = f"{lab} 死亡（体/世代）"
            parent_real_col = f"{lab} 親参加:実出生（回/世代）"
            parent_reserved_col = f"{lab} 親参加:出生予約（回/世代）"
            exposure = _sum(count_col)
            parent_real = _sum(parent_real_col)
            parent_reserved = _sum(parent_reserved_col)
            child_birth = _sum(child_birth_col)
            death_total = _sum(death_col)
            first_ratio = _first(ratio_col)
            last_ratio = _last(ratio_col)
            per_capita_parent = parent_real / max(exposure, 1.0)
            per_capita_child = child_birth / max(exposure, 1.0)
            parent_to_child_gap = parent_real - child_birth
            why = []
            if per_capita_parent > 0.08:
                why.append("親として子を残す効率が高い")
            elif per_capita_parent < 0.03:
                why.append("親として子を残す効率が低い")
            else:
                why.append("親としての繁殖効率は中程度")
            if child_birth > death_total:
                why.append("子としての増加が死亡を上回る")
            elif child_birth < death_total:
                why.append("死亡が子としての増加を上回る")
            if not np.isnan(first_ratio) and not np.isnan(last_ratio):
                if last_ratio > first_ratio + 0.03:
                    why.append("頻度は上昇")
                elif last_ratio < first_ratio - 0.03:
                    why.append("頻度は低下")
                else:
                    why.append("頻度はおおむね維持")
            if parent_to_child_gap > 0:
                why.append("混合ペアや相手側コピーにより、親参加ほど自型の子は増えていない可能性")
            parent_rows.append({
                "型": lab,
                "初期比率": first_ratio,
                "最新比率": last_ratio,
                "比率変化": (last_ratio - first_ratio) if not (np.isnan(first_ratio) or np.isnan(last_ratio)) else np.nan,
                "全期間親参加:出生予約": parent_reserved,
                "全期間親参加:実出生": parent_real,
                "全期間子として実出生": child_birth,
                "全期間死亡": death_total,
                "子出生-死亡": child_birth - death_total,
                "親繁殖効率/個体世代": per_capita_parent,
                "子発生効率/個体世代": per_capita_child,
                "読み取り": "、".join(why),
            })
        parent_df = pd.DataFrame(parent_rows).sort_values("親繁殖効率/個体世代", ascending=False)
        st.markdown("#### A. 親として残したコピー / 子として増えたコピー")
        st.dataframe(parent_df, use_container_width=True, hide_index=True, column_config={
            "初期比率": st.column_config.NumberColumn(format="%.3f"),
            "最新比率": st.column_config.NumberColumn(format="%.3f"),
            "比率変化": st.column_config.NumberColumn(format="%+.3f"),
            "全期間親参加:出生予約": st.column_config.NumberColumn(format="%.0f"),
            "全期間親参加:実出生": st.column_config.NumberColumn(format="%.0f"),
            "全期間子として実出生": st.column_config.NumberColumn(format="%.0f"),
            "全期間死亡": st.column_config.NumberColumn(format="%.0f"),
            "子出生-死亡": st.column_config.NumberColumn(format="%+.0f"),
            "親繁殖効率/個体世代": st.column_config.NumberColumn(format="%.4f"),
            "子発生効率/個体世代": st.column_config.NumberColumn(format="%.4f"),
        })
        if len(parent_df):
            top_parent = parent_df.iloc[0]
            low_parent = parent_df.iloc[-1]
            explain_box(
                "親子フローの中心解釈",
                f"親として最も子を残しやすい型は **{top_parent['型']}** です。なぜなら、全期間の個体存在量で割った親繁殖効率が最も高いからです。逆に **{low_parent['型']}** は親としての繁殖効率が低く、たとえ一時的に生存してもコピー数を増やしにくい可能性があります。\n\n"
                "ここで重要なのは、親参加と子としての実出生を分けることです。親参加が多いのに自型の子が少ない場合、混合ペアで相手型が子へコピーされている、あるいは出生後死亡で失われている可能性があります。子としての実出生が多いのに比率が伸びない場合は、出生後の死亡圧が強い可能性があります。"
            )

        # 2) 親→子フローマトリクス
        mat_parent_child = pd.DataFrame(0.0, index=list(PHILO_LABELS.values()), columns=list(PHILO_LABELS.values()))
        mat_pair = pd.DataFrame(0.0, index=list(PHILO_LABELS.values()), columns=list(PHILO_LABELS.values()))
        mat_source_child = pd.DataFrame(0.0, index=list(PHILO_LABELS.values()), columns=list(PHILO_LABELS.values()))
        for pi, plab in PHILO_LABELS.items():
            for ci, clab in PHILO_LABELS.items():
                mat_parent_child.loc[plab, clab] = _sum(f"親→子 実出生:{plab}→{clab}（回/世代）")
                mat_source_child.loc[plab, clab] = _sum(f"コピー元→子 実出生:{plab}→{clab}（回/世代）")
                if int(pi) <= int(ci):
                    v = _sum(f"親組合せ 実出生:{plab}×{clab}（回/世代）")
                    mat_pair.loc[plab, clab] = v
                    mat_pair.loc[clab, plab] = v
        st.markdown("#### B. 親→子フロー行列")
        st.caption("行=親として参加した型、列=実際に生まれた子の型。混合ペアでは、親2体ぶんが行に加算されます。")
        st.dataframe(mat_parent_child.style.format("{:.0f}"), use_container_width=True)
        st.markdown("#### C. 親組み合わせ行列")
        st.caption("どの型同士のペアが実出生につながったかです。対角線は同型同士、対角線以外は混合ペアです。")
        st.dataframe(mat_pair.style.format("{:.0f}"), use_container_width=True)
        st.markdown("#### D. コピー元→子フロー行列")
        st.caption("子が実際にどの型のコピーとして生まれたかです。現段階では突然変異を入れていないので基本的に対角線に出ます。将来、突然変異や文化的変換を入れたときに重要になります。")
        st.dataframe(mat_source_child.style.format("{:.0f}"), use_container_width=True)

        mixed_total = 0.0
        same_total = 0.0
        for i, lab_i in enumerate(PHILO_LABELS.values()):
            for j, lab_j in enumerate(PHILO_LABELS.values()):
                v = float(mat_pair.iloc[i, j])
                if i == j:
                    same_total += v
                else:
                    mixed_total += v
        # mat_pair is symmetric, off-diagonal counted twice; correction
        mixed_total = mixed_total / 2.0
        same_total = same_total
        if same_total + mixed_total > 0:
            mixed_ratio = mixed_total / (same_total + mixed_total)
            if mixed_ratio > 0.55:
                mix_msg = "混合ペアが多いので、型同士は完全に分離しておらず、環境中で交差しながら淘汰されています。これは、ある型が単独で勝ったというより、混合ペアの中でどちらのコピーが子へ渡るかが重要になる状態です。"
            elif mixed_ratio < 0.25:
                mix_msg = "同型ペアが多いので、型ごとに繁殖経路がやや分かれています。この場合、各型の行動方針がそのまま子孫数に反映されやすくなります。"
            else:
                mix_msg = "同型ペアと混合ペアがどちらもあります。型ごとの行動差と、型間の組み合わせ効果の両方を見る必要があります。"
            explain_box("親組み合わせから見た遺伝子の流れ", f"混合ペア比率は **{mixed_ratio:.3f}** です。{mix_msg}")

        # 3) 行動選択から見る因果候補
        action_rows = []
        for lab in PHILO_LABELS.values():
            row = {"型": lab}
            dominant_action = None
            dominant_rate = -1.0
            for act_i, act_label in PHILO_ACTION_LABELS.items():
                rate_col = f"{lab} 行動率:{act_label}（0-1）"
                val = _mean(rate_col)
                row[f"平均行動率:{act_label}"] = val
                if not np.isnan(val) and val > dominant_rate:
                    dominant_action = act_label
                    dominant_rate = val
            row["最多行動"] = dominant_action or "—"
            action_rows.append(row)
        action_df = pd.DataFrame(action_rows)
        st.markdown("#### E. 行動選択から見る『なぜ』")
        st.dataframe(action_df, use_container_width=True, hide_index=True, column_config={
            col: st.column_config.NumberColumn(format="%.3f") for col in action_df.columns if col.startswith("平均行動率:")
        })

        action_explain = []
        if len(action_df):
            for _, r in action_df.iterrows():
                lab = r["型"]
                dom = r.get("最多行動", "—")
                extra = []
                gather_rate = r.get("平均行動率:採取", np.nan)
                mate_rate = r.get("平均行動率:交尾", np.nan)
                avoid_rate = r.get("平均行動率:回避", np.nan)
                pred_rate = r.get("平均行動率:捕食", np.nan)
                if not np.isnan(gather_rate) and gather_rate > 0.25:
                    extra.append("採取が多く資源獲得型")
                if not np.isnan(mate_rate) and mate_rate > 0.12:
                    extra.append("交尾選択が多く繁殖志向")
                if not np.isnan(avoid_rate) and avoid_rate > 0.25:
                    extra.append("回避が多く死亡回避志向")
                if not np.isnan(pred_rate) and pred_rate > 0.08:
                    extra.append("捕食が多く高リスク資源獲得志向")
                action_explain.append(f"**{lab}** は最多行動が **{dom}** です。" + (" / ".join(extra) if extra else "大きく偏った行動は見えにくいです。"))
        explain_box("行動選択が因果候補になる理由", "\n\n".join(action_explain) + "\n\n行動率は直接の原因そのものではありませんが、出生・死亡・資源収支の前段階です。たとえば交尾率が高いのに子が少ないなら、配偶者・空きマス・近親回避で止まっている可能性があります。採取率が高いのに空腹率が高いなら、採取量・資源配置・移動コストの問題が疑えます。")

        # 4) 赤/青×哲学型の偏り
        team_rows = []
        for lab in PHILO_LABELS.values():
            red_last = _last(f"赤×{lab} 比率（赤内0-1）")
            blue_last = _last(f"青×{lab} 比率（青内0-1）")
            red_count = _last(f"赤×{lab} 数（体）")
            blue_count = _last(f"青×{lab} 数（体）")
            team_rows.append({
                "型": lab,
                "赤内最新比率": red_last,
                "青内最新比率": blue_last,
                "赤-青差": red_last - blue_last if not (np.isnan(red_last) or np.isnan(blue_last)) else np.nan,
                "赤最新数": red_count,
                "青最新数": blue_count,
            })
        team_df = pd.DataFrame(team_rows).sort_values("赤-青差", ascending=False)
        st.markdown("#### F. 赤チーム・青チーム内の哲学型偏り")
        st.dataframe(team_df, use_container_width=True, hide_index=True, column_config={
            "赤内最新比率": st.column_config.NumberColumn(format="%.3f"),
            "青内最新比率": st.column_config.NumberColumn(format="%.3f"),
            "赤-青差": st.column_config.NumberColumn(format="%+.3f"),
            "赤最新数": st.column_config.NumberColumn(format="%.0f"),
            "青最新数": st.column_config.NumberColumn(format="%.0f"),
        })
        if len(team_df):
            red_bias = team_df.iloc[0]
            blue_bias = team_df.iloc[-1]
            explain_box(
                "赤青差を哲学型の偏りとして読む",
                f"赤側に最も偏っている型は **{red_bias['型']}**、青側に最も偏っている型は **{blue_bias['型']}** です。赤青の個体数差を見るだけだとチーム色の優劣に見えますが、実際にはチーム内の哲学型・タカハト型・資源格差が偏っているだけの可能性があります。したがって、赤青差は『チームそのものの効果』と『チーム内遺伝子構成の効果』を分けて読む必要があります。"
            )

        # 5) 相関からの因果候補をもう一段増やす
        corr_rows = []
        for lab in PHILO_LABELS.values():
            pairs = [
                (f"{lab} 比率（0-1）", f"{lab} 親参加:実出生（回/世代）", "頻度と親繁殖参加"),
                (f"{lab} 比率（0-1）", f"{lab} 実出生（体/世代）", "頻度と子としての出生"),
                (f"{lab} 比率（0-1）", f"{lab} 死亡（体/世代）", "頻度と死亡"),
                (f"{lab} 比率（0-1）", f"{lab} 資源収支ネット（単位/世代）", "頻度と資源収支"),
                (f"{lab} 比率（0-1）", f"{lab} 空腹個体比率（0-1）", "頻度と空腹"),
                (f"{lab} 比率（0-1）", f"{lab} 行動率:交尾（0-1）", "頻度と交尾行動"),
                (f"{lab} 比率（0-1）", f"{lab} 行動率:採取（0-1）", "頻度と採取行動"),
            ]
            for a, b, name in pairs:
                cv = _corr(a, b)
                if not np.isnan(cv):
                    corr_rows.append({"型": lab, "関係": name, "相関": cv, "読み方": "正なら一緒に増える / 負なら逆方向。因果の証明ではなく、疑う経路の候補。"})
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows).sort_values("相関", key=lambda s: s.abs(), ascending=False).head(30)
            st.markdown("#### G. 頻度変化とイベントの相関：因果候補ランキング")
            st.dataframe(corr_df, use_container_width=True, hide_index=True, column_config={
                "相関": st.column_config.NumberColumn(format="%+.3f"),
            })
            top_corr = corr_df.iloc[0]
            explain_box("因果候補の読み方", f"最も強く連動している候補は **{top_corr['型']}** の **{top_corr['関係']}**（相関 {top_corr['相関']:+.3f}）です。これは原因の証明ではありませんが、次にON/OFF比較やseed比較で検証すべき最有力候補です。")


    def show_deep_causal_interpretation():
        """全履歴から、優勢/劣勢・淘汰圧・チーム差・因果候補を文章化する。"""
        if df is None or len(df) < 3:
            st.info("履歴が少ないため、深い読み取りにはもう少し世代を進めてください。")
            return

        d = df.copy()
        full_n = len(d)
        xcol = "世代（回)" if "世代（回)" in d.columns else "世代（回）"
        if xcol not in d.columns:
            xcol = d.columns[0]

        def has(col): return col in d.columns
        def s(col):
            if col not in d.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(d[col], errors="coerce")
        def last(col, default=0.0):
            ss = s(col).dropna()
            return float(ss.iloc[-1]) if len(ss) else float(default)
        def first(col, default=0.0):
            ss = s(col).dropna()
            return float(ss.iloc[0]) if len(ss) else float(default)
        def mean(col):
            ss = s(col).dropna()
            return float(ss.mean()) if len(ss) else 0.0
        def trend(col):
            ss = s(col).dropna()
            return float(ss.iloc[-1] - ss.iloc[0]) if len(ss) >= 2 else 0.0
        def slope(col):
            ss = s(col).dropna()
            if len(ss) < 3: return 0.0
            xx = np.arange(len(ss), dtype=float)
            try:
                return float(np.polyfit(xx, ss.to_numpy(dtype=float), 1)[0])
            except Exception:
                return 0.0
        def corr(a,b):
            if a not in d.columns or b not in d.columns: return np.nan
            aa = s(a); bb=s(b)
            m = aa.notna() & bb.notna()
            if int(m.sum()) < 3: return np.nan
            if float(aa[m].std()) == 0.0 or float(bb[m].std()) == 0.0: return np.nan
            return float(aa[m].corr(bb[m]))
        def fmt(x, nd=3):
            if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
            return f"{float(x):.{nd}f}"

        st.markdown("### 0.5) 深掘りサマリー：なぜそうなっているか")
        st.caption("この欄は最新10世代だけではなく、保存されている全履歴・前半/後半・最大/最小・相関を合わせて読みます。相関は原因の証明ではありませんが、次に疑うべき淘汰圧を絞る手がかりになります。")

        pop0, pop1 = first("個体数（体）"), last("個体数（体）")
        pop_max = float(s("個体数（体）").max()) if has("個体数（体）") else 0.0
        pop_min = float(s("個体数（体）").min()) if has("個体数（体）") else 0.0
        w_mean = mean("個体群全体W（増殖率）")
        res0, res1 = first("資源総量（単位）"), last("資源総量（単位）")
        res_slope = slope("資源総量（単位）")
        pop_slope = slope("個体数（体）")
        birth_total = float(s("出生数（体/世代）").sum()) if has("出生数（体/世代）") else 0.0
        death_total = float(s("死亡数（体/世代）").sum()) if has("死亡数（体/世代）") else 0.0
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_birth_resource = corr("出生数（体/世代）", "平均所持資源（単位/体）")
        c_birth_mate = corr("出生数（体/世代）", "交尾成立（回/世代）")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("個体数 初期→最新", f"{int(pop0)}→{int(pop1)}", f"最大{int(pop_max)} / 最小{int(pop_min)}")
        colB.metric("全期間平均W", f"{w_mean:.3f}", "1.0が維持線")
        colC.metric("出生総数/死亡総数", f"{int(birth_total)}/{int(death_total)}", f"差 {int(birth_total-death_total):+d}")
        colD.metric("資源 初期→最新", f"{int(res0)}→{int(res1)}", f"傾き {res_slope:+.2f}/世代")

        pop_read = []
        if w_mean > 1.02:
            pop_read.append("全期間平均Wが1を上回るため、個体群には増殖圧がかかっています。なぜならWは前世代比のコピー数で、平均的に1を超えると出生・生存が死亡を上回りやすいからです。")
        elif w_mean < 0.98:
            pop_read.append("全期間平均Wが1を下回るため、個体群には縮小圧がかかっています。なぜなら、出生があっても維持コスト・死亡・寿命による消失を十分に補えていない可能性が高いからです。")
        else:
            pop_read.append("全期間平均Wはほぼ1付近です。これは全体集団が爆発も崩壊もしていないため、集団内部の遺伝子差を観察しやすい状態です。")
        if not np.isnan(c_pop_res):
            if c_pop_res > 0.35:
                pop_read.append(f"個体数と資源総量の相関は **{c_pop_res:+.3f}** で正です。資源が多い局面ほど個体数も多く、資源供給が個体群維持を支えている可能性があります。")
            elif c_pop_res < -0.35:
                pop_read.append(f"個体数と資源総量の相関は **{c_pop_res:+.3f}** で負です。個体数が増えるほど盤面資源が削られている可能性があり、資源消費型の密度圧が疑えます。")
            else:
                pop_read.append(f"個体数と資源総量の相関は **{c_pop_res:+.3f}** で弱めです。盤面資源だけでなく、資源の位置・探索・交尾成立・死亡圧も個体数を左右している可能性があります。")
        if not np.isnan(c_birth_resource):
            pop_read.append(f"出生数と平均所持資源の相関は **{c_birth_resource:+.3f}** です。正なら繁殖制約が資源寄り、弱いなら資源以外、たとえば出会い・近親回避・空きマスが制約になっている可能性があります。")
        if not np.isnan(c_birth_mate):
            pop_read.append(f"出生数と交尾成立の相関は **{c_birth_mate:+.3f}** です。ここが強いほど、個体数変化の直接因は資源よりも配偶者探索・交尾成立に近いと読めます。")
        if not np.isnan(c_death_gini):
            pop_read.append(f"死亡数と資源格差Giniの相関は **{c_death_gini:+.3f}** です。正なら、資源の総量不足より『資源の偏り』が死亡圧に接続している可能性があります。")
        explain_box("個体群全体の読み取り", "\n\n".join(pop_read))

        # 遺伝子ごとの優勢/劣勢評価
        gene_rows = []
        for lab in PHILO_LABELS.values():
            count_col = f"{lab} 数（体）"; ratio_col=f"{lab} 比率（0-1）"; w_col=f"{lab} W"
            if ratio_col not in d.columns: continue
            count0, count1 = first(count_col), last(count_col)
            ratio0, ratio1 = first(ratio_col), last(ratio_col)
            w_avg = mean(w_col)
            w_slope = slope(w_col)
            net_avg = mean(f"{lab} 資源収支ネット（単位/世代）")
            hunger_avg = mean(f"{lab} 空腹個体比率（0-1）")
            birth_sum = float(s(f"{lab} 実出生（体/世代）").sum()) if has(f"{lab} 実出生（体/世代）") else 0.0
            death_sum = float(s(f"{lab} 死亡（体/世代）").sum()) if has(f"{lab} 死亡（体/世代）") else 0.0
            dominance_score = (ratio1 - ratio0) + 0.35 * (w_avg - 1.0) + 0.002 * (birth_sum - death_sum)
            why = []
            if ratio1 - ratio0 > 0.03:
                why.append("頻度が初期より上昇")
            elif ratio1 - ratio0 < -0.03:
                why.append("頻度が初期より低下")
            else:
                why.append("頻度変化は小さい")
            if w_avg > 1.01:
                why.append("平均Wが1超でコピー増加圧")
            elif w_avg < 0.99:
                why.append("平均Wが1未満でコピー減少圧")
            else:
                why.append("平均Wはほぼ維持線")
            if birth_sum > death_sum:
                why.append("実出生が死亡を上回る")
            elif birth_sum < death_sum:
                why.append("死亡が実出生を上回る")
            if hunger_avg > 0.45:
                why.append("空腹率が高く資源制約を受けやすい")
            if net_avg > 0:
                why.append("資源収支は正")
            elif net_avg < 0:
                why.append("資源収支は負")
            gene_rows.append({
                "型": lab,
                "初期数": int(count0), "最新数": int(count1), "数変化": int(count1-count0),
                "初期比率": ratio0, "最新比率": ratio1, "比率変化": ratio1-ratio0,
                "平均W": w_avg, "W傾き": w_slope,
                "出生合計": int(birth_sum), "死亡合計": int(death_sum), "出生-死亡": int(birth_sum-death_sum),
                "平均資源収支": net_avg, "平均空腹率": hunger_avg,
                "優勢スコア": dominance_score,
                "なぜ": "、".join(why)
            })
        if gene_rows:
            gene_df = pd.DataFrame(gene_rows).sort_values("優勢スコア", ascending=False)
            st.markdown("#### 遺伝子型の優勢・劣勢")
            st.dataframe(gene_df, use_container_width=True, hide_index=True, column_config={
                "初期比率": st.column_config.NumberColumn(format="%.3f"),
                "最新比率": st.column_config.NumberColumn(format="%.3f"),
                "比率変化": st.column_config.NumberColumn(format="%+.3f"),
                "平均W": st.column_config.NumberColumn(format="%.3f"),
                "W傾き": st.column_config.NumberColumn(format="%+.4f"),
                "平均資源収支": st.column_config.NumberColumn(format="%+.2f"),
                "平均空腹率": st.column_config.NumberColumn(format="%.3f"),
                "優勢スコア": st.column_config.NumberColumn(format="%+.3f"),
            })
            best = gene_df.iloc[0]; worst = gene_df.iloc[-1]
            explain_box(
                "哲学/通常個体の淘汰読み取り",
                f"このランで最も優勢に見える型は **{best['型']}**、最も劣勢に見える型は **{worst['型']}** です。ここでの優勢は単なる最新比率ではなく、比率変化・平均W・出生死亡差を合わせた暫定評価です。なぜなら、最新比率だけだと『最初から多かった型』や『他型より減りにくかった型』を過大評価してしまうからです。\n\n"
                f"**{best['型']}** は {best['なぜ']}。一方、**{worst['型']}** は {worst['なぜ']}。この差は、資源獲得・空腹回避・交尾成功・死亡回避のどれかに接続している可能性があります。"
            )

        # タカ/ハト、捕食、チーム
        contest_rows=[]
        if has("タカ数（体）") and has("ハト数（体）"):
            for lab in ["タカ", "ハト"]:
                count_col=f"{lab}数（体）"
                if lab == "ハト" and count_col not in d.columns: count_col="ハト数（体）"
                if lab == "タカ" and count_col not in d.columns: count_col="タカ数（体）"
                w_col=f"{lab} 適応度W（コピー増殖率）"
                if w_col not in d.columns and lab == "タカ": w_col="タカ 適応度W（コピー増殖率）"
                if w_col not in d.columns and lab == "ハト": w_col="ハト 適応度W（コピー増殖率）"
                if count_col in d.columns:
                    contest_rows.append({"争奪遺伝子": lab, "初期数": int(first(count_col)), "最新数": int(last(count_col)), "数変化": int(last(count_col)-first(count_col)), "平均W": mean(w_col)})
        if contest_rows:
            cdf=pd.DataFrame(contest_rows)
            st.markdown("#### 争奪遺伝子：タカ/ハトの淘汰圧")
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            if len(cdf)>=2:
                top=cdf.sort_values(["数変化","平均W"], ascending=False).iloc[0]
                explain_box("タカ/ハトの読み取り", f"このランでは **{top['争奪遺伝子']}** が相対的に優勢に見えます。なぜなら、数変化と平均Wが争奪遺伝子のコピー維持を直接示すからです。タカが伸びるなら衝突で得る利得がコストを上回る環境、ハトが伸びるなら戦闘コスト・過密・資源格差が攻撃性を罰している環境が疑えます。")

        # 捕食傾向
        if has("捕食傾向比率（0-1）"):
            pred_delta = trend("捕食傾向比率（0-1）")
            pred_success = mean("捕食成功率（0-1）")
            pred_gain = mean("捕食獲得資源（単位/世代）")
            pred_fail = mean("捕食失敗（回/世代）")
            if pred_delta > 0.02:
                pred_msg = "捕食傾向遺伝子は増加傾向です。捕食成功率や獲得資源が高いなら、捕食が資源不足への適応として働いている可能性があります。"
            elif pred_delta < -0.02:
                pred_msg = "捕食傾向遺伝子は低下傾向です。捕食失敗・コスト・被害が大きく、捕食が長期的には不利になっている可能性があります。"
            else:
                pred_msg = "捕食傾向遺伝子は大きく変化していません。捕食は強い選択圧ではないか、他の圧と釣り合っている可能性があります。"
            explain_box("捕食遺伝子の読み取り", f"{pred_msg}\n\n平均捕食成功率は **{pred_success:.3f}**、平均捕食獲得資源は **{pred_gain:.2f}**、平均捕食失敗は **{pred_fail:.2f}** です。なぜこれを見るかというと、捕食は短期的な資源獲得と失敗コストを同時に持つため、単に試行回数だけでは有利不利が判断できないからです。")

        # チーム差
        team_lines=[]
        if has("赤個体数（体）") and has("青個体数（体）"):
            red0, red1 = first("赤個体数（体）"), last("赤個体数（体）")
            blue0, blue1 = first("青個体数（体）"), last("青個体数（体）")
            red_share = red1 / max(red1+blue1, 1.0)
            blue_share = blue1 / max(red1+blue1, 1.0)
            team_lines.append(f"赤は **{int(red0)}→{int(red1)}体**、青は **{int(blue0)}→{int(blue1)}体** です。最新比率は赤{red_share:.3f}、青{blue_share:.3f}です。")
            if red_share > blue_share + 0.05:
                team_lines.append("赤チームが優位です。なぜなら、最新個体数比で青より明確に多く、同じ環境下でコピー維持に成功しているからです。")
            elif blue_share > red_share + 0.05:
                team_lines.append("青チームが優位です。なぜなら、最新個体数比で赤より明確に多く、同じ環境下でコピー維持に成功しているからです。")
            else:
                team_lines.append("赤青差は大きくありません。チーム色そのものより、哲学型・資源配置・局所密度などの内部差が効いている可能性があります。")
            if has("赤 平均所持資源") and has("青 平均所持資源"):
                red_res = mean("赤 平均所持資源"); blue_res=mean("青 平均所持資源")
                team_lines.append(f"平均所持資源は赤 **{red_res:.2f}**、青 **{blue_res:.2f}** です。資源差が個体数差と同じ向きなら、チーム差は資源獲得の差に支えられている可能性があります。逆向きなら、資源以外の死亡率・交尾成立・空間配置が疑えます。")
            if has("赤 Gini") and has("青 Gini"):
                red_g = mean("赤 Gini"); blue_g=mean("青 Gini")
                team_lines.append(f"資源格差Giniは赤 **{red_g:.3f}**、青 **{blue_g:.3f}** です。Giniが高いチームは、平均資源が同じでも一部個体に資源が偏り、低資源個体が死亡しやすくなる可能性があります。")
            explain_box("赤チーム・青チームの特徴", "\n\n".join(team_lines))

        # 淘汰圧の総合推定
        pressures=[]
        if w_mean < 0.99: pressures.append(("死亡/維持コスト圧", "全期間平均Wが1未満で、個体群維持がやや難しいため。"))
        if not np.isnan(c_death_gini) and c_death_gini > 0.35: pressures.append(("資源格差圧", "死亡とGiniが正に連動しており、総量不足より分配の偏りが死亡を生む可能性があるため。"))
        if has("交尾成立率（0-1）") and mean("交尾成立率（0-1）") < 0.25: pressures.append(("配偶者探索/交尾成立圧", "交尾成立率が低く、資源を持っていても出生へ変換できない可能性があるため。"))
        if has("過密で抑制された出生候補（回/世代）") and mean("過密で抑制された出生候補（回/世代）") > 0.5: pressures.append(("密度依存圧", "出生候補が過密で抑制されており、単純な繁殖力だけでなく空間的余地が選択されているため。"))
        if has("捕食成功率（0-1）") and mean("捕食試行（回/世代）") > 0 and mean("捕食成功率（0-1）") < 0.25: pressures.append(("捕食失敗コスト圧", "捕食試行に対して成功率が低く、捕食傾向がリスクになりやすいため。"))
        if not pressures:
            pressures.append(("弱い複合選択圧", "単独で突出した圧は見えにくく、資源・交尾・密度・遺伝的浮動が複合している可能性が高いため。"))
        pressure_df=pd.DataFrame([{"推定される淘汰圧":a,"なぜそう読めるか":b} for a,b in pressures])
        st.markdown("#### このランで働いていそうな淘汰圧")
        st.dataframe(pressure_df, use_container_width=True, hide_index=True)




    def show_v21_deep_gene_causality():
        """v21：遺伝子相互作用・淘汰圧・因果候補を、文章として深く読む。"""
        if df is None or len(df) < 4:
            st.info("深層因果サマリーには、もう少し世代履歴が必要です。最低でも数世代、できれば50世代以上進めてください。")
            return

        d = df.copy()

        def has(col):
            return col in d.columns

        def ser(col):
            if col not in d.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(d[col], errors="coerce")

        def first(col, default=0.0):
            s0 = ser(col).dropna()
            return float(s0.iloc[0]) if len(s0) else float(default)

        def lastv(col, default=0.0):
            s0 = ser(col).dropna()
            return float(s0.iloc[-1]) if len(s0) else float(default)

        def meanv(col, default=0.0):
            s0 = ser(col).dropna()
            return float(s0.mean()) if len(s0) else float(default)

        def sumv(col, default=0.0):
            s0 = ser(col).dropna()
            return float(s0.sum()) if len(s0) else float(default)

        def delta(col):
            return lastv(col) - first(col)

        def slopev(col):
            s0 = ser(col).dropna()
            if len(s0) < 4:
                return 0.0
            try:
                x = np.arange(len(s0), dtype=float)
                return float(np.polyfit(x, s0.to_numpy(dtype=float), 1)[0])
            except Exception:
                return 0.0

        def corr(a, b):
            if a not in d.columns or b not in d.columns:
                return np.nan
            aa = ser(a); bb = ser(b)
            m = aa.notna() & bb.notna()
            if int(m.sum()) < 4:
                return np.nan
            if float(aa[m].std()) == 0.0 or float(bb[m].std()) == 0.0:
                return np.nan
            try:
                return float(aa[m].corr(bb[m]))
            except Exception:
                return np.nan

        def pct(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            return f"{float(x)*100:.1f}%"

        def fmt(x, nd=3, signed=False):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            if signed:
                return f"{float(x):+.{nd}f}"
            return f"{float(x):.{nd}f}"

        def ratio_text(x, high="高い", low="低い"):
            if x > 0.08:
                return high
            if x < -0.08:
                return low
            return "平均的"

        st.markdown("### 深層因果サマリー：遺伝子が、どの経路で、何に押されているか")
        explain_box(
            "この欄で読む因果の形",
            "このモデルでの因果は、いきなり『カント型だから勝った』のように読みません。より正確には、**遺伝子 → 行動評価 → 実際の行動 → 資源収支・危険回避・出会い → 出生/死亡 → W/頻度** という鎖で読みます。さらに、哲学遺伝子は単独で働くとは限らず、タカ/ハト遺伝子、捕食傾向遺伝子、赤青チーム、局所密度、資源配置と結合して効果を変えます。つまりここでは『どの遺伝子が強いか』だけでなく、**どの遺伝子の効果が、別の遺伝子や環境圧によって増幅・抑制されているか**を見ます。"
        )

        labels = list(PHILO_LABELS.values())
        global_taka = meanv("タカ比率（0-1）", 0.0)
        global_pred = meanv("捕食傾向比率（0-1）", 0.0)
        global_mate = meanv("交尾成立率（0-1）", 0.0)
        global_gini = meanv("資源格差Gini（0-1）", 0.0)

        # --- A: 遺伝子相互作用の全体表 ---
        interaction_rows = []
        for lab in labels:
            if not has(f"{lab} 比率（0-1）"):
                continue
            r0 = first(f"{lab} 比率（0-1）")
            r1 = lastv(f"{lab} 比率（0-1）")
            wavg = meanv(f"{lab} W", 1.0)
            net = meanv(f"{lab} 資源収支ネット（単位/世代）")
            hunger = meanv(f"{lab} 空腹個体比率（0-1）")
            births = sumv(f"{lab} 実出生（体/世代）")
            deaths = sumv(f"{lab} 死亡（体/世代）")
            parent_real = sumv(f"{lab} 親参加:実出生（回/世代）")
            mate_success = sumv(f"{lab} 交尾成功参加（回/世代）")
            taka_ratio = meanv(f"{lab} タカ比率（0-1）")
            pred_ratio = meanv(f"{lab} 捕食傾向比率（0-1）")
            red_frac = meanv(f"{lab} 赤比率（型内0-1）") if has(f"{lab} 赤比率（型内0-1）") else np.nan
            gather_rate = meanv(f"{lab} 行動率:採取（0-1）")
            mate_rate = meanv(f"{lab} 行動率:交尾（0-1）")
            escape_rate = meanv(f"{lab} 行動率:回避（0-1）")
            battle_rate = meanv(f"{lab} 行動率:戦闘（0-1）")
            pred_act_rate = meanv(f"{lab} 行動率:捕食（0-1）")
            dominant_path = []
            if net > 0:
                dominant_path.append("資源収支+")
            if hunger < 0.25:
                dominant_path.append("低空腹")
            if births > deaths:
                dominant_path.append("出生超過")
            if parent_real > 0 and parent_real >= deaths:
                dominant_path.append("親として複製")
            if escape_rate > battle_rate and escape_rate > 0.15:
                dominant_path.append("回避寄り")
            if battle_rate > escape_rate and battle_rate > 0.12:
                dominant_path.append("争奪寄り")
            if pred_act_rate > 0.08:
                dominant_path.append("捕食寄り")
            if not dominant_path:
                dominant_path.append("弱い/複合")

            mediator = []
            if not np.isnan(taka_ratio) and taka_ratio - global_taka > 0.08:
                mediator.append("タカ多め")
            elif not np.isnan(taka_ratio) and taka_ratio - global_taka < -0.08:
                mediator.append("ハト多め")
            if not np.isnan(pred_ratio) and pred_ratio - global_pred > 0.04:
                mediator.append("捕食傾向多め")
            elif not np.isnan(pred_ratio) and pred_ratio - global_pred < -0.04:
                mediator.append("非捕食寄り")
            if not np.isnan(red_frac) and red_frac > 0.58:
                mediator.append("赤寄り")
            elif not np.isnan(red_frac) and red_frac < 0.42:
                mediator.append("青寄り")
            if not mediator:
                mediator.append("明確な同伴遺伝子なし")

            dominance = (r1 - r0) + 0.35 * (wavg - 1.0) + 0.0015 * (births - deaths) + 0.0008 * net - 0.05 * hunger
            interaction_rows.append({
                "型": lab,
                "初期比率": r0,
                "最新比率": r1,
                "比率変化": r1-r0,
                "平均W": wavg,
                "出生-死亡": births-deaths,
                "親参加実出生": parent_real,
                "交尾成功参加": mate_success,
                "平均資源収支": net,
                "平均空腹率": hunger,
                "採取率": gather_rate,
                "交尾率": mate_rate,
                "回避率": escape_rate,
                "戦闘率": battle_rate,
                "捕食行動率": pred_act_rate,
                "タカ比率差": taka_ratio - global_taka if not np.isnan(taka_ratio) else np.nan,
                "捕食傾向差": pred_ratio - global_pred if not np.isnan(pred_ratio) else np.nan,
                "赤偏り": red_frac - 0.5 if not np.isnan(red_frac) else np.nan,
                "主な経路": " / ".join(dominant_path),
                "同伴遺伝子・媒介": " / ".join(mediator),
                "優勢スコア": dominance,
            })

        if interaction_rows:
            inter_df = pd.DataFrame(interaction_rows).sort_values("優勢スコア", ascending=False)
            st.markdown("#### A. 遺伝子相互作用テーブル")
            st.caption("『同伴遺伝子・媒介』は、その哲学/通常型の内部でタカ・捕食傾向・赤青チームが平均より偏っているかを示します。ここが偏ると、哲学型そのものの効果と、タカ/捕食/チームの効果が混ざります。")
            st.dataframe(inter_df, use_container_width=True, hide_index=True, column_config={
                "初期比率": st.column_config.NumberColumn(format="%.3f"),
                "最新比率": st.column_config.NumberColumn(format="%.3f"),
                "比率変化": st.column_config.NumberColumn(format="%+.3f"),
                "平均W": st.column_config.NumberColumn(format="%.3f"),
                "出生-死亡": st.column_config.NumberColumn(format="%+.0f"),
                "平均資源収支": st.column_config.NumberColumn(format="%+.2f"),
                "平均空腹率": st.column_config.NumberColumn(format="%.3f"),
                "採取率": st.column_config.NumberColumn(format="%.3f"),
                "交尾率": st.column_config.NumberColumn(format="%.3f"),
                "回避率": st.column_config.NumberColumn(format="%.3f"),
                "戦闘率": st.column_config.NumberColumn(format="%.3f"),
                "捕食行動率": st.column_config.NumberColumn(format="%.3f"),
                "タカ比率差": st.column_config.NumberColumn(format="%+.3f"),
                "捕食傾向差": st.column_config.NumberColumn(format="%+.3f"),
                "赤偏り": st.column_config.NumberColumn(format="%+.3f"),
                "優勢スコア": st.column_config.NumberColumn(format="%+.3f"),
            })

            best = inter_df.iloc[0]
            worst = inter_df.iloc[-1]
            explain_box(
                "優勢・劣勢の読み方を一段深くする",
                f"現時点で最も優勢に見えるのは **{best['型']}** です。ただし、これは『その思想名が本質的に強い』という意味ではありません。**{best['型']}** は、{best['主な経路']} という経路でコピー維持に近づいており、さらに {best['同伴遺伝子・媒介']} という媒介条件を持っています。つまり、この型の優位は、哲学遺伝子単体ではなく、資源収支・出生死亡・タカ/捕食/チーム偏りとの組み合わせとして読まなければいけません。\n\n最も劣勢に見えるのは **{worst['型']}** です。**{worst['型']}** は、{worst['主な経路']} と出ていますが、比率変化・W・出生死亡差の合成では弱く出ています。ここで見るべきなのは、死亡が多いのか、出生へ変換できないのか、資源収支が悪いのか、あるいは同伴しているタカ/捕食/チーム構成が不利に働いているのかです。"
            )

        # --- B: 各型の因果鎖 ---
        st.markdown("#### B. 各遺伝子型の因果鎖：増減を、行動→資源→出生/死亡へ分解する")
        driver_candidates = [
            ("資源収支", "{lab} 資源収支ネット（単位/世代）", "資源を増やせるほど、維持コスト・繁殖投資を支払い、死亡を避けやすくなります。"),
            ("空腹率", "{lab} 空腹個体比率（0-1）", "空腹率が高い型は、同じ個体数でも死亡圧・繁殖停止圧を受けやすくなります。"),
            ("実出生", "{lab} 実出生（体/世代）", "出生は遺伝子コピー数を直接増やす最終出口です。"),
            ("親参加実出生", "{lab} 親参加:実出生（回/世代）", "その型が親として複製に参加したかを示します。子として増えただけなのか、親として増やしたのかを分けます。"),
            ("死亡", "{lab} 死亡（体/世代）", "死亡はコピー数を直接減らします。ただし個体数が多い型ほど死亡数も増えるため、Wや比率と併読します。"),
            ("交尾成功", "{lab} 交尾成功参加（回/世代）", "資源を持っていても交尾へ接続しなければコピーは増えません。"),
            ("採取率", "{lab} 行動率:採取（0-1）", "採取率は資源獲得の入口です。高くても資源収支が悪いなら、資源量や移動コストが詰まっています。"),
            ("交尾率", "{lab} 行動率:交尾（0-1）", "交尾率が高くても実出生が低いなら、相手不在・近親回避・空きマス・資源不足が詰まりです。"),
            ("回避率", "{lab} 行動率:回避（0-1）", "回避率は死亡回避に寄与しますが、高すぎると資源獲得や交尾機会を失う可能性もあります。"),
            ("戦闘率", "{lab} 行動率:戦闘（0-1）", "戦闘は資源移転を得ますが、コストや死亡圧を増やす可能性があります。"),
            ("捕食率", "{lab} 行動率:捕食（0-1）", "捕食は飢餓への短期対応ですが、失敗コストや相手減少で長期不利になることがあります。"),
            ("タカ比率", "{lab} タカ比率（0-1）", "タカ遺伝子と結びつくと、哲学型の効果が争奪戦略によって増幅・反転する可能性があります。"),
            ("捕食傾向比率", "{lab} 捕食傾向比率（0-1）", "捕食傾向と結びつくと、飢餓環境では有利、失敗コストが高い環境では不利に働く可能性があります。"),
            ("局所密度", "{lab} 平均局所密度（体/近傍）", "局所密度は交尾機会と過密コストの両方を生みます。"),
            ("足元資源", "{lab} 平均足元資源（単位/マス）", "足元資源が高い型は、空間配置か探索行動によって資源に接続できています。"),
        ]
        for lab in labels:
            if not has(f"{lab} 比率（0-1）"):
                continue
            rdelta = delta(f"{lab} 比率（0-1）")
            wavg = meanv(f"{lab} W", 1.0)
            with st.expander(f"{lab}：なぜ増えた/減ったのか", expanded=False):
                headline = []
                headline.append(f"初期比率 {pct(first(f'{lab} 比率（0-1）'))} → 最新比率 {pct(lastv(f'{lab} 比率（0-1）'))}、変化 {rdelta:+.3f}。平均Wは {wavg:.3f} です。")
                if rdelta > 0.03 and wavg >= 1.0:
                    headline.append("この型は、頻度変化と平均Wの両方から見て、実際にコピー維持・増加の方向にあります。")
                elif rdelta > 0.03 and wavg < 1.0:
                    headline.append("頻度は上がっていますが平均Wは1未満です。これは『全体が縮む中で相対的に残った』可能性があり、絶対的な増殖とは分けて読む必要があります。")
                elif rdelta < -0.03:
                    headline.append("頻度が下がっているため、この環境条件では相対的に不利な圧を受けています。どの圧かは下の連動指標で見ます。")
                else:
                    headline.append("頻度変化は小さく、強い選択圧よりも中立に近い、または複数の圧が釣り合っている可能性があります。")
                st.markdown("\n\n".join(headline))

                driver_rows = []
                target_ratio = f"{lab} 比率（0-1）"
                target_w = f"{lab} W"
                for name, tmpl, why_base in driver_candidates:
                    col = tmpl.format(lab=lab)
                    if not has(col):
                        continue
                    cr = corr(target_ratio, col)
                    cw = corr(target_w, col) if has(target_w) else np.nan
                    val_mean = meanv(col)
                    val_delta = delta(col)
                    strength = 0.0
                    if not np.isnan(cr): strength += abs(cr)
                    if not np.isnan(cw): strength += 0.6 * abs(cw)
                    if strength <= 0:
                        continue
                    sign_text = ""
                    if not np.isnan(cr):
                        if cr > 0.35:
                            sign_text = "比率上昇と同方向"
                        elif cr < -0.35:
                            sign_text = "比率上昇と逆方向"
                        else:
                            sign_text = "連動は弱め"
                    driver_rows.append({
                        "候補経路": name,
                        "平均値": val_mean,
                        "変化量": val_delta,
                        "比率との相関": cr,
                        "Wとの相関": cw,
                        "連動の強さ": strength,
                        "読み方": f"{sign_text}。{why_base}",
                    })
                if driver_rows:
                    drv = pd.DataFrame(driver_rows).sort_values("連動の強さ", ascending=False).head(8)
                    st.dataframe(drv, use_container_width=True, hide_index=True, column_config={
                        "平均値": st.column_config.NumberColumn(format="%.3f"),
                        "変化量": st.column_config.NumberColumn(format="%+.3f"),
                        "比率との相関": st.column_config.NumberColumn(format="%+.3f"),
                        "Wとの相関": st.column_config.NumberColumn(format="%+.3f"),
                        "連動の強さ": st.column_config.NumberColumn(format="%.3f"),
                    })
                    top = drv.iloc[0]
                    st.markdown(f"**最も疑うべき経路**：{top['候補経路']}。なぜなら、この型の比率またはWと最も強く連動しているからです。ただし、相関は原因の証明ではないので、比較実験実験でその経路を止めたときに同じ型の増減が変わるかを見ます。")
                else:
                    st.caption("この型については、十分な連動指標がまだ見つかりません。世代数を増やすと読める可能性があります。")

        # --- C: 親子フローの源流 ---
        st.markdown("#### C. 親子フロー：どの型が、どの親組み合わせからコピーを作ったか")
        pair_rows = []
        source_rows = []
        parent_rows = []
        for a in labels:
            for b in labels:
                col_src = f"コピー元→子 実出生:{a}→{b}（回/世代）"
                col_par = f"親→子 実出生:{a}→{b}（回/世代）"
                if has(col_src):
                    v = sumv(col_src)
                    if v > 0:
                        source_rows.append({"コピー元": a, "子": b, "合計": v})
                if has(col_par):
                    v = sumv(col_par)
                    if v > 0:
                        parent_rows.append({"親型": a, "子": b, "合計": v})
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i <= j:
                    col_pair = f"親組合せ 実出生:{a}×{b}（回/世代）"
                    if has(col_pair):
                        v = sumv(col_pair)
                        if v > 0:
                            pair_rows.append({"親組み合わせ": f"{a}×{b}", "同型ペア": "同型" if a == b else "混合", "実出生合計": v})
        c1, c2 = st.columns(2)
        with c1:
            if source_rows:
                srcdf = pd.DataFrame(source_rows).sort_values("合計", ascending=False)
                st.markdown("**コピー元→子**")
                st.dataframe(srcdf.head(15), use_container_width=True, hide_index=True)
                top_src = srcdf.iloc[0]
                explain_box("コピー元の読み方", f"最も多いコピー経路は **{top_src['コピー元']} → {top_src['子']}** です。これは、どの型のコピーが実際に子世代へ渡ったかを示します。ここが特定型に偏るほど、その型は死亡を避けるだけでなく、繁殖出口からコピーを増やしている可能性が高くなります。")
            else:
                st.caption("コピー元→子フローはまだ十分に記録されていません。")
        with c2:
            if pair_rows:
                pairdf = pd.DataFrame(pair_rows).sort_values("実出生合計", ascending=False)
                st.markdown("**親組み合わせ→実出生**")
                st.dataframe(pairdf.head(15), use_container_width=True, hide_index=True)
                mix_sum = float(pairdf.loc[pairdf["同型ペア"] == "混合", "実出生合計"].sum())
                same_sum = float(pairdf.loc[pairdf["同型ペア"] == "同型", "実出生合計"].sum())
                total_pair = max(mix_sum + same_sum, 1.0)
                explain_box("混合ペアと同型ペア", f"実出生のうち混合ペア由来は **{mix_sum/total_pair:.3f}**、同型ペア由来は **{same_sum/total_pair:.3f}** です。混合ペアが多いなら、型は孤立して淘汰されているのではなく、別型との交配の中でコピーが選ばれています。同型ペアが多いなら、空間配置や行動傾向によって同じ型同士が集まり、局所的な系統化が起きている可能性があります。")
            else:
                st.caption("親組み合わせフローはまだ十分に記録されていません。")

        # --- D: 淘汰圧ごとの受益/被害遺伝子 ---
        st.markdown("#### D. 淘汰圧ごとの受益型・被害型")
        pressure_rows = []
        if interaction_rows:
            base = pd.DataFrame(interaction_rows)
            def add_pressure(name, score_col, benefit_high=True, reason=""):
                if score_col not in base.columns:
                    return
                tmp = base[["型", score_col]].dropna().copy()
                if len(tmp) == 0:
                    return
                if benefit_high:
                    ben = tmp.sort_values(score_col, ascending=False).iloc[0]
                    hurt = tmp.sort_values(score_col, ascending=True).iloc[0]
                else:
                    ben = tmp.sort_values(score_col, ascending=True).iloc[0]
                    hurt = tmp.sort_values(score_col, ascending=False).iloc[0]
                pressure_rows.append({
                    "淘汰圧": name,
                    "有利に受けている型": ben["型"],
                    "不利に受けている型": hurt["型"],
                    "判定指標": score_col,
                    "なぜそう読めるか": reason,
                })
            add_pressure("資源獲得圧", "平均資源収支", True, "資源収支が高い型は、維持・移動・繁殖投資を支払いながら残りやすい。低い型は飢餓や繁殖停止へ接続しやすい。")
            add_pressure("飢餓/死亡圧", "平均空腹率", False, "空腹率が低い型は死亡圧を避けやすく、高い型は同じ出生力でも死亡でコピーを失いやすい。")
            add_pressure("繁殖出口圧", "出生-死亡", True, "出生-死亡が高い型は、親または子としてコピー数を直接増やせている。")
            add_pressure("争奪結合圧", "タカ比率差", True, "タカ比率差が高い型は争奪戦略と結合している。タカが有利な環境では伸び、戦闘コストが高い環境では逆に罰される。")
            add_pressure("捕食結合圧", "捕食傾向差", True, "捕食傾向が多い型は捕食成功が高い環境で有利だが、失敗コストが強い環境では不利になる。")
            add_pressure("チーム媒介圧", "赤偏り", True, "赤青の一方が空間配置・資源・局所密度で有利な場合、そのチームに偏った型が間接的に有利になる。")
        if pressure_rows:
            st.dataframe(pd.DataFrame(pressure_rows), use_container_width=True, hide_index=True)
            explain_box(
                "淘汰圧の読み方",
                "ここでいう淘汰圧は、『どの性質がコピー数を増減させる方向に働いたか』という候補です。資源獲得圧なら採取・資源収支が、繁殖出口圧なら交尾成功・実出生が、飢餓/死亡圧なら空腹率・死亡が、争奪結合圧ならタカ比率・戦闘が、捕食結合圧なら捕食傾向・捕食成功/失敗が見られます。重要なのは、**同じ哲学型でも、結合しているタカ/捕食/チーム遺伝子によって有利不利が変わる**という点です。"
            )

        # --- E: チーム差を遺伝子構成として読む ---
        if has("赤個体数（体）") and has("青個体数（体）"):
            st.markdown("#### E. 赤チーム・青チーム差：チームそのものか、内部遺伝子構成か")
            red_latest = lastv("赤個体数（体）")
            blue_latest = lastv("青個体数（体）")
            red_res = meanv("赤 平均所持資源") if has("赤 平均所持資源") else np.nan
            blue_res = meanv("青 平均所持資源") if has("青 平均所持資源") else np.nan
            red_g = meanv("赤 Gini") if has("赤 Gini") else np.nan
            blue_g = meanv("青 Gini") if has("青 Gini") else np.nan
            red_taka = meanv("赤タカ比率（0-1）") if has("赤タカ比率（0-1）") else np.nan
            blue_taka = meanv("青タカ比率（0-1）") if has("青タカ比率（0-1）") else np.nan
            comp_rows = []
            for lab in labels:
                rc = lastv(f"赤×{lab} 比率（赤内0-1）") if has(f"赤×{lab} 比率（赤内0-1）") else np.nan
                bc = lastv(f"青×{lab} 比率（青内0-1）") if has(f"青×{lab} 比率（青内0-1）") else np.nan
                if not np.isnan(rc) or not np.isnan(bc):
                    comp_rows.append({"型": lab, "赤内比率": rc, "青内比率": bc, "赤-青": rc-bc if not np.isnan(rc) and not np.isnan(bc) else np.nan})
            cols = st.columns(4)
            cols[0].metric("赤/青 最新個体数", f"{int(red_latest)}/{int(blue_latest)}")
            cols[1].metric("赤/青 平均資源", f"{fmt(red_res,2)}/{fmt(blue_res,2)}")
            cols[2].metric("赤/青 Gini", f"{fmt(red_g,3)}/{fmt(blue_g,3)}")
            cols[3].metric("赤/青 タカ比率", f"{fmt(red_taka,3)}/{fmt(blue_taka,3)}")
            if comp_rows:
                cdf = pd.DataFrame(comp_rows).sort_values("赤-青", ascending=False)
                st.dataframe(cdf, use_container_width=True, hide_index=True, column_config={
                    "赤内比率": st.column_config.NumberColumn(format="%.3f"),
                    "青内比率": st.column_config.NumberColumn(format="%.3f"),
                    "赤-青": st.column_config.NumberColumn(format="%+.3f"),
                })
                if red_latest > blue_latest * 1.08:
                    side = "赤"
                elif blue_latest > red_latest * 1.08:
                    side = "青"
                else:
                    side = "拮抗"
                top_comp = cdf.iloc[0] if len(cdf) else None
                bottom_comp = cdf.iloc[-1] if len(cdf) else None
                explain_box(
                    "チーム優位の因果候補",
                    f"最新個体数だけなら状態は **{side}** です。ただし、チーム優位はチーム色そのものの効果とは限りません。赤青で平均資源・Gini・タカ比率・哲学型構成が違えば、その差が媒介している可能性があります。現在、赤側に相対的に偏っている型は **{top_comp['型'] if top_comp is not None else '—'}**、青側に相対的に偏っている型は **{bottom_comp['型'] if bottom_comp is not None else '—'}** です。もし優位チームに特定の型やタカ/ハトが偏っているなら、チーム差は『色』ではなく、内部遺伝子構成と空間配置の結果として読むべきです。"
                )

        # --- F: 次に検証すべき操作 ---
        st.markdown("#### F. 次に疑うべき因果を、比較実験でどう潰すか")
        suggestions = []
        if meanv("交尾成立率（0-1）") < 0.25:
            suggestions.append({"疑う因果": "繁殖出口が詰まっている", "見るべき比較": "通常割合変更・近親回避OFF・密度依存OFF", "理由": "資源があっても交尾成立や空きマスで止まると、遺伝子差が出生へ出ません。"})
        if global_gini > 0.42 or (not np.isnan(corr("死亡数（体/世代）", "資源格差Gini（0-1）")) and corr("死亡数（体/世代）", "資源格差Gini（0-1）") > 0.35):
            suggestions.append({"疑う因果": "資源格差が死亡圧を作っている", "見るべき比較": "局所資源再生OFF/ON・資源量変更", "理由": "総資源ではなく分配の偏りが死亡を作ると、局所配置に強い型だけが残ります。"})
        if meanv("捕食試行（回/世代）") > 0:
            suggestions.append({"疑う因果": "捕食が有利/不利を媒介している", "見るべき比較": "捕食OFF", "理由": "捕食傾向を消したときに優勢型が変わるなら、哲学型の差は捕食遺伝子によって媒介されています。"})
        if meanv("過密で抑制された出生候補（回/世代）") > 0.5:
            suggestions.append({"疑う因果": "密度依存が出生選択を作っている", "見るべき比較": "密度依存OFF", "理由": "過密抑制が強いと、単に資源を持つ型ではなく、空間的に分散する型が有利になります。"})
        if not suggestions:
            suggestions.append({"疑う因果": "単独で強い圧はまだ見えにくい", "見るべき比較": "seed反復数を増やす・世代数を増やす", "理由": "複数の弱い圧が合成されている可能性があります。再現性を見るのが先です。"})
        st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)


    def show_v22_environment_gene_report():
        """v22：環境と遺伝子の結びつきを中心にした、読者向けの深い因果レポート。"""
        if df is None or len(df) < 4:
            st.info("環境-遺伝子レポートには、もう少し世代履歴が必要です。まず数十世代ほど進めてください。")
            return

        d = df.copy()

        def has(col):
            return col in d.columns

        def ser(col):
            if col not in d.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(d[col], errors="coerce")

        def first(col, default=np.nan):
            ss = ser(col).dropna()
            return float(ss.iloc[0]) if len(ss) else float(default)

        def lastv(col, default=np.nan):
            ss = ser(col).dropna()
            return float(ss.iloc[-1]) if len(ss) else float(default)

        def meanv(col, default=np.nan):
            ss = ser(col).dropna()
            return float(ss.mean()) if len(ss) else float(default)

        def sumv(col, default=0.0):
            ss = ser(col).dropna()
            return float(ss.sum()) if len(ss) else float(default)

        def slopev(col):
            ss = ser(col).dropna()
            if len(ss) < 4:
                return 0.0
            try:
                x = np.arange(len(ss), dtype=float)
                return float(np.polyfit(x, ss.to_numpy(dtype=float), 1)[0])
            except Exception:
                return 0.0

        def corr(a, b):
            if a not in d.columns or b not in d.columns:
                return np.nan
            aa = ser(a); bb = ser(b)
            m = aa.notna() & bb.notna()
            if int(m.sum()) < 5:
                return np.nan
            if float(aa[m].std()) == 0.0 or float(bb[m].std()) == 0.0:
                return np.nan
            try:
                return float(aa[m].corr(bb[m]))
            except Exception:
                return np.nan

        def fmt(x, nd=3, signed=False):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            return f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"

        def short_dir(x, eps=0.03):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "不明"
            if x > eps:
                return "増加"
            if x < -eps:
                return "減少"
            return "ほぼ維持"

        def rel_desc(x, high=0.08, low=-0.08):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "不明"
            if x > high:
                return "平均より高い"
            if x < low:
                return "平均より低い"
            return "平均付近"

        def evidence_strength(items):
            score = sum(1 for v in items if bool(v))
            if score >= 4:
                return "強い候補"
            if score >= 2:
                return "中程度の候補"
            return "弱い候補"

        st.markdown("### 環境-遺伝子統合レポート：何が、どこで、なぜ選ばれたか")
        st.markdown(
            "この欄は、単に『どの遺伝子が増えたか』ではなく、**環境の状態**と**遺伝子の行動傾向**を結びつけて読みます。"
            "このモデルでは、遺伝子は直接コピー数を増やすのではなく、行動評価を少し変えます。その行動評価が、資源のある場所へ行けるか、過密を避けられるか、交尾相手に出会えるか、捕食や争奪のリスクを取るかを変えます。"
            "したがって、本当に見るべき因果鎖は **遺伝子 → 行動 → 環境との接触 → 資源/危険/繁殖機会 → 出生・死亡 → W** です。"
        )

        labels = list(PHILO_LABELS.values())
        total_pop0 = first("個体数（体）", 0.0)
        total_pop1 = lastv("個体数（体）", 0.0)
        total_pop_slope = slopev("個体数（体）")
        w_mean = meanv("個体群全体W（増殖率）", 1.0)
        w_late = float(ser("個体群全体W（増殖率）").tail(max(3, len(d)//5)).mean()) if has("個体群全体W（増殖率）") else np.nan
        res0 = first("資源総量（単位）", 0.0)
        res1 = lastv("資源総量（単位）", 0.0)
        res_slope = slopev("資源総量（単位）")
        gini_mean = meanv("資源格差Gini（0-1）", 0.0)
        density_mean = meanv("平均局所密度（体/近傍）", 0.0)
        mate_mean = meanv("交尾成立率（0-1）", np.nan)
        kin_mean = meanv("近親交配回避（回/世代）", 0.0)
        density_block_mean = meanv("過密で抑制された出生候補（回/世代）", 0.0)
        pred_success_mean = meanv("捕食成功率（0-1）", np.nan)
        pred_try_mean = meanv("捕食試行（回/世代）", 0.0)
        birth_total = sumv("出生数（体/世代）", 0.0)
        death_total = sumv("死亡数（体/世代）", 0.0)
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_birth_mate = corr("出生数（体/世代）", "交尾成立率（0-1）")
        c_birth_res = corr("出生数（体/世代）", "平均所持資源（単位/体）")

        # --- 遺伝子ごとの詳細プロファイル ---
        profile_rows = []
        for lab in labels:
            ratio_col = f"{lab} 比率（0-1）"
            count_col = f"{lab} 数（体）"
            if not has(ratio_col) and not has(count_col):
                continue
            r0 = first(ratio_col, np.nan)
            r1 = lastv(ratio_col, np.nan)
            c0 = first(count_col, np.nan)
            c1 = lastv(count_col, np.nan)
            wavg = meanv(f"{lab} W", 1.0)
            wlast = lastv(f"{lab} W", 1.0)
            net = meanv(f"{lab} 資源収支ネット（単位/世代）", 0.0)
            hunger = meanv(f"{lab} 空腹個体比率（0-1）", np.nan)
            under = meanv(f"{lab} 平均足元資源（単位/マス）", np.nan)
            local_den = meanv(f"{lab} 平均局所密度（体/近傍）", np.nan)
            births = sumv(f"{lab} 実出生（体/世代）", 0.0)
            deaths = sumv(f"{lab} 死亡（体/世代）", 0.0)
            parent_births = sumv(f"{lab} 親参加:実出生（回/世代）", 0.0)
            mate_success = sumv(f"{lab} 交尾成功参加（回/世代）", 0.0)
            gather_rate = meanv(f"{lab} 行動率:採取（0-1）", np.nan)
            move_rate = meanv(f"{lab} 行動率:移動（0-1）", np.nan)
            mate_rate = meanv(f"{lab} 行動率:交尾（0-1）", np.nan)
            escape_rate = meanv(f"{lab} 行動率:回避（0-1）", np.nan)
            battle_rate = meanv(f"{lab} 行動率:戦闘（0-1）", np.nan)
            pred_rate = meanv(f"{lab} 行動率:捕食（0-1）", np.nan)
            taka_ratio = meanv(f"{lab} タカ比率（0-1）", np.nan)
            pred_gene = meanv(f"{lab} 捕食傾向比率（0-1）", np.nan)
            red_frac = meanv(f"{lab} 赤比率（型内0-1）", np.nan)
            # exposureで割った近似率。実数ではなく比較用。
            exposure = max(sumv(count_col, 0.0), 1.0)
            birth_rate = births / exposure
            death_rate = deaths / exposure
            parent_rate = parent_births / exposure
            advantage_score = 0.0
            if not np.isnan(r0) and not np.isnan(r1): advantage_score += (r1 - r0) * 3.0
            advantage_score += (wavg - 1.0) * 1.6
            advantage_score += birth_rate * 8.0
            advantage_score -= death_rate * 8.0
            advantage_score += max(net, -5.0) * 0.02
            if not np.isnan(hunger): advantage_score -= hunger * 0.4
            if not np.isnan(mate_success): advantage_score += mate_success / max(exposure, 1.0) * 3.0
            profile_rows.append({
                "型": lab,
                "初期数": c0,
                "最新数": c1,
                "数変化": c1-c0 if not np.isnan(c0) and not np.isnan(c1) else np.nan,
                "初期比率": r0,
                "最新比率": r1,
                "比率変化": r1-r0 if not np.isnan(r0) and not np.isnan(r1) else np.nan,
                "平均W": wavg,
                "最新W": wlast,
                "出生率近似": birth_rate,
                "死亡率近似": death_rate,
                "親出生率近似": parent_rate,
                "資源収支": net,
                "空腹率": hunger,
                "足元資源": under,
                "局所密度": local_den,
                "採取率": gather_rate,
                "移動率": move_rate,
                "交尾率": mate_rate,
                "回避率": escape_rate,
                "戦闘率": battle_rate,
                "捕食率": pred_rate,
                "タカ比率": taka_ratio,
                "捕食傾向": pred_gene,
                "赤比率": red_frac,
                "総合優勢スコア": advantage_score,
            })

        prof = pd.DataFrame(profile_rows)
        if prof.empty:
            st.warning("v22で読むための行動型列が見つかりません。")
            return
        prof = prof.sort_values("総合優勢スコア", ascending=False)
        strongest = prof.iloc[0]
        weakest = prof.iloc[-1]

        # --- 全体環境の診断 ---
        env_rows = []
        env_rows.append({
            "環境・圧": "個体群維持圧",
            "観察値": f"個体数 {total_pop0:.0f}→{total_pop1:.0f}、平均W {fmt(w_mean)}、終盤W {fmt(w_late)}",
            "何を意味するか": "Wが1を下回るほど、環境全体が出生より死亡・消耗を強くしている。1付近なら、全体崩壊より内部遺伝子差が見えやすい。",
            "遺伝子への作用": "低W環境では、短期増殖型よりも死亡を避ける型・資源を安定確保する型が残りやすい。高W環境では繁殖へ早く接続する型が伸びやすい。",
            "根拠の強さ": evidence_strength([abs(w_mean-1)>0.02, abs(total_pop_slope)>0.05, abs(total_pop1-total_pop0)>max(5,total_pop0*0.05)])
        })
        env_rows.append({
            "環境・圧": "資源総量圧",
            "観察値": f"資源 {res0:.0f}→{res1:.0f}、傾き {fmt(res_slope,4,signed=True)}、個体数との相関 {fmt(c_pop_res,3,signed=True)}",
            "何を意味するか": "資源総量が増えても個体が増えないなら、資源が盤面にあるだけで、探索・採取・繁殖に接続していない可能性がある。",
            "遺伝子への作用": "資源が局所的なら、視界・移動・採取判断に強い型が有利。資源は多いのに空腹が高い型は、環境資源をコピーへ変換できていない。",
            "根拠の強さ": evidence_strength([abs(res_slope)>0.5, not np.isnan(c_pop_res) and abs(c_pop_res)>0.35, res1>res0 and total_pop1<total_pop0])
        })
        env_rows.append({
            "環境・圧": "資源格差圧",
            "観察値": f"平均Gini {fmt(gini_mean)}、死亡との相関 {fmt(c_death_gini,3,signed=True)}",
            "何を意味するか": "総量ではなく分配が偏ると、一部個体だけが余剰を持ち、低資源個体が維持コストで削られる。",
            "遺伝子への作用": "足元資源・資源収支が高い型は利益を受ける。採取率が高いのに空腹な型は、局所枯渇や移動コストに負けている可能性がある。",
            "根拠の強さ": evidence_strength([gini_mean>0.35, not np.isnan(c_death_gini) and c_death_gini>0.3])
        })
        env_rows.append({
            "環境・圧": "繁殖出口圧",
            "観察値": f"出生合計 {birth_total:.0f}、死亡合計 {death_total:.0f}、交尾成立率平均 {fmt(mate_mean)}、出生×交尾相関 {fmt(c_birth_mate,3,signed=True)}、出生×資源相関 {fmt(c_birth_res,3,signed=True)}",
            "何を意味するか": "資源を持っていても、相手・空きマス・近親回避・密度抑制を突破できなければ出生にならない。",
            "遺伝子への作用": "交尾率・親出生率が高い型がコピーを直接伸ばす。交尾率が高いのに実出生が少ない型は、環境側の出口で詰まっている。",
            "根拠の強さ": evidence_strength([not np.isnan(mate_mean) and mate_mean<0.35, kin_mean>0.2, density_block_mean>0.5, not np.isnan(c_birth_mate) and abs(c_birth_mate)>0.35])
        })
        env_rows.append({
            "環境・圧": "局所密度・空間圧",
            "観察値": f"平均局所密度 {fmt(density_mean)}、過密出生抑制 {fmt(density_block_mean)}、近親回避 {fmt(kin_mean)}",
            "何を意味するか": "同じ資源量でも、個体が密集すると出生空間が減り、相手が近親なら交尾候補が減る。これは繁殖能力ではなく空間配置による淘汰圧。",
            "遺伝子への作用": "移動・回避・分散的行動が有利になる場合がある。一方で動きすぎる型は移動コストで不利になる。",
            "根拠の強さ": evidence_strength([density_mean>2.5, density_block_mean>0.5, kin_mean>0.2])
        })
        env_rows.append({
            "環境・圧": "捕食/争奪圧",
            "観察値": f"捕食試行平均 {fmt(pred_try_mean)}、捕食成功率 {fmt(pred_success_mean)}、タカ比率変化 {fmt(lastv('タカ比率（0-1）',0)-first('タカ比率（0-1）',0),3,signed=True)}",
            "何を意味するか": "捕食・争奪は資源不足を短期的に補うが、失敗や戦闘コストで死亡・消耗を増やす。成功率が低いなら罰、成功率が高いなら救済になる。",
            "遺伝子への作用": "タカや捕食傾向を多く持つ哲学型は、この圧によって有利にも不利にも振れる。ここを見ないと『哲学型の強さ』と『攻撃/捕食遺伝子の強さ』が混ざる。",
            "根拠の強さ": evidence_strength([pred_try_mean>0.2, not np.isnan(pred_success_mean) and (pred_success_mean<0.25 or pred_success_mean>0.55), abs(lastv('タカ比率（0-1）',0)-first('タカ比率（0-1）',0))>0.05])
        })
        if has("赤個体数（体）") and has("青個体数（体）"):
            red_delta = lastv("赤個体数（体）",0)-first("赤個体数（体）",0)
            blue_delta = lastv("青個体数（体）",0)-first("青個体数（体）",0)
            env_rows.append({
                "環境・圧": "赤青チーム媒介圧",
                "観察値": f"赤変化 {red_delta:+.0f}、青変化 {blue_delta:+.0f}、最終赤青差 {lastv('赤個体数（体）',0)-lastv('青個体数（体）',0):+.0f}",
                "何を意味するか": "チーム差が出ると、哲学型の優劣がチーム偏りで見かけ上変わる。強いチームに偏った型は、その型自体が強くなくても残る。",
                "遺伝子への作用": "各哲学型の赤比率・青比率を見て、チーム差が哲学型の増減を媒介していないか確認する必要がある。",
                "根拠の強さ": evidence_strength([abs(red_delta-blue_delta)>10, abs(lastv('赤個体数（体）',0)-lastv('青個体数（体）',0))>10])
            })

        st.markdown("#### A. 環境が作っている淘汰圧：遺伝子はどの環境に押されているか")
        st.dataframe(pd.DataFrame(env_rows), use_container_width=True, hide_index=True)

        st.markdown("#### B. 遺伝子別プロファイル：コピー数だけでなく、環境との接触を見る")
        st.caption("総合優勢スコアは説明補助です。比率変化・W・出生/死亡・資源収支・空腹率を合成した暫定値で、絶対的な適応度ではありません。")
        st.dataframe(prof, use_container_width=True, hide_index=True, column_config={
            "初期数": st.column_config.NumberColumn(format="%.0f"),
            "最新数": st.column_config.NumberColumn(format="%.0f"),
            "数変化": st.column_config.NumberColumn(format="%+.0f"),
            "初期比率": st.column_config.NumberColumn(format="%.3f"),
            "最新比率": st.column_config.NumberColumn(format="%.3f"),
            "比率変化": st.column_config.NumberColumn(format="%+.3f"),
            "平均W": st.column_config.NumberColumn(format="%.3f"),
            "最新W": st.column_config.NumberColumn(format="%.3f"),
            "出生率近似": st.column_config.NumberColumn(format="%.4f"),
            "死亡率近似": st.column_config.NumberColumn(format="%.4f"),
            "親出生率近似": st.column_config.NumberColumn(format="%.4f"),
            "資源収支": st.column_config.NumberColumn(format="%+.2f"),
            "空腹率": st.column_config.NumberColumn(format="%.3f"),
            "足元資源": st.column_config.NumberColumn(format="%.2f"),
            "局所密度": st.column_config.NumberColumn(format="%.2f"),
            "採取率": st.column_config.NumberColumn(format="%.3f"),
            "移動率": st.column_config.NumberColumn(format="%.3f"),
            "交尾率": st.column_config.NumberColumn(format="%.3f"),
            "回避率": st.column_config.NumberColumn(format="%.3f"),
            "戦闘率": st.column_config.NumberColumn(format="%.3f"),
            "捕食率": st.column_config.NumberColumn(format="%.3f"),
            "タカ比率": st.column_config.NumberColumn(format="%.3f"),
            "捕食傾向": st.column_config.NumberColumn(format="%.3f"),
            "赤比率": st.column_config.NumberColumn(format="%.3f"),
            "総合優勢スコア": st.column_config.NumberColumn(format="%+.3f"),
        })

        # --- 深い文章サマリー ---
        st.markdown("#### C. 長文サマリー：環境と遺伝子の因果関係")
        main_lines = []
        main_lines.append(f"このランでは、全体個体数は **{total_pop0:.0f}体から{total_pop1:.0f}体** へ変化しました。平均Wは **{fmt(w_mean)}**、終盤Wは **{fmt(w_late)}** です。これは、まず集団全体が増殖環境なのか縮小環境なのかを示します。Wが1付近なら、全体崩壊よりも内部の遺伝子差が読みやすく、Wが1から大きく外れるなら、特定遺伝子の差より環境全体の圧が強い可能性があります。")
        main_lines.append(f"資源環境を見ると、資源総量は **{res0:.0f}→{res1:.0f}** で、個体数との相関は **{fmt(c_pop_res,3,signed=True)}** です。ここが重要なのは、資源が多いことと、個体がその資源を使って子を残せることは別だからです。資源が増えているのに個体数や出生が伸びない場合、問題は資源量ではなく、資源への到達、局所配置、移動コスト、交尾相手、空きマスにあります。")
        main_lines.append(f"資源格差Giniの平均は **{fmt(gini_mean)}** で、死亡との相関は **{fmt(c_death_gini,3,signed=True)}** です。Giniが高く死亡と正に連動する場合、淘汰圧は『全員が少しずつ貧しい』のではなく、『資源を取れない個体が集中的に削られる』形になります。この場合、足元資源が高い型、局所資源を拾える型、移動コストを払いすぎない型が有利になります。")
        main_lines.append(f"繁殖出口では、交尾成立率平均が **{fmt(mate_mean)}**、近親回避平均が **{fmt(kin_mean)}**、過密による出生抑制平均が **{fmt(density_block_mean)}** です。これは、どれだけ資源を持っていても、相手・距離・血縁・空きマスの条件を満たさなければコピーにならないことを意味します。したがって、親参加実出生が多い型は繁殖出口を突破しており、交尾率が高いのに親出生が少ない型は、環境側の出口で詰まっています。")
        main_lines.append(f"捕食と争奪については、捕食試行平均 **{fmt(pred_try_mean)}**、捕食成功率 **{fmt(pred_success_mean)}** です。捕食やタカ的争奪は、資源不足の局面では短期救済になりますが、成功率が低い環境では失敗コストとして作用します。したがって、捕食傾向やタカ比率が多い型が増えたとしても、それは哲学型そのものの優位ではなく、捕食/争奪遺伝子との結合効果かもしれません。")
        st.markdown("\n\n".join(main_lines))

        # --- 各型の因果文章 ---
        st.markdown("#### D. 各型ごとの深い読み取り：その型は、どの環境で、何によって増減したのか")
        global_taka = meanv("タカ比率（0-1）", np.nan)
        global_pred = meanv("捕食傾向比率（0-1）", np.nan)
        global_hunger = meanv("空腹個体比率（0-1）", np.nan)
        global_under = meanv("平均足元資源（単位/マス）", np.nan)
        global_local_den = meanv("平均局所密度（体/近傍）", np.nan)
        for _, r in prof.iterrows():
            lab = str(r["型"])
            with st.expander(f"{lab}：環境との結びつき・遺伝子相互作用・因果候補", expanded=(lab in [str(strongest['型']), str(weakest['型'])])):
                lines = []
                lines.append(f"**観察事実。** {lab} は初期比率 **{fmt(r.get('初期比率'))}** から最新比率 **{fmt(r.get('最新比率'))}** へ変化し、比率変化は **{fmt(r.get('比率変化'),3, signed=True)}** です。平均Wは **{fmt(r.get('平均W'))}**、最新Wは **{fmt(r.get('最新W'))}** です。ここまでは実際のログから読める頻度変化です。")
                # environment contact
                contact_bits = []
                if not np.isnan(r.get("足元資源", np.nan)) and not np.isnan(global_under):
                    contact_bits.append(f"足元資源は全体平均に対して **{rel_desc(r['足元資源']-global_under, 0.5, -0.5)}** です")
                if not np.isnan(r.get("局所密度", np.nan)) and not np.isnan(global_local_den):
                    contact_bits.append(f"局所密度は全体平均に対して **{rel_desc(r['局所密度']-global_local_den, 0.5, -0.5)}** です")
                if not np.isnan(r.get("空腹率", np.nan)) and not np.isnan(global_hunger):
                    contact_bits.append(f"空腹率は全体平均に対して **{rel_desc(r['空腹率']-global_hunger, 0.05, -0.05)}** です")
                if contact_bits:
                    lines.append("**環境との接触。** " + "、".join(contact_bits) + "。これは、その型が同じ盤面でもどの資源・密度・飢餓条件に置かれていたかを示します。遺伝子は環境から独立して働くのではなく、こうした局所条件を通って初めて有利/不利になります。")
                # behavior route
                behaviors = {
                    "採取": r.get("採取率", np.nan),
                    "移動": r.get("移動率", np.nan),
                    "交尾": r.get("交尾率", np.nan),
                    "回避": r.get("回避率", np.nan),
                    "戦闘": r.get("戦闘率", np.nan),
                    "捕食": r.get("捕食率", np.nan),
                }
                valid_beh = {k:v for k,v in behaviors.items() if not (isinstance(v,float) and np.isnan(v))}
                if valid_beh:
                    top_beh = sorted(valid_beh.items(), key=lambda kv: kv[1], reverse=True)[:2]
                    lines.append("**行動経路。** 行動率では " + "、".join([f"{k} {fmt(v)}" for k,v in top_beh]) + f" が目立ちます。資源収支は **{fmt(r.get('資源収支'),2,signed=True)}**、出生率近似は **{fmt(r.get('出生率近似'),4)}**、死亡率近似は **{fmt(r.get('死亡率近似'),4)}** です。つまり、この型の増減は、単なる性格名ではなく、行動が資源・死亡・繁殖出口にどう変換されたかで読む必要があります。")
                # gene interaction
                interaction = []
                if not np.isnan(r.get("タカ比率", np.nan)) and not np.isnan(global_taka):
                    diff = r["タカ比率"] - global_taka
                    if diff > 0.08:
                        interaction.append("タカ遺伝子が平均より多く、争奪利得または戦闘コストの影響を受けやすい")
                    elif diff < -0.08:
                        interaction.append("ハト寄りで、戦闘回避・コスト回避の影響を受けやすい")
                if not np.isnan(r.get("捕食傾向", np.nan)) and not np.isnan(global_pred):
                    diff = r["捕食傾向"] - global_pred
                    if diff > 0.04:
                        interaction.append("捕食傾向が平均より多く、捕食成功率/失敗コストに結果を左右されやすい")
                    elif diff < -0.04:
                        interaction.append("非捕食寄りで、捕食の短期利得より安全性に寄りやすい")
                if not np.isnan(r.get("赤比率", np.nan)):
                    if r["赤比率"] > 0.58:
                        interaction.append("赤チームに偏っており、赤チームの資源・空間条件を媒介している可能性がある")
                    elif r["赤比率"] < 0.42:
                        interaction.append("青チームに偏っており、青チームの資源・空間条件を媒介している可能性がある")
                if interaction:
                    lines.append("**遺伝子間作用。** " + "。".join(interaction) + "。したがって、この型の優勢/劣勢を思想型単体へ帰属するのは危険です。タカ/ハト、捕食傾向、チーム偏りが、哲学型の効果を増幅または反転させている可能性があります。")
                # candidate cause classification
                support = []
                if r.get("比率変化", 0) > 0.03 and r.get("平均W", 1) >= 1.0: support.append("比率上昇とWが同じ方向")
                if r.get("出生率近似", 0) > r.get("死亡率近似", 0): support.append("出生率が死亡率を上回る")
                if r.get("資源収支", 0) > 0 and (np.isnan(r.get("空腹率", np.nan)) or r.get("空腹率", 1) < 0.35): support.append("資源収支と空腹率が有利")
                if r.get("親出生率近似", 0) > 0: support.append("親としてコピー産出に参加")
                if r.get("比率変化", 0) < -0.03 and r.get("平均W", 1) < 1.0: support.append("比率低下とW低下が同じ方向")
                strength = evidence_strength(support)
                if support:
                    lines.append(f"**因果候補の強さ。** この型については **{strength}** です。根拠は「" + "、".join(support) + "」です。ただし、これは単独ラン内の推定なので、最終確認には比較実験やseed反復が必要です。")
                else:
                    lines.append("**因果候補の強さ。** まだ弱いです。比率・W・出生死亡・資源収支の方向がそろっていないため、ドリフト、初期配置、チーム偏り、他遺伝子との結合で説明できる可能性があります。")
                st.markdown("\n\n".join(lines))

        # --- 淘汰圧ごとの作用対象 ---
        st.markdown("#### E. 淘汰圧ごとの作用対象：どの圧が、どの遺伝子を押したか")
        pressure_rows = []
        def top_by(col, ascending=False, n=2):
            if col not in prof.columns:
                return "—"
            tmp = prof[["型", col]].dropna().sort_values(col, ascending=ascending).head(n)
            return "、".join([f"{rr['型']}({fmt(rr[col],2)})" for _, rr in tmp.iterrows()]) if len(tmp) else "—"
        pressure_rows.append({
            "淘汰圧": "資源アクセス圧",
            "利益を受けやすい型": top_by("資源収支", ascending=False),
            "被害を受けやすい型": top_by("空腹率", ascending=False),
            "因果説明": "資源収支が高い型は、盤面資源を個体内資源へ変換できている。空腹率が高い型は、資源があっても到達・採取・維持コストで負けている。"
        })
        pressure_rows.append({
            "淘汰圧": "繁殖出口圧",
            "利益を受けやすい型": top_by("親出生率近似", ascending=False),
            "被害を受けやすい型": top_by("交尾率", ascending=False),
            "因果説明": "親出生率が高い型は実際にコピーを作っている。交尾率だけ高く親出生率が低い型は、相手・空きマス・近親回避・資源不足で出口が詰まっている可能性がある。"
        })
        pressure_rows.append({
            "淘汰圧": "死亡/飢餓圧",
            "利益を受けやすい型": top_by("死亡率近似", ascending=True),
            "被害を受けやすい型": top_by("死亡率近似", ascending=False),
            "因果説明": "死亡率が低い型は、同じ環境でも維持コスト・空腹・争奪/捕食リスクを避けている可能性がある。ただし死亡率だけでは増殖力は分からないため出生率と併読する。"
        })
        pressure_rows.append({
            "淘汰圧": "争奪/タカハト媒介圧",
            "利益を受けやすい型": top_by("タカ比率", ascending=False),
            "被害を受けやすい型": top_by("タカ比率", ascending=True),
            "因果説明": "タカ比率が高い型が伸びるなら争奪利得が効いている可能性、ハト寄りが伸びるなら戦闘コスト回避が効いている可能性がある。どちらが正しいかは戦闘獲得/損失とWで確認する。"
        })
        pressure_rows.append({
            "淘汰圧": "捕食媒介圧",
            "利益を受けやすい型": top_by("捕食傾向", ascending=False),
            "被害を受けやすい型": top_by("捕食率", ascending=False),
            "因果説明": "捕食傾向が高い型は、捕食成功率が高い環境では利益を得るが、成功率が低い環境では失敗コストで削られる。捕食率が高いのにWが低い型は、捕食が罰になっている疑いがある。"
        })
        st.dataframe(pd.DataFrame(pressure_rows), use_container_width=True, hide_index=True)

        # --- 事実・解釈・未確定の分離 ---
        st.markdown("#### F. 事実・強い候補・未確定を分ける")
        claims = []
        claims.append({"区分":"観察事実", "内容":f"最も総合優勢スコアが高い型は {strongest['型']}、最も低い型は {weakest['型']}。これはログから算出した事実だが、スコア式は分析補助である。"})
        claims.append({"区分":"強い候補", "内容":f"{strongest['型']} が伸びた理由は、比率変化・W・出生死亡・資源収支のうち複数が同方向なら強くなる。表Bと各型の詳細欄で、どの経路がそろっているか確認する。"})
        claims.append({"区分":"弱い候補", "内容":"タカ比率・捕食傾向・赤青偏りが強い場合、哲学型そのものではなく、同伴遺伝子やチーム環境が結果を媒介している可能性がある。"})
        claims.append({"区分":"未確定", "内容":"単独ランだけでは、初期配置・資源配置・遺伝的浮動を完全に排除できない。比較実験実験で、哲学遺伝子OFF、捕食OFF、密度依存OFF、通常100%などを同seedで比較する必要がある。"})
        st.dataframe(pd.DataFrame(claims), use_container_width=True, hide_index=True)

        # --- 次に見るべき比較 ---
        st.markdown("#### G. 次に潰すべき別解釈")
        next_rows = []
        if pred_try_mean > 0.1:
            next_rows.append({"別解釈": "哲学型差ではなく捕食傾向差である", "検証": "比較実験で捕食OFFを実行し、優勢型が変わるか見る", "理由": "捕食傾向が型に偏っていると、哲学型の効果と捕食遺伝子の効果が混ざるため。"})
        if density_block_mean > 0.2 or density_mean > 2.5:
            next_rows.append({"別解釈": "思想型差ではなく局所密度・空間配置差である", "検証": "密度依存OFF、局所資源再生OFFを比較", "理由": "過密や局所資源が強いと、行動方針よりも配置が結果を左右するため。"})
        if kin_mean > 0.1:
            next_rows.append({"別解釈": "繁殖力差ではなく近親回避制約である", "検証": "近親回避OFFを比較", "理由": "近親回避は生物学的には自然だが、小集団では交尾成立を強く抑えるため。"})
        if has("赤個体数（体）") and abs((lastv("赤個体数（体）",0)-lastv("青個体数（体）",0))) > max(8, total_pop1*0.08):
            next_rows.append({"別解釈": "哲学型差ではなく赤青チーム環境差である", "検証": "赤青別の哲学型構成、赤青別平均資源、赤青別タカ比率を見る", "理由": "強いチームに偏った型は、その型自身が強くなくても増えたように見えるため。"})
        next_rows.append({"別解釈": "たまたまそのseedで起きた遺伝的浮動である", "検証": "比較実験のseed反復数を増やす", "理由": "小集団・局所相互作用モデルでは、偶然の出生死亡が遺伝子頻度を大きく動かすため。"})
        st.dataframe(pd.DataFrame(next_rows), use_container_width=True, hide_index=True)

    def show_v20_comparison_mode():
        """v20：同じseedで条件だけを変え、観察された差分を因果候補として読む。"""
        st.markdown("### 比較実験モード：同じseedで条件だけを変える")
        explain_box(
            "なぜ比較実験が必要か",
            "一つのランだけでは、遺伝子頻度の変化が本当にその遺伝子の効果なのか、初期配置・資源配置・偶然の出生死亡で起きたのかを分けにくいです。比較実験では、同じseedを使ったまま一つの条件だけを変えて走らせます。すると、初期配置の偶然をかなりそろえたうえで、哲学遺伝子・通常個体割合・捕食・密度依存・局所資源再生・近親回避がどの程度結果を変えたかを比較できます。これは因果の証明そのものではありませんが、単なる相関よりずっと強い因果候補になります。"
        )

        scenario_options = {
            "哲学遺伝子OFF": {"enable_philo_gene": False},
            "通常100%（哲学個体なし）": {"initial_normal_pct": 100},
            "通常50%": {"initial_normal_pct": 50},
            "哲学100%（通常個体なし）": {"initial_normal_pct": 0},
            "捕食OFF": {"enable_predation": False},
            "密度依存OFF": {"enable_density_dependence": False},
            "局所資源再生OFF": {"enable_local_resource_regen": False},
            "近親回避OFF": {"enable_kin_avoidance": False},
            "生態補正OFF（密度・局所・近親・捕食OFF）": {
                "enable_density_dependence": False,
                "enable_local_resource_regen": False,
                "enable_kin_avoidance": False,
                "enable_predation": False,
            },
        }

        with st.expander("比較実験を実行する", expanded=False):
            st.caption("比較中は一時的に内部世界をリセットして複数ランを回します。終了後、現在見ている世界・履歴・世代は元に戻します。")
            comp_cols = st.columns([1.0, 1.0, 2.2])
            with comp_cols[0]:
                compare_generations = st.slider("比較で進める世代数", 10, 120, 40, 10, key="v20_compare_generations")
            with comp_cols[1]:
                compare_repeats = st.slider("seed反復数", 1, 3, 1, 1, key="v20_compare_repeats")
            with comp_cols[2]:
                selected_scenarios = st.multiselect(
                    "比較する条件",
                    options=list(scenario_options.keys()),
                    default=["哲学遺伝子OFF", "通常100%（哲学個体なし）", "捕食OFF"],
                    key="v20_selected_scenarios",
                )
            run_compare = st.button("同じseedで比較実験を実行", use_container_width=True, key="v20_run_compare")
            st.caption("軽量化のため初期値は40世代×1反復・条件3つにしています。必要なときだけ世代数・条件・反復数を増やしてください。")

            if run_compare:
                specs = [("基準", {})] + [(name, scenario_options[name]) for name in selected_scenarios]
                total_runs = len(specs) * int(compare_repeats)
                if total_runs > 18:
                    st.warning(f"比較条件が多く、{total_runs}本の内部ランになります。Cloudでは重くなりやすいので、条件数か反復数を減らすのがおすすめです。")
                try:
                    with st.spinner("比較実験実験を実行中です。現在の世界はあとで復元します。"):
                        result = _run_v20_comparison(specs, int(compare_generations), int(compare_repeats))
                        st.session_state["v20_compare_results"] = result
                except Exception as e:
                    st.error("比較実験中にエラーが出ました。通常の単独シミュレーションはそのまま使えます。")
                    st.caption("まずは『比較で進める世代数』を20〜40、seed反復数を1、比較条件を1〜3個に減らして再実行してください。")
                    st.exception(e)

        result = st.session_state.get("v20_compare_results", None)
        if not result:
            st.caption("まだ比較実験は実行されていません。現在の単独ランの読み取りは上のv19までで確認できます。")
            return

        raw_df = pd.DataFrame(result.get("raw", []))
        if raw_df.empty:
            st.warning("比較実験の結果が空でした。世代数を増やすか、エラーが出ていないか確認してください。")
            return

        metric_cols = [
            "最終個体数", "個体数変化", "平均W", "終盤W", "合計出生", "合計死亡", "出生-死亡",
            "最終資源総量", "資源変化", "哲学割合変化", "通常割合変化", "赤青差_最終",
            "タカ比率変化", "捕食傾向比率変化", "行動型多様度_最終"
        ]
        keep_cols = [c for c in metric_cols if c in raw_df.columns]
        agg = raw_df.groupby("条件", as_index=False)[keep_cols].mean()
        counts = raw_df.groupby("条件", as_index=False).size().rename(columns={"size": "反復数"})
        agg = counts.merge(agg, on="条件", how="left")
        order = {name: i for i, name in enumerate(result.get("order", []))}
        agg["_order"] = agg["条件"].map(order).fillna(999)
        agg = agg.sort_values("_order").drop(columns=["_order"])

        baseline = agg[agg["条件"] == "基準"]
        if not baseline.empty:
            base = baseline.iloc[0]
            for c in ["最終個体数", "個体数変化", "平均W", "終盤W", "合計出生", "合計死亡", "出生-死亡", "最終資源総量", "資源変化", "哲学割合変化", "通常割合変化", "赤青差_最終"]:
                if c in agg.columns and c in base.index:
                    agg[f"Δ{c}(基準差)"] = pd.to_numeric(agg[c], errors="coerce") - float(base[c])

        st.markdown("#### v20-A. 条件別の比較表")
        st.caption("Δ列は基準ランとの差です。同じseedで条件だけを変えているので、差が大きいほどその条件が結果を動かした因果候補になります。")
        fmt_cols = {}
        for c in agg.columns:
            if c == "条件":
                continue
            if "W" in c or "割合" in c or "比率" in c or "多様度" in c:
                fmt_cols[c] = st.column_config.NumberColumn(format="%.3f")
            elif c.startswith("Δ") or "変化" in c or "差" in c:
                fmt_cols[c] = st.column_config.NumberColumn(format="%+.2f")
            else:
                fmt_cols[c] = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(agg, use_container_width=True, hide_index=True, column_config=fmt_cols)

        # 視覚化：単位が混ざらないように、個体数差とW差を分ける。
        if not baseline.empty:
            delta_cols_pop = [c for c in ["Δ最終個体数(基準差)", "Δ出生-死亡(基準差)", "Δ最終資源総量(基準差)"] if c in agg.columns]
            delta_cols_w = [c for c in ["Δ平均W(基準差)", "Δ終盤W(基準差)", "Δ哲学割合変化(基準差)", "Δ通常割合変化(基準差)"] if c in agg.columns]
            chart_df = agg[agg["条件"] != "基準"].copy()
            if len(chart_df) and delta_cols_pop:
                st.markdown("#### v20-B. 基準との差：個体数・出生死亡・資源")
                st.bar_chart(chart_df.set_index("条件")[delta_cols_pop])
            if len(chart_df) and delta_cols_w:
                st.markdown("#### v20-C. 基準との差：W・比率変化")
                st.bar_chart(chart_df.set_index("条件")[delta_cols_w])

        # 行動型ごとの比率変化も比較する。
        philo_delta_cols = []
        for lab in PHILO_LABELS.values():
            c = f"{lab} 比率変化"
            if c in raw_df.columns:
                philo_delta_cols.append(c)
        if philo_delta_cols:
            ph_agg = raw_df.groupby("条件", as_index=False)[philo_delta_cols].mean()
            ph_agg["_order"] = ph_agg["条件"].map(order).fillna(999)
            ph_agg = ph_agg.sort_values("_order").drop(columns=["_order"])
            st.markdown("#### v20-D. 各条件で、どの行動型が伸びたか")
            st.caption("通常個体・各哲学型の比率変化です。基準と条件変更を比べると、どの遺伝子がどの環境で伸びやすいかが見えます。")
            st.dataframe(ph_agg, use_container_width=True, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in philo_delta_cols})

        # テキスト解釈：なぜその差が出たと読めるか。
        st.markdown("#### v20-E. 比較から読める因果候補")
        causal_lines = _v20_causal_text(agg)
        for line in causal_lines:
            st.markdown(f"- {line}")

        with st.expander("v20 生データ：各seed反復ごとの結果", expanded=False):
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

    def _run_v20_comparison(specs, generations: int, repeats: int):
        """現在の世界を退避し、同じseedで複数条件を走らせる。"""
        generations = max(1, int(generations))
        repeats = max(1, int(repeats))
        old = {
            "world": st.session_state.world,
            "history": list(st.session_state.history),
            "gen": int(st.session_state.gen),
            "phase": int(st.session_state.phase),
            "last_phase_executed": st.session_state.last_phase_executed,
            "sig": st.session_state.get("sig", None),
        }
        override_keys = sorted({k for _, ov in specs for k in ov.keys()} | {"seed"})
        old_globals = {k: globals().get(k, None) for k in override_keys}
        rows = []
        prog = st.progress(0, text="比較実験を準備中")
        total = max(1, len(specs) * repeats)
        done = 0
        try:
            for rep in range(repeats):
                base_seed = int(old_globals.get("seed", globals().get("seed", 0)))
                scenario_seed = int(base_seed + rep * 10007)
                for name, overrides in specs:
                    for k, v in old_globals.items():
                        if k in globals() and v is not None:
                            globals()[k] = v
                    globals()["seed"] = scenario_seed
                    for k, v in overrides.items():
                        globals()[k] = v
                    reset_world()
                    advance_generations(generations)
                    rows.append(_v20_summarize_history(st.session_state.history, name, scenario_seed, generations, rep + 1, overrides))
                    done += 1
                    prog.progress(done / total, text=f"比較実験中：{done}/{total}  {name}")
        finally:
            for k, v in old_globals.items():
                if v is not None:
                    globals()[k] = v
            st.session_state.world = old["world"]
            st.session_state.history = old["history"]
            st.session_state.gen = old["gen"]
            st.session_state.phase = old["phase"]
            st.session_state.last_phase_executed = old["last_phase_executed"]
            if old["sig"] is not None:
                st.session_state.sig = old["sig"]
            prog.empty()
        return {"raw": rows, "order": [name for name, _ in specs], "generations": generations, "repeats": repeats}

    def _v20_series(frame, col):
        if col not in frame.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[col], errors="coerce")

    def _v20_first(frame, col, default=np.nan):
        s = _v20_series(frame, col).dropna()
        return float(s.iloc[0]) if len(s) else default

    def _v20_last(frame, col, default=np.nan):
        s = _v20_series(frame, col).dropna()
        return float(s.iloc[-1]) if len(s) else default

    def _v20_mean(frame, col, default=np.nan):
        s = _v20_series(frame, col).dropna()
        return float(s.mean()) if len(s) else default

    def _v20_sum(frame, col, default=0.0):
        s = _v20_series(frame, col).dropna()
        return float(s.sum()) if len(s) else default

    def _v20_tail_mean(frame, col, frac=0.25, default=np.nan):
        s = _v20_series(frame, col).dropna()
        if not len(s):
            return default
        k = max(1, int(np.ceil(len(s) * float(frac))))
        return float(s.tail(k).mean())

    def _v20_summarize_history(history, scenario_name, scenario_seed, generations, repeat_index, overrides):
        frame = pd.DataFrame(history)
        if frame.empty:
            return {"条件": scenario_name, "seed": scenario_seed, "反復": repeat_index, "実行世代数": generations, "エラー": "履歴なし"}
        row = {
            "条件": scenario_name,
            "seed": int(scenario_seed),
            "反復": int(repeat_index),
            "実行世代数": int(generations),
            "変更した条件": ", ".join([f"{k}={v}" for k, v in overrides.items()]) if overrides else "なし",
            "初期個体数": _v20_first(frame, "個体数（体）", 0.0),
            "最終個体数": _v20_last(frame, "個体数（体）", 0.0),
            "最大個体数": float(_v20_series(frame, "個体数（体）").max()) if "個体数（体）" in frame.columns else np.nan,
            "最小個体数": float(_v20_series(frame, "個体数（体）").min()) if "個体数（体）" in frame.columns else np.nan,
            "平均W": _v20_mean(frame, "個体群全体W（増殖率）"),
            "終盤W": _v20_tail_mean(frame, "個体群全体W（増殖率）"),
            "合計出生": _v20_sum(frame, "出生数（体/世代）"),
            "合計死亡": _v20_sum(frame, "死亡数（体/世代）"),
            "最終資源総量": _v20_last(frame, "資源総量（単位）", 0.0),
            "初期資源総量": _v20_first(frame, "資源総量（単位）", 0.0),
            "平均Gini": _v20_mean(frame, "資源格差Gini（0-1）"),
            "平均交尾成立率": _v20_mean(frame, "交尾成立率（0-1）"),
            "平均捕食成功率": _v20_mean(frame, "捕食成功率（0-1）"),
            "赤最終": _v20_last(frame, "赤個体数（体）", 0.0),
            "青最終": _v20_last(frame, "青個体数（体）", 0.0),
            "哲学割合初期": _v20_first(frame, "哲学個体割合（0-1）", 0.0),
            "哲学割合最終": _v20_last(frame, "哲学個体割合（0-1）", 0.0),
            "通常割合初期": _v20_first(frame, "通常個体割合（0-1）", 0.0),
            "通常割合最終": _v20_last(frame, "通常個体割合（0-1）", 0.0),
            "タカ比率初期": _v20_first(frame, "タカ比率（0-1）", 0.0),
            "タカ比率最終": _v20_last(frame, "タカ比率（0-1）", 0.0),
            "捕食傾向比率初期": _v20_first(frame, "捕食傾向比率（0-1）", 0.0),
            "捕食傾向比率最終": _v20_last(frame, "捕食傾向比率（0-1）", 0.0),
            "行動型多様度_最終": _v20_last(frame, "行動型多様度（通常含むSimpson）", 0.0),
        }
        row["個体数変化"] = row["最終個体数"] - row["初期個体数"]
        row["出生-死亡"] = row["合計出生"] - row["合計死亡"]
        row["資源変化"] = row["最終資源総量"] - row["初期資源総量"]
        row["赤青差_最終"] = row["赤最終"] - row["青最終"]
        row["哲学割合変化"] = row["哲学割合最終"] - row["哲学割合初期"]
        row["通常割合変化"] = row["通常割合最終"] - row["通常割合初期"]
        row["タカ比率変化"] = row["タカ比率最終"] - row["タカ比率初期"]
        row["捕食傾向比率変化"] = row["捕食傾向比率最終"] - row["捕食傾向比率初期"]

        best_lab = None
        best_score = -1e9
        worst_lab = None
        worst_score = 1e9
        for lab in PHILO_LABELS.values():
            r0 = _v20_first(frame, f"{lab} 比率（0-1）", 0.0)
            r1 = _v20_last(frame, f"{lab} 比率（0-1）", 0.0)
            wmean = _v20_mean(frame, f"{lab} W", 1.0)
            births = _v20_sum(frame, f"{lab} 実出生（体/世代）")
            deaths = _v20_sum(frame, f"{lab} 死亡（体/世代）")
            exposure = _v20_sum(frame, f"{lab} 数（体）")
            bd_rate = (births - deaths) / max(exposure, 1.0)
            score = (r1 - r0) * 2.0 + (wmean - 1.0) + bd_rate
            row[f"{lab} 比率変化"] = r1 - r0
            row[f"{lab} 平均W"] = wmean
            row[f"{lab} 出生-死亡率"] = bd_rate
            row[f"{lab} 優勢スコア"] = score
            if score > best_score:
                best_score = score; best_lab = lab
            if score < worst_score:
                worst_score = score; worst_lab = lab
        row["推定優勢型"] = best_lab if best_lab is not None else "不明"
        row["推定劣勢型"] = worst_lab if worst_lab is not None else "不明"
        return row

    def _v20_causal_text(agg):
        lines = []
        if agg.empty or "基準" not in set(agg["条件"]):
            return ["基準ランがないため、条件差の比較はできません。"]
        base = agg[agg["条件"] == "基準"].iloc[0]
        base_pop = float(base.get("最終個体数", 0.0))
        pop_threshold = max(5.0, abs(base_pop) * 0.05)
        for _, r in agg.iterrows():
            name = str(r.get("条件", ""))
            if name == "基準":
                continue
            dpop = float(r.get("Δ最終個体数(基準差)", np.nan)) if "Δ最終個体数(基準差)" in r.index else np.nan
            dw = float(r.get("Δ終盤W(基準差)", np.nan)) if "Δ終盤W(基準差)" in r.index else np.nan
            dbd = float(r.get("Δ出生-死亡(基準差)", np.nan)) if "Δ出生-死亡(基準差)" in r.index else np.nan
            direction = "ほぼ同等"
            if not np.isnan(dpop):
                if dpop > pop_threshold:
                    direction = "基準より個体群維持に有利"
                elif dpop < -pop_threshold:
                    direction = "基準より個体群維持に不利"
            reason = f"**{name}** は {direction} です。"
            if not np.isnan(dpop):
                reason += f"最終個体数差が {dpop:+.1f} 体"
            if not np.isnan(dw):
                reason += f"、終盤W差が {dw:+.3f}"
            if not np.isnan(dbd):
                reason += f"、出生-死亡差が {dbd:+.1f}"
            reason += " だからです。"

            if name.startswith("哲学遺伝子OFF"):
                if not np.isnan(dpop) and dpop < -pop_threshold:
                    reason += "哲学的行動補正を外すと集団が弱くなるため、この設定では哲学遺伝子が資源獲得・死亡回避・繁殖成立のどれかに寄与している可能性があります。"
                elif not np.isnan(dpop) and dpop > pop_threshold:
                    reason += "哲学的行動補正を外すと集団が強くなるため、この設定では哲学補正が過剰な抑制や行動の偏りとして働いている可能性があります。"
                else:
                    reason += "差が小さいため、哲学遺伝子の効果は弱いか、通常個体割合80%によって薄まっている可能性があります。"
            elif name.startswith("通常100%"):
                reason += "哲学個体を完全に消した対照です。基準との差が大きければ、哲学個体の存在自体が生態系に影響している可能性があります。差が小さければ、現状の哲学補正は中立行動に近い可能性があります。"
            elif name.startswith("哲学100%"):
                reason += "通常個体を消した条件です。ここで個体群が強くなるなら哲学型同士の相互作用が有利、弱くなるなら通常個体が緩衝材・中立対照として生態安定に寄与している可能性があります。"
            elif name.startswith("捕食OFF"):
                reason += "捕食を外した差なので、捕食が短期資源獲得として有利なのか、失敗コストや個体数減少として不利なのかを読む条件です。"
            elif name.startswith("密度依存OFF"):
                reason += "密度依存を外した差なので、過密が出生を抑えているのか、逆に過密抑制が資源枯渇や崩壊を防いでいるのかを読む条件です。"
            elif name.startswith("局所資源再生OFF"):
                reason += "局所資源再生を外した差なので、個体が資源を使った場所に再び資源が戻る仕組みが、生存や繁殖へ接続しているかを読む条件です。"
            elif name.startswith("近親回避OFF"):
                reason += "近親回避を外した差なので、近親回避が繁殖機会を減らすコストなのか、系統多様性を保つ利益なのかを読む条件です。"
            elif name.startswith("生態補正OFF"):
                reason += "複数の生態補正を同時に外しているため、個別因果ではなく、モデル全体がどの程度それらの補正に依存しているかを見る条件です。"
            lines.append(reason)
        if not lines:
            lines.append("比較条件が基準だけなので、因果候補はまだ読めません。")
        lines.append("注意：v20の差分は、同じseedで条件だけを変えた比較なので、単独ラン内の相関よりは強い根拠です。ただし、最終的な主張にはseed反復を増やし、同じ傾向が再現するかを見る必要があります。")
        return lines



    def show_public_causal_report():
        """外部向けの整理済み因果レポート。重複した短評ではなく、原因経路を一つの流れで説明する。"""
        if len(df) < 2:
            st.info("分析レポートには、少なくとも2世代以上の履歴が必要です。まず数十世代ほど進めてください。")
            return

        # ---- safe helpers ----
        def srs(col):
            if col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce")
        def meanv(col, default=np.nan, frame=None):
            frame = df if frame is None else frame
            if col not in frame.columns:
                return default
            v = pd.to_numeric(frame[col], errors="coerce").dropna()
            return float(v.mean()) if len(v) else default
        def firstv(col, default=np.nan):
            v = srs(col).dropna()
            return float(v.iloc[0]) if len(v) else default
        def lastv(col, default=np.nan):
            v = srs(col).dropna()
            return float(v.iloc[-1]) if len(v) else default
        def sumv(col, default=0.0):
            v = srs(col).dropna()
            return float(v.sum()) if len(v) else default
        def corr(a, b):
            if a not in df.columns or b not in df.columns:
                return np.nan
            d = pd.concat([srs(a), srs(b)], axis=1).dropna()
            if len(d) < 4:
                return np.nan
            return float(d.iloc[:, 0].corr(d.iloc[:, 1]))
        def fmt(x, digits=3, signed=False):
            try:
                if x is None or pd.isna(x):
                    return "—"
                sign = "+" if signed and float(x) >= 0 else ""
                return f"{sign}{float(x):.{digits}f}"
            except Exception:
                return "—"
        def direction(x, pos=0.03, neg=-0.03):
            if x is None or pd.isna(x):
                return "不明"
            if x > pos:
                return "増加"
            if x < neg:
                return "減少"
            return "ほぼ維持"
        def rel_word(x, high, low=None):
            if x is None or pd.isna(x):
                return "不明"
            low = -high if low is None else low
            if x > high:
                return "高い"
            if x < low:
                return "低い"
            return "平均付近"
        def strongest_name(rows, key, reverse=True):
            vals = [r for r in rows if key in r and not pd.isna(r[key])]
            if not vals:
                return "—"
            vals = sorted(vals, key=lambda r: r[key], reverse=reverse)
            return str(vals[0]["型"])

        full_n = len(df)
        early = df.head(max(3, full_n // 4))
        late = df.tail(max(3, full_n // 4))
        recent = df.tail(min(80, full_n))

        # ---- global environment diagnostics ----
        pop0, pop1 = firstv("個体数（体）", 0), lastv("個体数（体）", 0)
        w_mean = meanv("個体群全体W（増殖率）", np.nan)
        w_late = meanv("個体群全体W（増殖率）", np.nan, late)
        births = meanv("出生数（体/世代）", 0, late)
        deaths = meanv("死亡数（体/世代）", 0, late)
        bd_ratio = births / max(deaths, 1e-9)
        res0, res1 = firstv("資源総量（単位）", 0), lastv("資源総量（単位）", 0)
        bag0, bag1 = firstv("平均所持資源（単位/体）", np.nan), lastv("平均所持資源（単位/体）", np.nan)
        gini_mean = meanv("資源格差Gini（0-1）", np.nan)
        gini_late = meanv("資源格差Gini（0-1）", np.nan, late)
        density_mean = meanv("平均局所密度（体/近傍）", np.nan, late)
        mate_rate = meanv("交尾成立率（0-1）", np.nan, late)
        kin_block = meanv("近親交配回避（回/世代）", 0, late)
        density_block = meanv("過密で抑制された出生候補（回/世代）", 0, late)
        pred_success = meanv("捕食成功率（0-1）", 0, late)
        pred_attempt = meanv("捕食試行（回/世代）", 0, late)
        contest_gain = meanv("争奪で得たV合計（単位/世代）", 0, late)
        contest_cost = meanv("争奪で支払ったC合計（単位/世代）", 0, late)
        contest_net = contest_gain - contest_cost
        c_birth_bag = corr("出生数（体/世代）", "平均所持資源（単位/体）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_birth_density = corr("出生数（体/世代）", "平均局所密度（体/近傍）")

        # ---- gene profiles ----
        labels = list(PHILO_LABELS.values())
        rows = []
        total_parent_births = 0.0
        for lab in labels:
            total_parent_births += sumv(f"{lab} 親参加:実出生（回/世代）", 0.0)
        for lab in labels:
            n0 = firstv(f"{lab} 数（体）", 0)
            n1 = lastv(f"{lab} 数（体）", 0)
            r0 = firstv(f"{lab} 比率（0-1）", 0)
            r1 = lastv(f"{lab} 比率（0-1）", 0)
            rd = r1 - r0
            w_avg = meanv(f"{lab} W", np.nan)
            w_l = meanv(f"{lab} W", np.nan, late)
            net = meanv(f"{lab} 資源収支ネット（単位/世代）", np.nan, late)
            hunger = meanv(f"{lab} 空腹個体比率（0-1）", np.nan, late)
            birth_real = sumv(f"{lab} 実出生（体/世代）", 0)
            parent_real = sumv(f"{lab} 親参加:実出生（回/世代）", 0)
            deaths_sum = sumv(f"{lab} 死亡（体/世代）", 0)
            parent_share = parent_real / max(total_parent_births, 1e-9)
            gather = meanv(f"{lab} 行動率:採取（0-1）", np.nan, late)
            move = meanv(f"{lab} 行動率:移動（0-1）", np.nan, late)
            mate = meanv(f"{lab} 行動率:交尾（0-1）", np.nan, late)
            escape = meanv(f"{lab} 行動率:回避（0-1）", np.nan, late)
            battle = meanv(f"{lab} 行動率:戦闘（0-1）", np.nan, late)
            pred = meanv(f"{lab} 行動率:捕食（0-1）", np.nan, late)
            foot = meanv(f"{lab} 平均足元資源（単位/マス）", np.nan, late)
            local_den = meanv(f"{lab} 平均局所密度（体/近傍）", np.nan, late)
            hawk = meanv(f"{lab} タカ比率（0-1）", np.nan, late)
            pred_gene = meanv(f"{lab} 捕食傾向比率（0-1）", np.nan, late)
            red_ratio = meanv(f"{lab} 赤比率（0-1）", np.nan, late)
            blue_ratio = meanv(f"{lab} 青比率（0-1）", np.nan, late)
            # score is only for ordering; text explains components.
            score = 0.0
            if not pd.isna(rd): score += rd * 8
            if not pd.isna(w_l): score += (w_l - 1.0) * 4
            if not pd.isna(parent_share): score += parent_share * 2
            if not pd.isna(hunger): score -= hunger
            if not pd.isna(net): score += np.tanh(net / 60.0)
            if deaths_sum > parent_real and parent_real > 0: score -= 0.6
            rows.append({
                "型": lab, "初期数": n0, "最新数": n1, "数変化": n1-n0,
                "初期比率": r0, "最新比率": r1, "比率変化": rd,
                "平均W": w_avg, "終盤W": w_l, "資源収支": net, "空腹率": hunger,
                "親として残したコピー": parent_real, "子として生まれたコピー": birth_real,
                "死亡": deaths_sum, "親出生シェア": parent_share,
                "採取率": gather, "移動率": move, "交尾率": mate, "回避率": escape, "戦闘率": battle, "捕食率": pred,
                "足元資源": foot, "局所密度": local_den,
                "タカ比率": hawk, "捕食傾向比率": pred_gene,
                "赤比率": red_ratio, "青比率": blue_ratio,
                "総合スコア": score,
            })
        gene_df = pd.DataFrame(rows).sort_values("総合スコア", ascending=False)
        strongest = gene_df.iloc[0].to_dict() if len(gene_df) else {}
        weakest = gene_df.iloc[-1].to_dict() if len(gene_df) else {}
        avg_hunger = np.nanmean([r["空腹率"] for r in rows]) if rows else np.nan
        avg_foot = np.nanmean([r["足元資源"] for r in rows]) if rows else np.nan
        avg_density = np.nanmean([r["局所密度"] for r in rows]) if rows else np.nan
        avg_hawk = np.nanmean([r["タカ比率"] for r in rows]) if rows else np.nan
        avg_pred_gene = np.nanmean([r["捕食傾向比率"] for r in rows]) if rows else np.nan

        st.markdown("### 分析レポート")
        explain_box(
            "このレポートで見る因果の形",
            "このゲームでは、遺伝子が直接『勝つ』わけではありません。遺伝子は、個体が資源・危険・相手・空間をどう評価するかを少し変えます。その評価差が、採取、移動、回避、交尾、捕食、争奪の選び方を変えます。最後に、その行動が環境条件とぶつかり、資源を得る、空腹になる、相手に会えない、過密で子が置けない、死亡する、という経路を通ってコピー数Wへ変わります。したがってここでは、結果だけではなく **遺伝子 → 行動 → 環境との接触 → 資源/危険/繁殖機会 → 出生/死亡 → W** の順に読みます。"
        )

        # 1) environment diagnosis
        st.markdown("#### 1. 環境が作っている圧")
        env_rows = []
        env_rows.append({
            "圧": "個体群維持圧",
            "観察": f"個体数 {int(pop0)}→{int(pop1)}、終盤W {fmt(w_late)}、終盤の出生/死亡比 {fmt(bd_ratio,2)}",
            "結果を生む仕組み": "Wが1を下回る環境では、どの遺伝子もまず死亡・維持コスト・繁殖失敗を避けないと残れません。Wが1付近なら、全体崩壊ではなく、内部の遺伝子差が表に出やすくなります。",
            "この圧で有利になりやすい型": "低空腹・低死亡・親出生がある型。単に攻撃的な型ではなく、資源を失いにくく繁殖出口まで到達する型。"
        })
        env_rows.append({
            "圧": "資源アクセス圧",
            "観察": f"資源総量 {int(res0)}→{int(res1)}、平均所持資源 {fmt(bag0,2)}→{fmt(bag1,2)}、個体数×資源相関 {fmt(c_pop_res,3,True)}",
            "結果を生む仕組み": "資源総量が多くても、個体がその場所にいない、見えていない、移動コストで失う、採取前に混雑するならコピーにはなりません。ここでは『資源がある』ではなく『資源を個体内資源へ変換できたか』が選択圧になります。",
            "この圧で有利になりやすい型": f"足元資源が高い型（{strongest_name(rows,'足元資源')} など）や、資源収支が高く空腹率が低い型。"
        })
        env_rows.append({
            "圧": "資源格差・飢餓圧",
            "観察": f"終盤Gini {fmt(gini_late)}、死亡×Gini相関 {fmt(c_death_gini,3,True)}",
            "結果を生む仕組み": "Giniが高い時、資源不足は全員へ均等に来るのではなく、一部の個体を集中的に削ります。死亡とGiniが正に連動するなら、淘汰圧は『全体の貧しさ』ではなく『資源を取れなかった個体の脱落』として働いています。",
            "この圧で有利になりやすい型": f"空腹率が低い型（{strongest_name(rows,'空腹率', reverse=False)}）と、移動コストを払いすぎず局所資源に接続できる型。"
        })
        env_rows.append({
            "圧": "繁殖出口圧",
            "観察": f"交尾成立率 {fmt(mate_rate)}、出生×平均資源相関 {fmt(c_birth_bag,3,True)}、近親回避 {fmt(kin_block,2)}、過密出生抑制 {fmt(density_block,2)}",
            "結果を生む仕組み": "資源を持っていても、交尾相手・血縁条件・空きマス・局所密度を通過しなければコピーは生まれません。交尾率が高いのに親出生が少ない型は、行動の意欲ではなく環境側の出口で詰まっています。",
            "この圧で有利になりやすい型": f"親として残したコピーが多い型（{strongest_name(rows,'親として残したコピー')}）と、適度な局所密度で相手に会える型。"
        })
        env_rows.append({
            "圧": "空間・局所密度圧",
            "観察": f"平均局所密度 {fmt(density_mean,2)}、出生×局所密度相関 {fmt(c_birth_density,3,True)}",
            "結果を生む仕組み": "局所密度は二面性があります。高いと交尾相手には会いやすいが、競争・衝突・出生場所不足が増えます。低いと安全でも交尾相手が見つからずコピーが増えません。",
            "この圧で有利になりやすい型": "密集しすぎず孤立しすぎない型。回避や移動が多すぎる型は、危険を避けても繁殖機会を失う場合があります。"
        })
        env_rows.append({
            "圧": "争奪・捕食媒介圧",
            "観察": f"争奪ネット {fmt(contest_net,2,True)}、捕食試行 {fmt(pred_attempt,2)}、捕食成功率 {fmt(pred_success)}",
            "結果を生む仕組み": "タカや捕食傾向は、資源不足を短期的に救うことがあります。しかし成功率が低い、コストが高い、相手を減らして繁殖機会まで減らす場合は罰になります。ここを見ないと、哲学型の効果と攻撃/捕食遺伝子の効果が混ざります。",
            "この圧で有利になりやすい型": "争奪ネットが正ならタカ寄りが利益を受けやすく、負ならハト寄り・回避寄りが利益を受けやすい。捕食成功率が高いなら捕食傾向型、低いなら非捕食型が有利になりやすい。"
        })
        st.dataframe(pd.DataFrame(env_rows), use_container_width=True, hide_index=True)

        # 2) gene table
        st.markdown("#### 2. 遺伝子型別：結果と原因経路を一枚で見る")
        display_cols = [
            "型", "初期数", "最新数", "数変化", "比率変化", "平均W", "終盤W",
            "親として残したコピー", "子として生まれたコピー", "死亡", "資源収支", "空腹率",
            "足元資源", "局所密度", "採取率", "移動率", "交尾率", "回避率", "戦闘率", "捕食率",
            "タカ比率", "捕食傾向比率", "赤比率", "青比率", "総合スコア"
        ]
        st.caption("総合スコアは順位づけの補助です。結論は、比率変化・W・親出生・死亡・資源収支・空腹率・同伴遺伝子を一緒に読んで判断します。")
        st.dataframe(
            gene_df[display_cols], use_container_width=True, hide_index=True,
            column_config={
                "比率変化": st.column_config.NumberColumn(format="%+.3f"),
                "平均W": st.column_config.NumberColumn(format="%.3f"),
                "終盤W": st.column_config.NumberColumn(format="%.3f"),
                "資源収支": st.column_config.NumberColumn(format="%+.1f"),
                "空腹率": st.column_config.NumberColumn(format="%.3f"),
                "足元資源": st.column_config.NumberColumn(format="%.2f"),
                "局所密度": st.column_config.NumberColumn(format="%.2f"),
                "採取率": st.column_config.NumberColumn(format="%.3f"),
                "移動率": st.column_config.NumberColumn(format="%.3f"),
                "交尾率": st.column_config.NumberColumn(format="%.3f"),
                "回避率": st.column_config.NumberColumn(format="%.3f"),
                "戦闘率": st.column_config.NumberColumn(format="%.3f"),
                "捕食率": st.column_config.NumberColumn(format="%.3f"),
                "タカ比率": st.column_config.NumberColumn(format="%.3f"),
                "捕食傾向比率": st.column_config.NumberColumn(format="%.3f"),
                "赤比率": st.column_config.NumberColumn(format="%.3f"),
                "青比率": st.column_config.NumberColumn(format="%.3f"),
                "総合スコア": st.column_config.NumberColumn(format="%+.3f"),
            }
        )

        # 3) team analysis, not tautological
        st.markdown("#### 3. 赤チーム・青チーム差：数が多い理由を分解する")
        red0, red1 = firstv("赤個体数（体）", 0), lastv("赤個体数（体）", 0)
        blue0, blue1 = firstv("青個体数（体）", 0), lastv("青個体数（体）", 0)
        red_res, blue_res = meanv("赤 平均所持資源", np.nan, late), meanv("青 平均所持資源", np.nan, late)
        red_g, blue_g = meanv("赤 Gini", np.nan, late), meanv("青 Gini", np.nan, late)
        red_hawk, blue_hawk = meanv("赤タカ比率（0-1）", np.nan, late), meanv("青タカ比率（0-1）", np.nan, late)
        red_share = red1 / max(red1 + blue1, 1)
        blue_share = blue1 / max(red1 + blue1, 1)
        team_rows = []
        team_rows.append({"比較軸":"個体数変化", "赤":f"{int(red0)}→{int(red1)}", "青":f"{int(blue0)}→{int(blue1)}", "読み取り":"これは結果であって原因ではありません。以下の資源・格差・タカ比率・内部構成を見るための入口です。"})
        team_rows.append({"比較軸":"平均所持資源", "赤":fmt(red_res,2), "青":fmt(blue_res,2), "読み取り":"多い側が平均資源も高いなら、チーム差は資源アクセスで支えられている可能性があります。多い側の資源が低いなら、死亡回避・交尾・配置が原因候補です。"})
        team_rows.append({"比較軸":"資源格差Gini", "赤":fmt(red_g,3), "青":fmt(blue_g,3), "読み取り":"Giniが高いチームは平均資源が同じでも貧しい個体を多く作ります。死亡が多いなら、チーム差は資源格差を通じて生まれている可能性があります。"})
        team_rows.append({"比較軸":"タカ比率", "赤":fmt(red_hawk,3), "青":fmt(blue_hawk,3), "読み取り":"タカ比率が高いチームが伸びるなら争奪利得が原因候補、低いチームが伸びるなら戦闘コスト回避が原因候補です。"})
        # internal type bias
        red_biases=[]; blue_biases=[]
        for r in rows:
            if not pd.isna(r.get("赤比率", np.nan)):
                red_biases.append((r["型"], r["赤比率"]))
            if not pd.isna(r.get("青比率", np.nan)):
                blue_biases.append((r["型"], r["青比率"]))
        red_top = max(red_biases, key=lambda x:x[1])[0] if red_biases else "—"
        blue_top = max(blue_biases, key=lambda x:x[1])[0] if blue_biases else "—"
        team_rows.append({"比較軸":"内部の行動型構成", "赤":red_top, "青":blue_top, "読み取り":"優位チームに特定の哲学型や通常個体が偏っているなら、チーム差は色そのものではなく、内部遺伝子構成と空間配置の組み合わせで生じている可能性があります。"})
        st.dataframe(pd.DataFrame(team_rows), use_container_width=True, hide_index=True)

        team_text = []
        if abs(red_share - blue_share) < 0.06:
            team_text.append("赤青の差は小さいため、チーム色そのものを大きな原因として扱うより、各チーム内の遺伝子構成・資源配置・局所密度を優先して見ます。")
        else:
            side = "赤" if red_share > blue_share else "青"
            other = "青" if side == "赤" else "赤"
            side_res = red_res if side == "赤" else blue_res
            other_res = blue_res if side == "赤" else red_res
            side_g = red_g if side == "赤" else blue_g
            other_g = blue_g if side == "赤" else red_g
            side_h = red_hawk if side == "赤" else blue_hawk
            other_h = blue_hawk if side == "赤" else red_hawk
            team_text.append(f"{side}チームが数では優勢です。ただし、理由は『コピー維持に成功しているから』ではありません。それは結果です。原因候補は、{side}側が資源へ届きやすかったのか、資源格差が低かったのか、タカ/ハト構成が環境に合っていたのか、特定の行動型が{side}側へ偏ったのか、に分けて読む必要があります。")
            if not pd.isna(side_res) and not pd.isna(other_res):
                if side_res > other_res + 0.5:
                    team_text.append(f"平均所持資源は{side}が高めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。この場合、{side}優勢の一部は、資源アクセスの良さが死亡回避や繁殖余剰へ変換された結果として説明できます。")
                elif side_res < other_res - 0.5:
                    team_text.append(f"平均所持資源はむしろ{side}が低めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。それでも{side}が多いなら、資源量以外の要因、たとえば死亡率の低さ、交尾機会、空間配置、内部遺伝子構成を疑うべきです。")
            if not pd.isna(side_g) and not pd.isna(other_g):
                if side_g < other_g - 0.03:
                    team_text.append(f"{side}のGiniが低めです。資源が均等に行き渡ると、低資源で脱落する個体が減り、チーム全体の維持に効く可能性があります。")
                elif side_g > other_g + 0.03:
                    team_text.append(f"{side}のGiniは高めです。数で勝っていても、内部では一部個体に資源が偏っている可能性があり、長期的には不安定要因です。")
            if not pd.isna(side_h) and not pd.isna(other_h):
                if side_h > other_h + 0.05 and contest_net > 0:
                    team_text.append(f"{side}はタカ比率が高く、かつ争奪ネットが正です。この組み合わせなら、争奪利得がチーム優勢を支えた可能性があります。")
                elif side_h > other_h + 0.05 and contest_net <= 0:
                    team_text.append(f"{side}はタカ比率が高い一方、争奪ネットは正とは言えません。この場合、タカ性は優位の原因ではなくコスト要因かもしれません。")
                elif side_h < other_h - 0.05 and contest_net <= 0:
                    team_text.append(f"{side}はタカ比率が低く、争奪ネットも弱い/負です。この場合、戦闘コストを避けたことがチーム維持に効いた可能性があります。")
        explain_box("チーム差の読み取り", "\n\n".join(team_text))

        # 4) gene causal narratives
        st.markdown("#### 4. 各遺伝子型の原因経路")
        st.caption("ここでは『増えた/減った』を結論として置かず、その結果を生んだ入口・経路・出口・媒介遺伝子を分けます。")
        for _, r in gene_df.iterrows():
            lab = str(r["型"])
            expanded = lab in [str(strongest.get("型", "")), str(weakest.get("型", ""))]
            with st.expander(f"{lab}：何が結果を生んだのか", expanded=expanded):
                lines = []
                lines.append(f"**結果。** 初期比率 {fmt(r['初期比率'])} から最新比率 {fmt(r['最新比率'])} へ変化し、比率変化は {fmt(r['比率変化'],3,True)}、終盤Wは {fmt(r['終盤W'])} です。これは、この型が現在の環境で相対的に {direction(r['比率変化'])} していることを示します。")
                # direct reproduction/death
                if r["親として残したコピー"] > r["死亡"]:
                    lines.append(f"**直接経路。** 親として残したコピー {fmt(r['親として残したコピー'],0)} が死亡 {fmt(r['死亡'],0)} を上回っています。この型は単に生き残っただけでなく、繁殖出口を通ってコピーを作る経路が働いています。")
                elif r["親として残したコピー"] == 0 and r["死亡"] > 0:
                    lines.append(f"**直接経路。** 親として残したコピーがほぼなく、死亡は {fmt(r['死亡'],0)} あります。この型は、残存していても次世代への出口が弱く、死亡圧または繁殖制約に押されています。")
                else:
                    lines.append(f"**直接経路。** 親出生 {fmt(r['親として残したコピー'],0)} と死亡 {fmt(r['死亡'],0)} の差が大きくありません。増減の主因は、出生数そのものより、他型の増減・相対頻度・環境配置である可能性があります。")
                # resource/environment contact
                contact = []
                if not pd.isna(r["資源収支"]):
                    contact.append(f"資源収支は {fmt(r['資源収支'],1,True)}")
                if not pd.isna(r["空腹率"]):
                    contact.append(f"空腹率は全型平均に対して {rel_word(float(r['空腹率'])-avg_hunger, 0.05)}")
                if not pd.isna(r["足元資源"]):
                    contact.append(f"足元資源は全型平均に対して {rel_word(float(r['足元資源'])-avg_foot, 0.4)}")
                if not pd.isna(r["局所密度"]):
                    contact.append(f"局所密度は全型平均に対して {rel_word(float(r['局所密度'])-avg_density, 0.4)}")
                lines.append("**環境との接触。** " + "、".join(contact) + "。ここが重要なのは、同じ遺伝子でも、資源の近くにいるか、密集しているか、孤立しているかで有利不利が変わるからです。")
                # action pathway
                actions = {"採取": r["採取率"], "移動": r["移動率"], "交尾": r["交尾率"], "回避": r["回避率"], "戦闘": r["戦闘率"], "捕食": r["捕食率"]}
                valid_actions = {k:v for k,v in actions.items() if not pd.isna(v)}
                if valid_actions:
                    top_action = max(valid_actions, key=valid_actions.get)
                    lines.append(f"**行動経路。** 最も目立つ行動は {top_action}（{fmt(valid_actions[top_action])}）です。採取が高いのに空腹が高いなら資源配置や移動コストで詰まり、交尾が高いのに親出生が低いなら空きマス・近親回避・相手不足で詰まり、回避が高いのに比率が伸びないなら生存はできても繁殖機会を逃している可能性があります。")
                # interacting genes and team mediation
                mediators=[]
                if not pd.isna(r["タカ比率"]):
                    if r["タカ比率"] > avg_hawk + 0.06:
                        mediators.append("タカ遺伝子に偏っており、争奪利得/戦闘コストがこの型の結果を媒介している")
                    elif r["タカ比率"] < avg_hawk - 0.06:
                        mediators.append("ハト寄りで、戦闘回避がこの型の維持に関わっている可能性がある")
                if not pd.isna(r["捕食傾向比率"]):
                    if r["捕食傾向比率"] > avg_pred_gene + 0.03:
                        mediators.append("捕食傾向遺伝子に偏っており、捕食成功率や失敗コストが結果を左右しやすい")
                if not pd.isna(r["赤比率"]) and not pd.isna(r["青比率"]):
                    if r["赤比率"] > r["青比率"] + 0.08:
                        mediators.append("赤チーム環境に偏っており、赤側の資源・密度・内部構成を通じた見かけの優位/劣位が混ざる")
                    elif r["青比率"] > r["赤比率"] + 0.08:
                        mediators.append("青チーム環境に偏っており、青側の資源・密度・内部構成を通じた見かけの優位/劣位が混ざる")
                if not mediators:
                    mediators.append("目立つ同伴遺伝子の偏りは弱く、この型単体の行動評価差または局所環境差を優先して疑う")
                lines.append("**遺伝子間作用。** " + "。".join(mediators) + "。")
                # strength of causal claim
                supports=[]
                if r["比率変化"] > 0 and not pd.isna(r["終盤W"]) and r["終盤W"] >= 1: supports.append("比率変化と終盤Wが同じ方向")
                if r["親として残したコピー"] > r["死亡"]: supports.append("親出生が死亡を上回る")
                if not pd.isna(r["資源収支"]) and r["資源収支"] > 0 and not pd.isna(r["空腹率"]) and r["空腹率"] < avg_hunger: supports.append("資源収支と空腹率が有利")
                if supports:
                    lines.append("**因果候補の強さ。** 根拠は「" + "、".join(supports) + "」です。複数の根拠が同じ方向なら比較的強い候補ですが、単独ランなので最終判断には比較実験が必要です。")
                else:
                    lines.append("**因果候補の強さ。** まだ弱いです。比率、W、親出生、死亡、資源収支が一方向にそろっていないため、初期配置・遺伝的浮動・チーム偏りでも説明できます。")
                st.markdown("\n\n".join(lines))

        # 5) pressures target table
        st.markdown("#### 5. 淘汰圧ごとの作用先")
        pressure_rows = [
            {"淘汰圧":"資源アクセス圧", "利益を受けやすい型":strongest_name(rows,"資源収支"), "被害を受けやすい型":strongest_name(rows,"空腹率"), "どう作用するか":"資源を拾える型は維持・交尾・出生に回せる。資源があるのに空腹な型は、移動・認識・局所配置のどこかで環境と切断されている。"},
            {"淘汰圧":"繁殖出口圧", "利益を受けやすい型":strongest_name(rows,"親として残したコピー"), "被害を受けやすい型":"交尾率が高いのに親出生が少ない型", "どう作用するか":"資源や交尾意図ではなく、相手・近親回避・空きマス・密度がコピー化を止める。"},
            {"淘汰圧":"飢餓/死亡圧", "利益を受けやすい型":strongest_name(rows,"空腹率", reverse=False), "被害を受けやすい型":strongest_name(rows,"死亡"), "どう作用するか":"死亡はコピーを直接減らす。低死亡型は増殖力が弱くても相対的に残る場合がある。"},
            {"淘汰圧":"争奪媒介圧", "利益を受けやすい型":"タカ型またはハト型のうち、争奪ネットの符号と合う型", "被害を受けやすい型":"争奪ネットと逆向きの型", "どう作用するか":f"争奪ネットは {fmt(contest_net,2,True)}。正ならタカ/戦闘寄りが利得を得やすく、負なら戦闘回避が有利になりやすい。"},
            {"淘汰圧":"捕食媒介圧", "利益を受けやすい型":"捕食成功率が高い時の捕食傾向型", "被害を受けやすい型":"捕食成功率が低い時の捕食傾向型", "どう作用するか":f"捕食成功率は {fmt(pred_success)}。捕食は資源獲得にも失敗コストにもなるので、試行回数だけでは判断しない。"},
            {"淘汰圧":"チーム媒介圧", "利益を受けやすい型":f"優位チーム側に偏る型（赤:{red_top} / 青:{blue_top}）", "被害を受けやすい型":"不利な資源・密度・内部構成のチームに偏る型", "どう作用するか":"チーム色は見た目の分類だが、空間配置・資源アクセス・相互作用相手を媒介する。色ではなく、その色の環境条件を読む。"},
        ]
        st.dataframe(pd.DataFrame(pressure_rows), use_container_width=True, hide_index=True)

        # 6) what remains uncertain
        st.markdown("#### 6. まだ断定しないこと")
        st.markdown("""
- **『青が多いから青が強い』とは言いません。** 青が多いのは結果であり、原因は資源アクセス、Gini、局所密度、タカ/ハト構成、哲学型構成、出生出口のどこかにあります。  
- **『カント型だから強い』とも言いません。** カント型が伸びたように見えても、カント型にハト遺伝子が多い、捕食傾向が少ない、有利チームに偏った、死亡が少ないだけ、という別解釈があります。  
- **単独ランでは因果は確定しません。** このレポートは原因候補を絞るものです。強い主張にするには、比較実験で捕食OFF、密度依存OFF、近親回避OFF、通常割合変更、seed反復を行い、同じ傾向が再現するかを見ます。  
- **足りないログもあります。** さらに完全に近づけるなら、チーム別出生/死亡、型別死因、型別移動コスト、型別交尾失敗理由、環境区画別の出生率を追加すると、原因推定がもう一段強くなります。
""")


    def show_public_causal_report_v24():
        """外部向けの統合因果レポート。結果の言い換えを避け、環境と遺伝子の接続を深く説明する。"""
        if len(df) < 3:
            st.info("分析レポートには、少なくとも3世代以上の履歴が必要です。まず数十世代ほど進めてください。")
            return

        # ---------- safe helpers ----------
        def has(col):
            return col in df.columns
        def srs(col, frame=None):
            frame = df if frame is None else frame
            if col not in frame.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(frame[col], errors="coerce")
        def clean(col, frame=None):
            return srs(col, frame).dropna()
        def firstv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[0]) if len(v) else default
        def lastv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[-1]) if len(v) else default
        def meanv(col, default=np.nan, frame=None):
            v = clean(col, frame)
            return float(v.mean()) if len(v) else default
        def sumv(col, default=0.0, frame=None):
            v = clean(col, frame)
            return float(v.sum()) if len(v) else default
        def minv(col, default=np.nan):
            v = clean(col)
            return float(v.min()) if len(v) else default
        def maxv(col, default=np.nan):
            v = clean(col)
            return float(v.max()) if len(v) else default
        def corr(a, b):
            if a not in df.columns or b not in df.columns:
                return np.nan
            d = pd.concat([srs(a), srs(b)], axis=1).dropna()
            if len(d) < 5:
                return np.nan
            try:
                return float(d.iloc[:, 0].corr(d.iloc[:, 1]))
            except Exception:
                return np.nan
        def fmt(x, digits=3, signed=False):
            try:
                if x is None or pd.isna(x):
                    return "—"
                sign = "+" if signed and float(x) >= 0 else ""
                return f"{sign}{float(x):.{digits}f}"
            except Exception:
                return "—"
        def pct(x, digits=1):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return f"{float(x)*100:.{digits}f}%"
            except Exception:
                return "—"
        def arrow_delta(x, digits=3):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return f"{float(x):+.{digits}f}"
            except Exception:
                return "—"
        def rel(value, base, margin):
            if value is None or base is None or pd.isna(value) or pd.isna(base):
                return "不明"
            d = float(value) - float(base)
            if d > margin:
                return "高い"
            if d < -margin:
                return "低い"
            return "近い"
        def trend_word(x, pos=0.03, neg=-0.03):
            if x is None or pd.isna(x):
                return "不明"
            if x > pos:
                return "増加"
            if x < neg:
                return "減少"
            return "ほぼ維持"
        def safe_int(x):
            try:
                return str(int(round(float(x))))
            except Exception:
                return "—"
        def join_sentences(lines):
            return "\n\n".join([str(x) for x in lines if str(x).strip()])
        def get_top(rows, key, reverse=True):
            vals = [r for r in rows if key in r and not pd.isna(r[key])]
            if not vals:
                return None
            return sorted(vals, key=lambda r: r[key], reverse=reverse)[0]

        # ---------- windows ----------
        n = len(df)
        third = max(3, n // 3)
        early = df.head(third)
        middle = df.iloc[third:2*third] if n >= third * 3 else df.iloc[max(0, n//3):max(1, 2*n//3)]
        late = df.tail(third)
        long_recent = df.tail(min(120, n))

        # ---------- global variables ----------
        pop0, pop1 = firstv("個体数（体）", 0), lastv("個体数（体）", 0)
        red0, red1 = firstv("個体数（赤体）", 0), lastv("個体数（赤体）", 0)
        blue0, blue1 = firstv("個体数（青体）", 0), lastv("個体数（青体）", 0)
        pop_w_all = meanv("個体群全体W（増殖率）", np.nan)
        pop_w_late = meanv("個体群全体W（増殖率）", np.nan, late)
        pop_w_early = meanv("個体群全体W（増殖率）", np.nan, early)
        births_late = meanv("出生数（体/世代）", np.nan, late)
        deaths_late = meanv("死亡数（体/世代）", np.nan, late)
        births_total = sumv("出生数（体/世代）", 0.0)
        deaths_total = sumv("死亡数（体/世代）", 0.0)
        bd_ratio_late = births_late / max(deaths_late, 1e-9) if not pd.isna(births_late) and not pd.isna(deaths_late) else np.nan

        res0, res1 = firstv("資源総量（単位）", np.nan), lastv("資源総量（単位）", np.nan)
        res_min, res_max = minv("資源総量（単位）", np.nan), maxv("資源総量（単位）", np.nan)
        bag0, bag1 = firstv("平均所持資源（単位/体）", np.nan), lastv("平均所持資源（単位/体）", np.nan)
        gini_all = meanv("資源格差Gini（0-1）", np.nan)
        gini_late = meanv("資源格差Gini（0-1）", np.nan, late)
        resource_cells_late = meanv("資源マス割合（0-1）", np.nan, late)
        density_late = meanv("平均局所密度（体/近傍）", np.nan, late)
        mate_rate_late = meanv("交尾成立率（0-1）", np.nan, late)
        kin_block_late = meanv("近親交配回避（回/世代）", 0.0, late)
        density_block_late = meanv("過密で抑制された出生候補（回/世代）", 0.0, late)
        pred_attempt_late = meanv("捕食試行（回/世代）", 0.0, late)
        pred_success_late = meanv("捕食成功率（0-1）", np.nan, late)
        contest_gain_late = meanv("争奪で得たV合計（単位/世代）", 0.0, late)
        contest_cost_late = meanv("争奪で支払ったC合計（単位/世代）", 0.0, late)
        contest_net_late = contest_gain_late - contest_cost_late
        normal0, normal1 = firstv("通常個体数（体）", np.nan), lastv("通常個体数（体）", np.nan)
        philo0, philo1 = firstv("哲学個体数（体）", np.nan), lastv("哲学個体数（体）", np.nan)

        c_birth_bag = corr("出生数（体/世代）", "平均所持資源（単位/体）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_birth_density = corr("出生数（体/世代）", "平均局所密度（体/近傍）")
        c_death_bag = corr("死亡数（体/世代）", "平均所持資源（単位/体）")
        c_birth_mate = corr("出生数（体/世代）", "交尾成立率（0-1）")

        # ---------- gene profiles ----------
        labels = [PHILO_LABELS[k] for k in sorted(PHILO_LABELS.keys()) if isinstance(PHILO_LABELS.get(k), str)]
        total_parent_births = 0.0
        for lab in labels:
            total_parent_births += sumv(f"{lab} 親参加:実出生（回/世代）", 0.0)
        rows = []
        for lab in labels:
            n0 = firstv(f"{lab} 数（体）", np.nan)
            n1 = lastv(f"{lab} 数（体）", np.nan)
            r0 = firstv(f"{lab} 比率（0-1）", np.nan)
            r1 = lastv(f"{lab} 比率（0-1）", np.nan)
            rd = r1 - r0 if not pd.isna(r0) and not pd.isna(r1) else np.nan
            row = {
                "型": lab,
                "初期数": n0, "最新数": n1, "数変化": (n1 - n0 if not pd.isna(n0) and not pd.isna(n1) else np.nan),
                "初期比率": r0, "最新比率": r1, "比率変化": rd,
                "平均W": meanv(f"{lab} W", np.nan),
                "前期W": meanv(f"{lab} W", np.nan, early),
                "終盤W": meanv(f"{lab} W", np.nan, late),
                "資源収支": meanv(f"{lab} 資源収支ネット（単位/世代）", np.nan, late),
                "採取獲得": meanv(f"{lab} 採取獲得（単位/世代）", np.nan, late),
                "移動支払": meanv(f"{lab} 移動支払（単位/世代）", np.nan, late),
                "維持支払": meanv(f"{lab} 維持支払（単位/世代）", np.nan, late),
                "空腹率": meanv(f"{lab} 空腹個体比率（0-1）", np.nan, late),
                "平均所持資源": meanv(f"{lab} 平均所持資源（単位/体）", np.nan, late),
                "足元資源": meanv(f"{lab} 平均足元資源（単位/マス）", np.nan, late),
                "局所密度": meanv(f"{lab} 平均局所密度（体/近傍）", np.nan, late),
                "出生": sumv(f"{lab} 実出生（体/世代）", 0.0),
                "親出生": sumv(f"{lab} 親参加:実出生（回/世代）", 0.0),
                "死亡": sumv(f"{lab} 死亡（体/世代）", 0.0),
                "交尾試行": meanv(f"{lab} 交尾試行参加（回/世代）", np.nan, late),
                "交尾成功": meanv(f"{lab} 交尾成功参加（回/世代）", np.nan, late),
                "捕食試行": meanv(f"{lab} 捕食試行（回/世代）", np.nan, late),
                "捕食成功": meanv(f"{lab} 捕食成功（回/世代）", np.nan, late),
                "捕食失敗": meanv(f"{lab} 捕食失敗（回/世代）", np.nan, late),
                "タカ比率": meanv(f"{lab} タカ比率（0-1）", np.nan, late),
                "捕食傾向比率": meanv(f"{lab} 捕食傾向比率（0-1）", np.nan, late),
                "赤比率": meanv(f"{lab} 赤比率（0-1）", np.nan, late),
                "青比率": meanv(f"{lab} 青比率（0-1）", np.nan, late),
                "採取率": meanv(f"{lab} 行動率:採取（0-1）", np.nan, late),
                "移動率": meanv(f"{lab} 行動率:移動（0-1）", np.nan, late),
                "交尾率": meanv(f"{lab} 行動率:交尾（0-1）", np.nan, late),
                "回避率": meanv(f"{lab} 行動率:回避（0-1）", np.nan, late),
                "戦闘率": meanv(f"{lab} 行動率:戦闘（0-1）", np.nan, late),
                "捕食率": meanv(f"{lab} 行動率:捕食（0-1）", np.nan, late),
            }
            row["親出生シェア"] = row["親出生"] / max(total_parent_births, 1e-9)
            # Not a truth score; it only chooses display order. Explanation below remains evidence-based.
            score = 0.0
            if not pd.isna(row["比率変化"]): score += row["比率変化"] * 10
            if not pd.isna(row["終盤W"]): score += (row["終盤W"] - 1.0) * 5
            if not pd.isna(row["親出生シェア"]): score += row["親出生シェア"] * 1.5
            if not pd.isna(row["資源収支"]): score += np.tanh(row["資源収支"] / 50.0)
            if not pd.isna(row["空腹率"]): score -= row["空腹率"]
            if row["親出生"] < row["死亡"] and row["死亡"] > 0: score -= 0.5
            row["表示順スコア"] = score
            rows.append(row)
        gene_df = pd.DataFrame(rows)
        if len(gene_df):
            gene_df = gene_df.sort_values("表示順スコア", ascending=False)
        avg = {}
        for key in ["資源収支","空腹率","平均所持資源","足元資源","局所密度","タカ比率","捕食傾向比率","親出生シェア","交尾成功"]:
            vals = [r[key] for r in rows if key in r and not pd.isna(r[key])]
            avg[key] = float(np.mean(vals)) if vals else np.nan

        top_ratio = get_top(rows, "比率変化", True)
        bottom_ratio = get_top(rows, "比率変化", False)
        top_w = get_top(rows, "終盤W", True)
        top_parent = get_top(rows, "親出生シェア", True)
        top_resource = get_top(rows, "資源収支", True)
        high_hunger = get_top(rows, "空腹率", True)
        low_hunger = get_top(rows, "空腹率", False)

        st.markdown("### 総合分析レポート")
        explain_box(
            "このレポートが説明しようとしていること",
            "ここで知りたいのは『どの色・どの型が多いか』ではありません。多いことは結果です。知りたいのは、その結果を作った仕組みです。ネオライフゲームでは、遺伝子は直接に個体数を増やす魔法ではなく、個体の判断の癖を少し変えるものです。ある遺伝子は資源へ寄りやすくし、ある遺伝子は危険を避けやすくし、ある遺伝子は交尾や捕食や争奪の評価を変えます。その判断の差が、盤面上の資源量、資源の偏り、局所密度、相手との出会い、近親回避、空きマス、捕食成功率、争奪利得とぶつかったとき、初めて出生や死亡の差になります。だからこのレポートでは、**遺伝子 → 行動傾向 → 環境との接触 → ボトルネック → 出生/死亡 → コピー数** という流れで読みます。"
        )

        # ---------- glossary for external readers ----------
        with st.expander("用語の読み方：初めて見る人向け", expanded=True):
            st.markdown("""
**W** はコピー数の増殖率です。1より大きければ前世代より増え、1より小さければ減っています。ただしWだけでは原因は分かりません。Wは、出生、死亡、資源不足、交尾失敗、密度、争奪、捕食が合わさった最終結果です。

**資源総量** は盤面にある資源の量です。**平均所持資源** は個体が実際に持っている資源です。この2つがズレると重要です。盤面に資源が多いのに個体が貧しいなら、資源が存在していても個体の行動・視野・移動・局所配置とつながっていません。

**Gini** は資源格差です。平均資源が同じでも、Giniが高いと一部だけが豊かで、低資源個体が死にやすくなります。これは『資源量』ではなく『資源の分配』が作る淘汰圧です。

**局所密度** は近くにどれだけ個体がいるかです。密度が高いと出会いやすくなりますが、空きマス不足、資源競争、過密出生抑制も起きます。つまり密度は利益にもコストにもなります。

**タカ/ハト** は争奪場面の遺伝子です。争奪の利得がコストを上回る環境ではタカ性が効きますが、争奪コストが大きい環境ではハト性、つまり戦闘回避が生存に効きます。

**捕食傾向** は捕食を選びやすくする遺伝子です。捕食は成功すれば資源になりますが、失敗や遭遇環境によってはコストになります。したがって、捕食傾向は捕食成功率と一緒に読まないと意味がありません。

**哲学型** は思想家そのものの再現ではなく、行動評価関数です。ヒューム型は観察・局所経験、ストア型は自己制御と危険回避、デカルト型は明確な安全と利得、カント型は短期搾取より安定性を重く見る、という形で操作化されています。通常個体はそれらの補正を持たない対照群です。
""")

        # ---------- narrative of the world ----------
        st.markdown("#### 1. この世界は、個体に何を要求しているか")
        world_lines = []
        world_lines.append(f"このランでは、個体数は **{safe_int(pop0)}体 → {safe_int(pop1)}体**、全期間平均Wは **{fmt(pop_w_all)}**、終盤Wは **{fmt(pop_w_late)}** です。ここで重要なのは、これは『世界全体の結果』であって、まだ原因ではないという点です。終盤Wが1を下回るなら、世界全体がコピーを削る方向に傾いており、終盤Wが1を上回るなら、出生か生存のどこかがうまく回っています。")
        if not pd.isna(pop_w_late):
            if pop_w_late < 0.97:
                world_lines.append("終盤Wが1を下回っています。この場合、遺伝子間の優劣を見る前に、環境そのものが厳しい可能性があります。厳しい世界では、派手に増やす遺伝子より、死亡を避ける遺伝子、資源を失いにくい遺伝子、少ない繁殖機会を確実に通す遺伝子が残りやすくなります。")
            elif pop_w_late > 1.03:
                world_lines.append("終盤Wが1を上回っています。この場合、環境にはまだ増殖余地があります。増殖余地がある世界では、危険回避だけでなく、資源を早く回収する遺伝子、交尾出口に到達する遺伝子、密度を利用できる遺伝子が有利になりやすいです。")
            else:
                world_lines.append("終盤Wはおおむね1付近です。これは外部から見ると地味ですが、研究上はかなり面白い状態です。世界全体が即崩壊も爆発増殖もしないため、個体群全体の勢いではなく、内部の遺伝子差が表に出やすくなります。")
        world_lines.append(f"出生と死亡を見ると、全期間の出生合計は **{safe_int(births_total)}**、死亡合計は **{safe_int(deaths_total)}**、終盤の出生/死亡比は **{fmt(bd_ratio_late,2)}** です。出生/死亡比は、単に個体数の増減を言い換えるための値ではなく、資源や交尾や密度が最終的に繁殖出口を通れたかを示す入口です。")
        if not pd.isna(bd_ratio_late):
            if bd_ratio_late < 0.9:
                world_lines.append("終盤の出生/死亡比が低いので、現在の主な制約は『増やすこと』よりも『増やせないこと』にあります。資源を持っていても相手に会えない、近親回避で弾かれる、過密で子を置けない、死亡が繁殖前に来る、といった繁殖出口の詰まりを疑うべきです。")
            elif bd_ratio_late > 1.1:
                world_lines.append("終盤の出生/死亡比が高いので、現在は繁殖出口がある程度開いています。この場合、どの遺伝子がその出口に多く到達しているか、つまり親参加実出生と交尾成功の偏りが重要になります。")
        world_lines.append(f"資源環境では、資源総量が **{safe_int(res0)} → {safe_int(res1)}**、最小 **{safe_int(res_min)}**、最大 **{safe_int(res_max)}**、平均所持資源が **{fmt(bag0,2)} → {fmt(bag1,2)}**、終盤Giniが **{fmt(gini_late)}** です。資源総量だけを見ても原因は分かりません。資源総量は『環境側の在庫』で、平均所持資源は『個体が実際に回収できた資源』です。この2つが同じ向きなら資源と行動が接続していますが、ズレるなら資源配置・視野・移動コスト・局所密度が選択圧になっています。")
        if not pd.isna(res1) and not pd.isna(res0) and not pd.isna(bag1) and not pd.isna(bag0):
            if res1 > res0 and bag1 <= bag0:
                world_lines.append("資源総量は増えているのに、平均所持資源は伸びていません。このとき『資源不足』というより『資源に届かない』ことが問題です。資源は盤面にあるが、個体の視野・移動・局所配置がそこへ接続していないため、資源アクセス圧が強くなります。")
            elif res1 < res0 and bag1 > bag0:
                world_lines.append("資源総量は減っているのに、平均所持資源は上がっています。これは個体が環境ストックをうまく回収している状態です。ただし長く続くと、環境在庫を削り、後の世代で資源枯渇圧を作る可能性があります。")
            elif res1 > res0 and bag1 > bag0:
                world_lines.append("資源総量と平均所持資源がともに増えています。環境供給と個体回収が同時に回っているので、死亡圧は資源そのものよりも、局所密度、交尾制約、争奪・捕食のリスクから来ている可能性が上がります。")
            elif res1 < res0 and bag1 < bag0:
                world_lines.append("資源総量と平均所持資源がともに減っています。環境在庫も個体内資源も弱っているため、飢餓・移動コスト・過剰採取が連鎖している可能性があります。この世界では資源獲得型が有利になる一方で、移動しすぎる型は逆に損をするかもしれません。")
        if not pd.isna(gini_late):
            if gini_late > 0.45:
                world_lines.append("終盤Giniが高めです。これは平均資源だけでは見えない重要な圧です。平均的には資源があっても、一部個体に偏っているなら、貧しい個体は維持コストを払えず死にます。この場合、有利なのは『平均資源が高い型』だけでなく、『低資源に落ちにくい型』です。")
            elif gini_late < 0.25:
                world_lines.append("終盤Giniは低めです。資源格差が弱いなら、死亡圧は資源分配よりも、交尾機会、密度、移動コスト、捕食・争奪イベントから来ている可能性があります。")
        explain_box("世界全体の物語", join_sentences(world_lines))

        # ---------- environmental pressure table ----------
        st.markdown("#### 2. 環境圧の整理：何が淘汰圧になっているか")
        pressure_rows = []
        pressure_rows.append({
            "環境圧": "資源アクセス圧",
            "見ている数値": f"資源総量 {safe_int(res1)} / 平均所持資源 {fmt(bag1,2)} / Gini {fmt(gini_late)}",
            "原因として読む条件": "資源総量と平均所持資源がズレる、またはGiniが高いとき。",
            "遺伝子への作用": "採取へ寄る型、移動コストを抑える型、足元資源の高い場所に残れる型が有利になる。資源を見つけても移動で失う型は不利。",
            "確認する列": "型別資源収支、空腹率、足元資源、移動支払、採取獲得"
        })
        pressure_rows.append({
            "環境圧": "繁殖出口圧",
            "見ている数値": f"交尾成立率 {fmt(mate_rate_late)} / 近親回避 {fmt(kin_block_late,1)} / 過密抑制 {fmt(density_block_late,1)}",
            "原因として読む条件": "資源があるのに出生が伸びない、交尾行動があるのに親出生が少ないとき。",
            "遺伝子への作用": "生存だけではなく、相手・空きマス・非近親相手へ到達できる型が有利になる。回避型は死ににくいが、交尾出口から遠ざかると増えない。",
            "確認する列": "親参加実出生、交尾成功参加、交尾試行、局所密度、近親回避、過密抑制"
        })
        pressure_rows.append({
            "環境圧": "密度・空間圧",
            "見ている数値": f"終盤局所密度 {fmt(density_late,2)} / 出生×密度相関 {fmt(c_birth_density,3,True)}",
            "原因として読む条件": "個体が多いのに出生が伸びない、または資源が局所的に枯れるとき。",
            "遺伝子への作用": "密度を利用して相手に会う型は有利。過密で子を置けない型、局所資源を食い尽くす型は不利。",
            "確認する列": "型別局所密度、過密出生抑制、交尾成功、資源収支"
        })
        pressure_rows.append({
            "環境圧": "争奪媒介圧",
            "見ている数値": f"争奪ネット {fmt(contest_net_late,2,True)} = 利得 {fmt(contest_gain_late,2)} - コスト {fmt(contest_cost_late,2)}",
            "原因として読む条件": "タカ比率の高い型が増える/減る、争奪ネットが大きく正負に振れるとき。",
            "遺伝子への作用": "争奪ネットが正ならタカ性が資源獲得を増幅する。負ならタカ性は消耗要因になり、ハト性・回避性が有利になる。",
            "確認する列": "型別タカ比率、戦闘率、戦闘獲得、戦闘損失、死亡"
        })
        pressure_rows.append({
            "環境圧": "捕食媒介圧",
            "見ている数値": f"捕食試行 {fmt(pred_attempt_late,2)} / 捕食成功率 {fmt(pred_success_late)}",
            "原因として読む条件": "捕食傾向の高い型が伸びる/減る、資源収支と死亡が同時に変わるとき。",
            "遺伝子への作用": "捕食成功率が高ければ捕食傾向は資源獲得経路になる。成功率が低いなら失敗コストや危険接触を増やす経路になる。",
            "確認する列": "型別捕食傾向比率、捕食試行、捕食成功、捕食失敗、捕食獲得、死亡"
        })
        pressure_rows.append({
            "環境圧": "チーム媒介圧",
            "見ている数値": f"赤 {safe_int(red0)}→{safe_int(red1)} / 青 {safe_int(blue0)}→{safe_int(blue1)}",
            "原因として読む条件": "赤青の数に差が出るが、資源・Gini・内部遺伝子構成も同時に違うとき。",
            "遺伝子への作用": "チーム色そのものではなく、その色が置かれた空間、資源、相互作用相手、同伴遺伝子が媒介して特定型を押し上げる。",
            "確認する列": "赤/青平均資源、赤/青Gini、赤/青タカ比率、チーム内の型構成"
        })
        st.dataframe(pd.DataFrame(pressure_rows), use_container_width=True, hide_index=True)

        # ---------- gene ranking table ----------
        st.markdown("#### 3. 遺伝子型ごとの現在地")
        if len(gene_df):
            display_cols = ["型","初期数","最新数","数変化","初期比率","最新比率","比率変化","終盤W","親出生","死亡","資源収支","空腹率","足元資源","局所密度","タカ比率","捕食傾向比率","赤比率","青比率"]
            display_cols = [c for c in display_cols if c in gene_df.columns]
            shown = gene_df[display_cols].copy()
            for c in ["初期比率","最新比率","比率変化","終盤W","空腹率","タカ比率","捕食傾向比率","赤比率","青比率"]:
                if c in shown.columns:
                    shown[c] = shown[c].map(lambda x: fmt(x,3, signed=(c=="比率変化")))
            for c in ["資源収支","足元資源","局所密度"]:
                if c in shown.columns:
                    shown[c] = shown[c].map(lambda x: fmt(x,2, signed=(c=="資源収支")))
            st.dataframe(shown, use_container_width=True, hide_index=True)
        explain_box(
            "この表をどう読むか",
            "比率変化だけで優劣を決めないでください。比率が増えた型でも、親出生が少なく死亡も少ないだけなら『増殖した』というより『他型が減ったことで相対的に残った』可能性があります。逆に、比率が減った型でも親出生が多いなら、出生はできているが死亡や競争で削られているのかもしれません。資源収支、空腹率、局所密度、タカ比率、捕食傾向、赤青偏りを一緒に見ると、その型が何を通じて増減したかが見えます。"
        )

        # ---------- strong narrative hypotheses ----------
        st.markdown("#### 4. 現時点での有力な因果仮説")
        hypothesis_lines = []
        if top_ratio:
            lab = top_ratio["型"]
            hypothesis_lines.append(f"**頻度上昇が最も大きい型は {lab}** です。ただし、これは『{lab}が本質的に強い』という意味ではありません。原因候補は、①終盤Wが維持線を超えている、②親として子を残している、③死亡が相対的に少ない、④資源収支がよい、⑤有利なチーム・タカ/ハト・捕食遺伝子と同伴している、のどれかです。")
            if not pd.isna(top_ratio.get("終盤W", np.nan)):
                if top_ratio["終盤W"] >= 1:
                    hypothesis_lines.append(f"{lab}の終盤Wは **{fmt(top_ratio['終盤W'])}** なので、少なくとも終盤ではコピー維持に失敗していません。ここからさらに、親出生が高ければ『繁殖で増えた』、死亡が低ければ『生存で残った』、資源収支が高ければ『資源経路で支えられた』と分解します。")
                else:
                    hypothesis_lines.append(f"{lab}は比率では伸びていますが、終盤Wは **{fmt(top_ratio['終盤W'])}** です。これは、その型が絶対的に増殖しているというより、他型の減少によって相対的に比率が上がった可能性を残します。")
        if bottom_ratio:
            lab = bottom_ratio["型"]
            hypothesis_lines.append(f"**頻度低下が最も大きい型は {lab}** です。低下の原因は、資源を取れない、空腹が高い、死亡が多い、親出生に参加できない、または不利なチーム/同伴遺伝子に偏った、のどれかです。ここで重要なのは『減ったから弱い』ではなく、『どの環境圧で削られたか』です。")
        if top_parent:
            hypothesis_lines.append(f"親としてコピーを残す経路では **{top_parent['型']}** が目立ちます。これは繁殖出口に到達している型です。もしこの型の比率が伸びていないなら、出生後または親世代の死亡・資源損失・チーム不利で削られている可能性があります。")
        if top_resource:
            hypothesis_lines.append(f"資源経路では **{top_resource['型']}** が目立ちます。資源収支がよい型は、維持コストを払い、交尾に回す余剰を作りやすいです。ただし、資源収支がよくても親出生が低ければ、資源獲得後に繁殖出口で詰まっています。")
        if high_hunger:
            hypothesis_lines.append(f"空腹圧を強く受けていそうなのは **{high_hunger['型']}** です。空腹率が高い型は、死亡に直結しなくても、交尾や移動に使う余剰が減り、長期的にはコピー数で不利になります。")
        if not hypothesis_lines:
            hypothesis_lines.append("まだ十分な履歴がないか、型別ログが不足しているため、有力な因果仮説は出せません。世代数を増やすと、資源・密度・出生・死亡の経路が見えやすくなります。")
        explain_box("因果仮説の本文", join_sentences(hypothesis_lines))

        # ---------- team analysis ----------
        st.markdown("#### 5. 赤チーム・青チーム：どちらが多いかではなく、なぜ差が生まれたか")
        red_res, blue_res = meanv("赤 平均所持資源", np.nan, late), meanv("青 平均所持資源", np.nan, late)
        red_g, blue_g = meanv("赤 Gini", np.nan, late), meanv("青 Gini", np.nan, late)
        red_h, blue_h = meanv("赤 タカ比率（0-1）", np.nan, late), meanv("青 タカ比率（0-1）", np.nan, late)
        # type leaning to teams: red_ratio and blue_ratio inside each row mean within-type team split.
        red_lean = None
        blue_lean = None
        lean_rows = []
        for r in rows:
            rr, br = r.get("赤比率", np.nan), r.get("青比率", np.nan)
            if not pd.isna(rr) and not pd.isna(br):
                lean_rows.append({"型": r["型"], "赤寄り": rr - br, "青寄り": br - rr})
        if lean_rows:
            red_lean = sorted(lean_rows, key=lambda x: x["赤寄り"], reverse=True)[0]
            blue_lean = sorted(lean_rows, key=lambda x: x["青寄り"], reverse=True)[0]
        team_table = pd.DataFrame([
            {"比較軸":"個体数変化", "赤":f"{safe_int(red0)}→{safe_int(red1)}", "青":f"{safe_int(blue0)}→{safe_int(blue1)}", "原因としての読み方":"これは原因ではなく結果。ここから資源・格差・争奪・内部構成を調べる。"},
            {"比較軸":"平均所持資源", "赤":fmt(red_res,2), "青":fmt(blue_res,2), "原因としての読み方":"多い側が平均資源も高いなら資源アクセスが支えている可能性。逆なら、資源以外の要因が候補。"},
            {"比較軸":"資源格差Gini", "赤":fmt(red_g,3), "青":fmt(blue_g,3), "原因としての読み方":"Giniが低い側は貧困個体を作りにくい。平均資源が同じでも格差が死亡圧を変える。"},
            {"比較軸":"タカ比率", "赤":fmt(red_h,3), "青":fmt(blue_h,3), "原因としての読み方":"争奪ネットが正なら高タカ側が利得を得やすい。負なら高タカ側が消耗しやすい。"},
            {"比較軸":"内部の型偏り", "赤":red_lean["型"] if red_lean else "—", "青":blue_lean["型"] if blue_lean else "—", "原因としての読み方":"チーム差が色そのものではなく、チーム内に偏った行動型によって生まれた可能性を見る。"},
        ])
        st.dataframe(team_table, use_container_width=True, hide_index=True)
        team_lines = []
        if red1 > blue1 * 1.08:
            side, other = "赤", "青"
            side_res, other_res = red_res, blue_res
            side_g, other_g = red_g, blue_g
            side_h, other_h = red_h, blue_h
        elif blue1 > red1 * 1.08:
            side, other = "青", "赤"
            side_res, other_res = blue_res, red_res
            side_g, other_g = blue_g, red_g
            side_h, other_h = blue_h, red_h
        else:
            side = other = None
            side_res = other_res = side_g = other_g = side_h = other_h = np.nan
        if side is None:
            team_lines.append("赤青の個体数差は大きくありません。この場合、チーム色そのものを原因として扱うより、各チーム内部の遺伝子構成や局所資源配置を見る方が重要です。")
        else:
            team_lines.append(f"最新数だけ見ると **{side}チーム** が多いです。しかし、これはまだ結果です。原因として疑うべきなのは、{side}側が資源を多く持つのか、資源格差が低いのか、タカ/ハト構成が争奪環境に合っているのか、あるいは有利な行動型が{side}側に偏っているのか、という点です。")
            if not pd.isna(side_res) and not pd.isna(other_res):
                if side_res > other_res + 0.5:
                    team_lines.append(f"平均所持資源は{side}側が高めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。この場合、{side}優位は資源アクセスによって支えられた可能性があります。資源を多く持つ個体は維持コストを払いやすく、交尾や移動に回せる余剰も持ちやすいためです。")
                elif side_res < other_res - 0.5:
                    team_lines.append(f"平均所持資源はむしろ{side}側が低めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。それでも{side}が多いなら、資源量ではなく、死亡回避、交尾出口、空間配置、または内部遺伝子構成が数の差を作っている可能性があります。")
                else:
                    team_lines.append(f"平均所持資源は赤青で大差ありません。この場合、チーム差を資源量だけでは説明しにくく、Gini、タカ/ハト構成、局所密度、交尾出口の差が候補になります。")
            if not pd.isna(side_g) and not pd.isna(other_g):
                if side_g < other_g - 0.03:
                    team_lines.append(f"{side}側はGiniが低めです。これは重要です。平均資源が同程度でも、資源が均等に配られるほど低資源で脱落する個体が減ります。つまり{side}側は『豊かだから残った』だけでなく、『貧しい個体を作りにくかったから残った』可能性があります。")
                elif side_g > other_g + 0.03:
                    team_lines.append(f"{side}側はGiniが高めです。数では優位でも、内部に資源格差を抱えています。この場合、現在の優位は長期的には不安定で、富んだ少数個体に支えられている可能性があります。")
            if not pd.isna(side_h) and not pd.isna(other_h):
                if side_h > other_h + 0.05 and contest_net_late > 0:
                    team_lines.append(f"{side}側はタカ比率が高く、争奪ネットも正です。これは、争奪がコストではなく利得として働き、{side}側の個体が資源を奪って維持・繁殖へ回した可能性を示します。")
                elif side_h > other_h + 0.05 and contest_net_late <= 0:
                    team_lines.append(f"{side}側はタカ比率が高い一方、争奪ネットは正ではありません。これは、タカ性が優位の原因というより、むしろ消耗リスクになっている可能性があります。それでも{side}が多いなら、別の要因がタカのコストを上回っているはずです。")
                elif side_h < other_h - 0.05 and contest_net_late <= 0:
                    team_lines.append(f"{side}側はタカ比率が低く、争奪ネットも弱い/負です。この場合、戦闘を避けたことで無駄な消耗を減らし、結果としてコピー維持に有利だった可能性があります。")
        explain_box("チーム差の因果説明", join_sentences(team_lines))

        # ---------- detailed gene stories ----------
        st.markdown("#### 6. 各遺伝子型の詳しい物語")
        theory_by_label = {v: PHILO_THEORY.get(k, "") for k, v in PHILO_LABELS.items()}
        if len(gene_df):
            for _, r in gene_df.iterrows():
                lab = str(r["型"])
                expanded = lab in [str(top_ratio["型"]) if top_ratio else "", str(bottom_ratio["型"]) if bottom_ratio else ""]
                with st.expander(f"{lab}：この型は何によって押し上げられ、何に削られているか", expanded=expanded):
                    lines = []
                    theory = theory_by_label.get(lab, "")
                    if theory:
                        lines.append(f"**この型の意味。** {theory} ここでの型名は思想家本人を再現するものではなく、行動評価の癖です。したがって『思想として正しいか』ではなく、『この評価の癖が、環境内でどの行動と結びつき、コピー数にどう返ったか』を見ます。")
                    lines.append(f"**観察された結果。** 初期数は {safe_int(r['初期数'])}、最新数は {safe_int(r['最新数'])}、比率は {pct(r['初期比率'])} から {pct(r['最新比率'])} へ変化しました。比率変化は {arrow_delta(r['比率変化'])}、終盤Wは {fmt(r['終盤W'])} です。これは現象の入口であって、まだ説明ではありません。")
                    # resource mechanism
                    res_lines = []
                    if not pd.isna(r["資源収支"]):
                        if r["資源収支"] > avg.get("資源収支", 0) + 2:
                            res_lines.append(f"終盤の資源収支は {fmt(r['資源収支'],2,True)} で、全型平均より高めです。これは、この型が環境資源を個体内資源へ変換する経路を持っている可能性を示します。")
                        elif r["資源収支"] < avg.get("資源収支", 0) - 2:
                            res_lines.append(f"終盤の資源収支は {fmt(r['資源収支'],2,True)} で、全型平均より低めです。資源を見つけられない、移動で失う、維持コストが重い、あるいは危険行動で損をしている可能性があります。")
                        else:
                            res_lines.append(f"終盤の資源収支は {fmt(r['資源収支'],2,True)} で、平均から大きく外れていません。資源経路だけでこの型の増減を説明するのは弱く、繁殖出口や死亡回避を見る必要があります。")
                    if not pd.isna(r["足元資源"]):
                        if r["足元資源"] > avg.get("足元資源", 0) + 0.3:
                            res_lines.append(f"足元資源が高めです。これは、この型が資源の近くに留まりやすいか、資源のある場所へ移動しやすいことを示します。")
                        elif r["足元資源"] < avg.get("足元資源", 0) - 0.3:
                            res_lines.append(f"足元資源が低めです。この型は、資源のある局所環境に乗れていない可能性があります。")
                    if not pd.isna(r["空腹率"]):
                        if r["空腹率"] > avg.get("空腹率", 0) + 0.05:
                            res_lines.append(f"空腹率が高めです。空腹は即死亡だけでなく、移動・交尾・維持に使う余剰を奪うため、長期的なコピー数を下げます。")
                        elif r["空腹率"] < avg.get("空腹率", 0) - 0.05:
                            res_lines.append(f"空腹率が低めです。これは死亡回避だけでなく、繁殖に必要な資源余剰を残しやすいという意味でも有利です。")
                    if not res_lines:
                        res_lines.append("資源関連ログが不足しているため、資源経路はまだ読めません。")
                    lines.append("**資源・環境経路。** " + " ".join(res_lines))
                    # reproduction mechanism
                    rep_lines = []
                    if r["親出生"] > r["死亡"]:
                        rep_lines.append(f"親として残したコピー {fmt(r['親出生'],0)} が死亡 {fmt(r['死亡'],0)} を上回っています。この型は、生き残るだけでなく繁殖出口を通ってコピーを作っています。")
                    elif r["親出生"] == 0 and r["死亡"] > 0:
                        rep_lines.append(f"親として残したコピーがほぼなく、死亡は {fmt(r['死亡'],0)} あります。この型は繁殖出口に届いていないため、相手不足、資源不足、近親回避、過密、死亡タイミングを疑います。")
                    else:
                        rep_lines.append(f"親出生 {fmt(r['親出生'],0)} と死亡 {fmt(r['死亡'],0)} の差が大きくありません。この型の比率変化は、出生力だけでなく他型の減少や局所配置に左右されている可能性があります。")
                    if not pd.isna(r["交尾率"]) and not pd.isna(r["交尾成功"]):
                        if r["交尾率"] > 0.05 and r["交尾成功"] <= avg.get("交尾成功", 0):
                            rep_lines.append("交尾行動があるのに交尾成功が平均以下なら、意思ではなく出口で詰まっています。相手が近親、空きマスがない、局所密度が悪い、資源不足で成立しない、といった環境側の制限が候補です。")
                        elif r["交尾成功"] > avg.get("交尾成功", 0):
                            rep_lines.append("交尾成功が平均より高めなので、この型は相手・資源・空きマスの条件を比較的通過できています。")
                    lines.append("**繁殖経路。** " + " ".join(rep_lines))
                    # death/survival mechanism
                    surv_lines = []
                    if not pd.isna(r["回避率"]):
                        if r["回避率"] > 0.08:
                            surv_lines.append("回避率が高めです。これは死亡を減らす方向に働く可能性がありますが、同時に資源獲得や交尾機会から遠ざかることもあります。")
                    if not pd.isna(r["戦闘率"]):
                        if r["戦闘率"] > 0.05:
                            if contest_net_late > 0:
                                surv_lines.append("戦闘率があり、争奪ネットが正なので、争奪が資源獲得として働いている可能性があります。")
                            else:
                                surv_lines.append("戦闘率がある一方、争奪ネットは正ではありません。戦闘が消耗や死亡圧に変わっている可能性があります。")
                    if not pd.isna(r["捕食率"]):
                        if r["捕食率"] > 0.03:
                            if not pd.isna(pred_success_late) and pred_success_late > 0.5:
                                surv_lines.append("捕食率があり、捕食成功率も高めなので、捕食は資源獲得経路になっている可能性があります。")
                            else:
                                surv_lines.append("捕食率がある一方、捕食成功率は高いとは言えません。捕食傾向が危険接触や失敗コストを増やしている可能性があります。")
                    if not surv_lines:
                        surv_lines.append("目立つ危険行動は強くありません。生存差があるなら、危険行動ではなく資源不足・密度・チーム環境が原因かもしれません。")
                    lines.append("**危険・生存経路。** " + " ".join(surv_lines))
                    # gene-gene and team mediation
                    med_lines = []
                    if not pd.isna(r["タカ比率"]):
                        if r["タカ比率"] > avg.get("タカ比率", 0) + 0.06:
                            if contest_net_late > 0:
                                med_lines.append("この型はタカ遺伝子に偏り、争奪ネットも正です。したがって、哲学型そのものだけでなく、タカ性が資源獲得を増幅している可能性があります。")
                            else:
                                med_lines.append("この型はタカ遺伝子に偏っていますが、争奪ネットは正ではありません。タカ性は優位の原因ではなく、むしろコストとして働いている可能性があります。")
                        elif r["タカ比率"] < avg.get("タカ比率", 0) - 0.06:
                            if contest_net_late <= 0:
                                med_lines.append("この型はハト寄りで、争奪ネットも弱い/負です。戦闘を避ける同伴遺伝子が生存を支えている可能性があります。")
                            else:
                                med_lines.append("この型はハト寄りですが、争奪ネットは正です。この場合、争奪利得を取り逃がしている可能性もあります。")
                    if not pd.isna(r["捕食傾向比率"]):
                        if r["捕食傾向比率"] > avg.get("捕食傾向比率", 0) + 0.03:
                            med_lines.append("捕食傾向遺伝子に偏っています。捕食成功率が高ければ資源獲得を増幅し、低ければ失敗コストを増幅します。")
                    if not pd.isna(r["赤比率"]) and not pd.isna(r["青比率"]):
                        if r["赤比率"] > r["青比率"] + 0.08:
                            med_lines.append("赤チームに偏っています。この型の結果には、赤側の資源・格差・相互作用相手の条件が混ざっている可能性があります。")
                        elif r["青比率"] > r["赤比率"] + 0.08:
                            med_lines.append("青チームに偏っています。この型の結果には、青側の資源・格差・相互作用相手の条件が混ざっている可能性があります。")
                    if not med_lines:
                        med_lines.append("目立つ同伴遺伝子・チーム偏りは弱めです。この場合、この型の行動評価そのもの、または局所環境差を優先して見ます。")
                    lines.append("**遺伝子間作用・チーム媒介。** " + " ".join(med_lines))
                    # conclusion and test
                    test_lines = []
                    if not pd.isna(r["比率変化"]) and r["比率変化"] > 0:
                        test_lines.append("この型が本当に有利か確かめるには、同じseedで通常割合を変える、捕食OFF、密度依存OFF、争奪コスト変更を行い、同じ方向に伸びるかを見ます。")
                    elif not pd.isna(r["比率変化"]) and r["比率変化"] < 0:
                        test_lines.append("この型が何に削られたか確かめるには、資源再生、移動コスト、捕食、近親回避、密度依存を一つずつ外し、どの条件で低下が止まるかを見ます。")
                    else:
                        test_lines.append("この型は大きく動いていないため、強い淘汰を受けていないか、複数の圧が打ち消し合っている可能性があります。比較実験で環境圧を一つずつ外すと分かります。")
                    lines.append("**次に確かめること。** " + " ".join(test_lines))
                    st.markdown(join_sentences(lines))

        # ---------- facts / hypotheses / unknowns ----------
        st.markdown("#### 7. 事実・因果候補・未確定を分ける")
        fact_lines = []
        if top_ratio:
            fact_lines.append(f"観察事実：{top_ratio['型']} の比率変化が最も大きい（{arrow_delta(top_ratio['比率変化'])}）。")
        if bottom_ratio:
            fact_lines.append(f"観察事実：{bottom_ratio['型']} の比率低下が最も大きい（{arrow_delta(bottom_ratio['比率変化'])}）。")
        fact_lines.append(f"観察事実：終盤Wは {fmt(pop_w_late)}、出生/死亡比は {fmt(bd_ratio_late,2)}、終盤Giniは {fmt(gini_late)}。")
        strong_lines = []
        if top_ratio and not pd.isna(top_ratio.get("終盤W", np.nan)) and top_ratio["終盤W"] >= 1 and top_ratio.get("親出生", 0) > top_ratio.get("死亡", 0):
            strong_lines.append(f"強めの因果候補：{top_ratio['型']} は比率・終盤W・親出生が同じ方向にそろっているため、繁殖出口を通じて増えた可能性がある。")
        if top_resource and not pd.isna(top_resource.get("資源収支", np.nan)) and top_resource["資源収支"] > avg.get("資源収支", 0):
            strong_lines.append(f"強めの因果候補：{top_resource['型']} は資源収支が高く、資源アクセス圧の利益を受けている可能性がある。")
        if low_hunger and not pd.isna(low_hunger.get("空腹率", np.nan)) and low_hunger["空腹率"] < avg.get("空腹率", 0):
            strong_lines.append(f"強めの因果候補：{low_hunger['型']} は空腹率が低く、死亡回避・繁殖余剰の点で有利な可能性がある。")
        if not strong_lines:
            strong_lines.append("強めの因果候補：まだ複数指標が同じ方向にそろっていません。現時点では、初期配置やチーム偏りの影響も大きく残ります。")
        weak_lines = []
        weak_lines.append("弱い候補：チーム差は、チーム色そのものではなく、チーム内の資源・Gini・タカ比率・行動型構成で説明できる可能性があります。")
        weak_lines.append("弱い候補：哲学型差は、哲学型単独ではなく、タカ/ハト、捕食傾向、通常個体割合、有利な局所環境との結合効果かもしれません。")
        unknown_lines = []
        unknown_lines.append("未確定：単独ランでは、遺伝子の効果と偶然の配置差を分離できません。v20比較実験で同じseedの条件差を見ます。")
        unknown_lines.append("未確定：チーム別の出生・死亡、型別の死因、交尾失敗理由、移動コストの内訳がまだ弱いです。これらを追加すると、因果推定はさらに強くなります。")
        st.markdown("**観察事実**")
        for x in fact_lines:
            st.markdown(f"- {x}")
        st.markdown("**強めの因果候補**")
        for x in strong_lines:
            st.markdown(f"- {x}")
        st.markdown("**弱い因果候補・別解釈**")
        for x in weak_lines:
            st.markdown(f"- {x}")
        st.markdown("**まだ断定しないこと**")
        for x in unknown_lines:
            st.markdown(f"- {x}")

        explain_box(
            "生物として面白いところ",
            "このモデルで面白いのは、強い遺伝子が単独で勝つのではなく、環境と結びついたときだけ有利さが現れる点です。資源が多い世界では採取寄りが強そうに見えますが、資源が局所に偏ると移動コストが問題になります。密度が高い世界では交尾相手に会いやすい一方、過密で子が置けなくなります。争奪が得ならタカ性は利得になり、争奪が損なら同じタカ性が負担になります。捕食も成功率が高ければ資源獲得ですが、低ければ危険接触です。つまり遺伝子の価値は固定ではなく、環境が変わると意味が変わります。ここに、自然淘汰の一番おもしろい部分があります。"
        )




    def show_public_causal_report_v25():
        """外部向けの長文因果レポート。結果の列挙ではなく、環境と遺伝子の因果経路を文章で読ませる。"""
        if len(df) < 3:
            st.info("分析レポートには、少なくとも3世代以上の履歴が必要です。まず数十世代ほど進めてください。")
            return

        # ---------- helpers ----------
        def has(col):
            return col in df.columns
        def srs(col, frame=None):
            frame = df if frame is None else frame
            if col not in frame.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(frame[col], errors="coerce")
        def clean(col, frame=None):
            return srs(col, frame).dropna()
        def firstv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[0]) if len(v) else default
        def lastv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[-1]) if len(v) else default
        def meanv(col, default=np.nan, frame=None):
            v = clean(col, frame)
            return float(v.mean()) if len(v) else default
        def sumv(col, default=0.0, frame=None):
            v = clean(col, frame)
            return float(v.sum()) if len(v) else default
        def minv(col, default=np.nan):
            v = clean(col)
            return float(v.min()) if len(v) else default
        def maxv(col, default=np.nan):
            v = clean(col)
            return float(v.max()) if len(v) else default
        def corr(a, b):
            if a not in df.columns or b not in df.columns:
                return np.nan
            d = pd.concat([srs(a), srs(b)], axis=1).dropna()
            if len(d) < 6:
                return np.nan
            try:
                return float(d.iloc[:, 0].corr(d.iloc[:, 1]))
            except Exception:
                return np.nan
        def fmt(x, digits=3, signed=False):
            try:
                if x is None or pd.isna(x):
                    return "—"
                sign = "+" if signed and float(x) >= 0 else ""
                return f"{sign}{float(x):.{digits}f}"
            except Exception:
                return "—"
        def pct(x, digits=1):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return f"{float(x)*100:.{digits}f}%"
            except Exception:
                return "—"
        def cnt(x):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return str(int(round(float(x))))
            except Exception:
                return "—"
        def delta(x, digits=3):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return f"{float(x):+.{digits}f}"
            except Exception:
                return "—"
        def join(lines):
            return "\n\n".join([str(x).strip() for x in lines if str(x).strip()])
        def relation(value, ref, margin):
            if pd.isna(value) or pd.isna(ref):
                return "読めない"
            if value > ref + margin:
                return "高い"
            if value < ref - margin:
                return "低い"
            return "近い"
        def strength_label(*conditions):
            true_count = sum(1 for c in conditions if bool(c))
            if true_count >= 3:
                return "かなり強い候補"
            if true_count == 2:
                return "中程度の候補"
            if true_count == 1:
                return "弱い候補"
            return "未確定"

        def section(title, body):
            st.markdown(f"#### {title}")
            st.markdown(body)

        # ---------- windows ----------
        n = len(df)
        third = max(3, n // 3)
        early = df.head(third)
        middle = df.iloc[third:2*third] if n >= third * 3 else df.iloc[max(0, n//3):max(1, 2*n//3)]
        late = df.tail(third)
        recent = df.tail(min(120, n))

        # ---------- global state ----------
        pop0, pop1 = firstv("個体数（体）", 0), lastv("個体数（体）", 0)
        red0, red1 = firstv("個体数（赤体）", 0), lastv("個体数（赤体）", 0)
        blue0, blue1 = firstv("個体数（青体）", 0), lastv("個体数（青体）", 0)
        pop_w_all = meanv("個体群全体W（増殖率）", np.nan)
        pop_w_early = meanv("個体群全体W（増殖率）", np.nan, early)
        pop_w_mid = meanv("個体群全体W（増殖率）", np.nan, middle)
        pop_w_late = meanv("個体群全体W（増殖率）", np.nan, late)
        births_total = sumv("出生数（体/世代）", 0.0)
        deaths_total = sumv("死亡数（体/世代）", 0.0)
        births_early, births_late = meanv("出生数（体/世代）", np.nan, early), meanv("出生数（体/世代）", np.nan, late)
        deaths_early, deaths_late = meanv("死亡数（体/世代）", np.nan, early), meanv("死亡数（体/世代）", np.nan, late)
        bd_late = births_late / max(deaths_late, 1e-9) if not pd.isna(births_late) and not pd.isna(deaths_late) else np.nan

        res0, res1 = firstv("資源総量（単位）", np.nan), lastv("資源総量（単位）", np.nan)
        res_min, res_max = minv("資源総量（単位）", np.nan), maxv("資源総量（単位）", np.nan)
        res_early, res_late = meanv("資源総量（単位）", np.nan, early), meanv("資源総量（単位）", np.nan, late)
        bag_early, bag_late = meanv("平均所持資源（単位/体）", np.nan, early), meanv("平均所持資源（単位/体）", np.nan, late)
        gini_early, gini_late = meanv("資源格差Gini（0-1）", np.nan, early), meanv("資源格差Gini（0-1）", np.nan, late)
        resource_cells_late = meanv("資源マス割合（0-1）", np.nan, late)
        density_late = meanv("平均局所密度（体/近傍）", np.nan, late)
        mate_rate_late = meanv("交尾成立率（0-1）", np.nan, late)
        kin_block_late = meanv("近親交配回避（回/世代）", 0.0, late)
        density_block_late = meanv("過密で抑制された出生候補（回/世代）", 0.0, late)
        pred_attempt_late = meanv("捕食試行（回/世代）", 0.0, late)
        pred_success_late = meanv("捕食成功率（0-1）", np.nan, late)
        contest_gain_late = meanv("争奪で得たV合計（単位/世代）", 0.0, late)
        contest_cost_late = meanv("争奪で支払ったC合計（単位/世代）", 0.0, late)
        contest_net_late = contest_gain_late - contest_cost_late
        normal0, normal1 = firstv("通常個体数（体）", np.nan), lastv("通常個体数（体）", np.nan)
        philo0, philo1 = firstv("哲学個体数（体）", np.nan), lastv("哲学個体数（体）", np.nan)
        div0, div1 = firstv("哲学遺伝子多様度（Simpson）", np.nan), lastv("哲学遺伝子多様度（Simpson）", np.nan)

        # correlation is only support; not proof.
        c_birth_bag = corr("出生数（体/世代）", "平均所持資源（単位/体）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_birth_density = corr("出生数（体/世代）", "平均局所密度（体/近傍）")
        c_birth_mate = corr("出生数（体/世代）", "交尾成立率（0-1）")
        c_death_bag = corr("死亡数（体/世代）", "平均所持資源（単位/体）")

        # ---------- gene rows ----------
        labels = [PHILO_LABELS[k] for k in sorted(PHILO_LABELS.keys())]
        total_parent_births = 0.0
        for lab in labels:
            total_parent_births += sumv(f"{lab} 親参加:実出生（回/世代）", 0.0)
        rows = []
        for lab in labels:
            n0 = firstv(f"{lab} 数（体）", np.nan)
            n1 = lastv(f"{lab} 数（体）", np.nan)
            r0 = firstv(f"{lab} 比率（0-1）", np.nan)
            r1 = lastv(f"{lab} 比率（0-1）", np.nan)
            rd = r1 - r0 if not pd.isna(r0) and not pd.isna(r1) else np.nan
            row = {
                "型": lab,
                "初期数": n0, "最新数": n1, "数変化": (n1 - n0 if not pd.isna(n0) and not pd.isna(n1) else np.nan),
                "初期比率": r0, "最新比率": r1, "比率変化": rd,
                "平均W": meanv(f"{lab} W", np.nan),
                "前期W": meanv(f"{lab} W", np.nan, early),
                "終盤W": meanv(f"{lab} W", np.nan, late),
                "資源収支": meanv(f"{lab} 資源収支ネット（単位/世代）", np.nan, late),
                "採取獲得": meanv(f"{lab} 採取獲得（単位/世代）", np.nan, late),
                "移動支払": meanv(f"{lab} 移動支払（単位/世代）", np.nan, late),
                "維持支払": meanv(f"{lab} 維持支払（単位/世代）", np.nan, late),
                "空腹率": meanv(f"{lab} 空腹個体比率（0-1）", np.nan, late),
                "平均所持資源": meanv(f"{lab} 平均所持資源（単位/体）", np.nan, late),
                "足元資源": meanv(f"{lab} 平均足元資源（単位/マス）", np.nan, late),
                "局所密度": meanv(f"{lab} 平均局所密度（体/近傍）", np.nan, late),
                "親出生": sumv(f"{lab} 親参加:実出生（回/世代）", 0.0),
                "子として出生": sumv(f"{lab} 実出生（体/世代）", 0.0),
                "死亡": sumv(f"{lab} 死亡（体/世代）", 0.0),
                "交尾試行": meanv(f"{lab} 交尾試行参加（回/世代）", np.nan, late),
                "交尾成功": meanv(f"{lab} 交尾成功参加（回/世代）", np.nan, late),
                "捕食試行": meanv(f"{lab} 捕食試行（回/世代）", np.nan, late),
                "捕食成功": meanv(f"{lab} 捕食成功（回/世代）", np.nan, late),
                "捕食失敗": meanv(f"{lab} 捕食失敗（回/世代）", np.nan, late),
                "タカ比率": meanv(f"{lab} タカ比率（0-1）", np.nan, late),
                "捕食傾向比率": meanv(f"{lab} 捕食傾向比率（0-1）", np.nan, late),
                "赤比率": meanv(f"{lab} 赤比率（0-1）", np.nan, late),
                "青比率": meanv(f"{lab} 青比率（0-1）", np.nan, late),
                "採取率": meanv(f"{lab} 行動率:採取（0-1）", np.nan, late),
                "移動率": meanv(f"{lab} 行動率:移動（0-1）", np.nan, late),
                "交尾率": meanv(f"{lab} 行動率:交尾（0-1）", np.nan, late),
                "回避率": meanv(f"{lab} 行動率:回避（0-1）", np.nan, late),
                "戦闘率": meanv(f"{lab} 行動率:戦闘（0-1）", np.nan, late),
                "捕食率": meanv(f"{lab} 行動率:捕食（0-1）", np.nan, late),
            }
            row["親出生シェア"] = row["親出生"] / max(total_parent_births, 1e-9)
            rows.append(row)
        gene_df = pd.DataFrame(rows)
        if len(gene_df):
            gene_df["説明用スコア"] = 0.0
            for idx, r in gene_df.iterrows():
                score = 0.0
                if not pd.isna(r.get("比率変化", np.nan)): score += r["比率変化"] * 12
                if not pd.isna(r.get("終盤W", np.nan)): score += (r["終盤W"] - 1.0) * 4
                if not pd.isna(r.get("資源収支", np.nan)): score += np.tanh(r["資源収支"] / 40.0)
                if not pd.isna(r.get("空腹率", np.nan)): score -= r["空腹率"]
                score += r.get("親出生シェア", 0) * 1.5
                gene_df.at[idx, "説明用スコア"] = score
            gene_df = gene_df.sort_values("説明用スコア", ascending=False)
        avgs = {}
        for key in ["資源収支","空腹率","平均所持資源","足元資源","局所密度","親出生","死亡","タカ比率","捕食傾向比率","交尾成功","交尾率","回避率","戦闘率","捕食率","移動率","採取率"]:
            avgs[key] = float(gene_df[key].mean()) if len(gene_df) and key in gene_df.columns else np.nan
        def top_by(key, reverse=True):
            if not len(gene_df) or key not in gene_df.columns:
                return None
            d = gene_df.dropna(subset=[key])
            if len(d) == 0:
                return None
            return d.sort_values(key, ascending=not reverse).iloc[0].to_dict()
        top_ratio = top_by("比率変化", True)
        bottom_ratio = top_by("比率変化", False)
        top_resource = top_by("資源収支", True)
        bottom_resource = top_by("資源収支", False)
        low_hunger = top_by("空腹率", False)
        high_hunger = top_by("空腹率", True)
        top_parent = top_by("親出生", True)
        high_death = top_by("死亡", True)
        high_hawk = top_by("タカ比率", True)
        high_pred = top_by("捕食傾向比率", True)

        st.markdown("### 総合分析レポート")
        section(
            "このレポートの立場",
            """
この欄は、数字をもう一度言い換えるための欄ではありません。ここで見たいのは、**遺伝子コピー数の増減が、どの環境条件を通って生まれたのか**です。

ネオライフゲームでは、遺伝子は個体の中にありますが、個体だけを見ても淘汰は読めません。個体は、資源がある場所にいるか、近くに相手がいるか、周囲が混みすぎていないか、戦闘や捕食が割に合う状況か、同じチームにどんな遺伝子が集まっているか、という環境に置かれています。つまり、ある遺伝子が増えるとき、それは単にその遺伝子が「強い」からではなく、**その遺伝子が作る行動傾向と、その時点の環境が噛み合った**から増えた可能性があります。

したがって、このレポートでは、結果をすぐに優劣として断定しません。まず「何が起きたか」を確認し、次に「それを起こしうる経路」を、資源、格差、密度、交尾、争奪、捕食、チーム偏り、親子フローに分けて追います。最後に、どこまでが観察事実で、どこからがまだ仮説かを分けます。
"""
        )

        # ---------- world story ----------
        pop_direction = "維持に近い"
        if not pd.isna(pop_w_late):
            if pop_w_late > 1.03:
                pop_direction = "増殖圧がある"
            elif pop_w_late < 0.97:
                pop_direction = "縮小圧がある"
        world = []
        world.append(f"このランでは、個体数は {cnt(pop0)} 体から {cnt(pop1)} 体へ変化しています。全期間平均Wは {fmt(pop_w_all)}、終盤Wは {fmt(pop_w_late)} なので、終盤の集団状態は **{pop_direction}** と読めます。ここで重要なのは、Wが結果の指標であって、原因ではないことです。Wが1を上回る/下回る理由は、出生が増えたのか、死亡が減ったのか、資源が持てたのか、交尾出口が開いたのか、危険行動が得になったのかを分けないと分かりません。")
        if not pd.isna(bd_late):
            if bd_late >= 1.05:
                world.append(f"終盤の出生/死亡比は {fmt(bd_late,2)} で、出生が死亡を上回る方向です。これは『個体が増えている』というより、正確には、資源・相手・空きマス・近親回避を通過できたコピーが、死亡による消失を上回っているという意味です。")
            elif bd_late <= 0.95:
                world.append(f"終盤の出生/死亡比は {fmt(bd_late,2)} で、死亡側が重くなっています。この場合、どの遺伝子が負けたかより先に、死亡を作る環境圧を見ます。資源不足、資源格差、過密、争奪/捕食の失敗、交尾前に死ぬタイミングのずれが候補です。")
            else:
                world.append(f"終盤の出生/死亡比は {fmt(bd_late,2)} で、出生と死亡がかなり拮抗しています。この状態は、遺伝子差を見るには面白いです。集団全体が爆発・崩壊していないので、小さな行動差や環境との相性が、頻度の変化として見えやすくなるからです。")
        if not pd.isna(div0) and not pd.isna(div1):
            if div1 < div0 - 0.04:
                world.append(f"哲学遺伝子多様度は {fmt(div0)} から {fmt(div1)} へ下がっています。これは、いくつかの型が相対的に削られ、別の型へコピーが偏り始めたということです。ただし、多様度低下だけでは自然選択とは限りません。少数個体が偶然よい場所に置かれた遺伝的浮動でも起こります。だから、親出生、死亡、資源収支、チーム偏りを同時に見ます。")
            elif div1 > div0 + 0.04:
                world.append(f"哲学遺伝子多様度は {fmt(div0)} から {fmt(div1)} へ上がっています。これは一つの型が独占するより、複数の行動方針が環境内で併存している状態です。資源場所、密度、危険、繁殖機会が空間的にばらつくと、単一の最適戦略より複数戦略の共存が起こりやすくなります。")
            else:
                world.append(f"哲学遺伝子多様度は {fmt(div0)} から {fmt(div1)} で大きくは動いていません。これは、強い一方向の淘汰がまだ出ていないか、複数の淘汰圧が互いに打ち消し合っている状態として読めます。")
        section("まず、この世界では何が起きているか", join(world))

        # ---------- environment story ----------
        env = []
        if not pd.isna(res_early) and not pd.isna(res_late):
            if res_late > res_early * 1.15:
                env.append(f"資源総量は前期平均 {fmt(res_early,1)} から終盤平均 {fmt(res_late,1)} へ増えています。ここで短絡して『資源が増えたから有利』とは言えません。盤面に資源があることと、個体がその資源を所持資源に変換できることは別です。もし平均所持資源や出生が伸びていないなら、資源は存在しているが、個体の認識・移動・採取・局所配置がそこへ接続できていない可能性があります。")
            elif res_late < res_early * 0.85:
                env.append(f"資源総量は前期平均 {fmt(res_early,1)} から終盤平均 {fmt(res_late,1)} へ減っています。この場合、資源は世界全体の制限要因として働いている可能性があります。採取に強い型が増える一方、移動コストが高い型、危険回避で資源地帯に入れない型、交尾へ資源を回せない型が削られやすくなります。")
            else:
                env.append(f"資源総量は前期平均 {fmt(res_early,1)}、終盤平均 {fmt(res_late,1)} で大きくは崩れていません。したがって、資源の総量よりも、資源がどこにあり、誰が取れているか、つまり局所性と格差が重要になります。")
        if not pd.isna(gini_late):
            if gini_late >= 0.45:
                env.append(f"終盤の資源格差Giniは {fmt(gini_late)} と高めです。これは、平均値だけでは見えない淘汰圧です。平均資源が十分でも、資源が一部の個体に偏ると、低資源個体は維持コストを払えず、交尾にも移動にも回せません。つまりGiniが高い世界では、『多く持つ個体がさらに繁殖する』という濃縮が起きやすく、資源を持てない遺伝子型は急に削られます。")
            elif gini_late <= 0.30:
                env.append(f"終盤の資源格差Giniは {fmt(gini_late)} と低めです。この環境では、極端に貧しい個体が生まれにくいため、死亡圧は弱まりやすいです。そのかわり、遺伝子差は資源量より、交尾相手の見つけ方、密度、危険回避、行動コストに現れやすくなります。")
            else:
                env.append(f"終盤の資源格差Giniは {fmt(gini_late)} で中程度です。資源だけで全てが決まる世界ではなく、資源アクセスと繁殖出口、危険行動が同時に効いていると考えられます。")
        if not pd.isna(bag_late):
            env.append(f"終盤の平均所持資源は {fmt(bag_late,2)} です。これは個体が実際に使える内部資源であり、盤面の資源総量より繁殖や生存に近い指標です。出生が所持資源と強く連動しているなら、資源獲得が繁殖の主経路です。逆に連動が弱いなら、相手不足、空きマス不足、近親回避、危険死が出口を塞いでいます。今回の出生と平均所持資源の相関は {fmt(c_birth_bag)} です。")
        if not pd.isna(density_late):
            env.append(f"終盤の平均局所密度は {fmt(density_late,2)} です。密度は二面性があります。近い個体が多いほど交尾相手に会いやすい一方、過密になると子を置く空きマスがなくなり、出生候補が詰まります。出生と局所密度の相関は {fmt(c_birth_density)}、過密抑制は平均 {fmt(density_block_late,2)} 回/世代です。密度が味方なのか敵なのかは、この二つを合わせて判断します。")
        if not pd.isna(mate_rate_late):
            env.append(f"終盤の交尾成立率は {fmt(mate_rate_late)} です。交尾成立率が低い場合、個体が生きていて資源を持っていても、コピーは増えません。ここでは『生き残る能力』と『子を残す能力』が分かれます。生存寄りの型が数を維持していても、親出生が低いなら、長期的には繁殖型に置き換えられる可能性があります。")
        if kin_block_late > 0:
            env.append(f"近親交配回避は終盤平均 {fmt(kin_block_late,2)} 回/世代発生しています。これは単なる制約ではなく、遺伝子フローの向きを変える圧です。近縁個体の近くに集まりやすい型は交尾機会があっても弾かれ、非近縁個体と出会いやすい移動・配置を持つ型が有利になります。")
        if pred_attempt_late > 0:
            env.append(f"捕食試行は終盤平均 {fmt(pred_attempt_late,2)} 回/世代、成功率は {fmt(pred_success_late)} です。捕食は『攻撃的だから強い』ではありません。成功率が高ければ資源獲得経路ですが、低ければ失敗コストと危険接触です。捕食傾向遺伝子は、環境の獲物密度と成功率に依存して、利益にも負担にも変わります。")
        if contest_gain_late or contest_cost_late:
            env.append(f"争奪の終盤ネットは {fmt(contest_net_late,2,True)} です。争奪ネットが正なら、タカ的傾向は資源獲得を増幅しやすい。負なら、同じタカ性が消耗として働きます。したがって、タカ遺伝子の価値は固定ではなく、争奪の収支によって反転します。")
        section("環境はどんな淘汰圧を作っているか", join(env))

        # ---------- summary table ----------
        if len(gene_df):
            display_df = gene_df[["型","初期比率","最新比率","比率変化","終盤W","資源収支","空腹率","親出生","死亡","タカ比率","捕食傾向比率","赤比率","青比率"]].copy()
            for c in ["初期比率","最新比率","空腹率","タカ比率","捕食傾向比率","赤比率","青比率"]:
                display_df[c] = display_df[c].map(lambda x: pct(x) if not pd.isna(x) else "—")
            for c in ["比率変化","終盤W","資源収支"]:
                display_df[c] = display_df[c].map(lambda x: fmt(x,3, c in ["比率変化","資源収支"]) if not pd.isna(x) else "—")
            for c in ["親出生","死亡"]:
                display_df[c] = display_df[c].map(lambda x: cnt(x))
            st.markdown("#### 遺伝子型ごとの観察表")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ---------- gene stories ----------
        st.markdown("#### 遺伝子型ごとの物語")
        theory_by_label = {v: PHILO_THEORY.get(k, "") for k, v in PHILO_LABELS.items()}
        if len(gene_df):
            for _, rr in gene_df.iterrows():
                r = rr.to_dict()
                lab = r["型"]
                with st.expander(f"{lab}：結果ではなく、結果を生んだ経路を読む", expanded=(lab in [(top_ratio or {}).get("型", None), (bottom_ratio or {}).get("型", None)])):
                    parts = []
                    theory = theory_by_label.get(lab, "")
                    if theory:
                        parts.append(f"**この型がモデル内で意味すること。** {theory} ただし、ここで見ているのは思想家本人の優劣ではありません。この型名は、個体が世界を評価するときの重みづけを表すラベルです。その重みづけが、資源・密度・危険・交尾相手という環境にぶつかったとき、どのようにコピー数へ返るかを観察しています。")
                    # observed pattern
                    parts.append(f"**観察された結果。** 初期は {cnt(r['初期数'])} 体、最新は {cnt(r['最新数'])} 体です。比率は {pct(r['初期比率'])} から {pct(r['最新比率'])} へ動き、変化量は {delta(r['比率変化'])} です。終盤Wは {fmt(r['終盤W'])} です。ここまでは結果です。この型が本当に有利だったのか、他の型が減ったため相対的に増えただけなのか、次に経路を分解します。")
                    # resource path
                    res_mechanisms = []
                    res_rel = relation(r.get("資源収支", np.nan), avgs.get("資源収支", np.nan), 2.0)
                    hunger_rel = relation(r.get("空腹率", np.nan), avgs.get("空腹率", np.nan), 0.05)
                    foot_rel = relation(r.get("足元資源", np.nan), avgs.get("足元資源", np.nan), 0.25)
                    if res_rel == "高い":
                        res_mechanisms.append(f"資源収支が平均より高いです（{fmt(r['資源収支'],2,True)}）。これは、この型が環境資源を個体内資源へ変換する経路を比較的うまく通っていることを示します。採取場所へ行けている、無駄な移動が少ない、危険行動で資源を失いにくい、あるいは争奪/捕食が利益になっている可能性があります。")
                    elif res_rel == "低い":
                        res_mechanisms.append(f"資源収支が平均より低いです（{fmt(r['資源収支'],2,True)}）。この型は、世界に資源があってもそれを自分の資源に変換できていない可能性があります。原因としては、資源のない場所に留まる、移動コストが重い、危険回避で資源地帯へ入れない、捕食や戦闘で失敗する、などが考えられます。")
                    else:
                        res_mechanisms.append(f"資源収支は平均に近いです（{fmt(r['資源収支'],2,True)}）。この型の増減を資源経路だけで説明するのは弱く、繁殖出口や死亡回避、同伴遺伝子を見る必要があります。")
                    if foot_rel == "高い":
                        res_mechanisms.append("足元資源が平均より高めなので、この型は資源の近くに位置している可能性があります。ただし、足元に資源があるだけでは十分ではなく、採取行動に移るか、移動や維持で失わないかが次の分岐です。")
                    elif foot_rel == "低い":
                        res_mechanisms.append("足元資源が低めなので、局所資源配置との相性は悪い可能性があります。もしこの型が残っているなら、資源以外の経路、たとえば死亡回避や交尾出口で補っているかもしれません。")
                    if hunger_rel == "高い":
                        res_mechanisms.append(f"空腹率は高めです（{pct(r['空腹率'])}）。空腹は単なる状態ではなく、次の行動選択を狭める圧です。維持コストを払えない、交尾に必要な余剰がない、移動しても回収できない、という連鎖でWを下げます。")
                    elif hunger_rel == "低い":
                        res_mechanisms.append(f"空腹率は低めです（{pct(r['空腹率'])}）。これは死亡回避だけでなく、繁殖に回す余剰を残しやすいという意味でも有利です。")
                    parts.append("**資源・環境との接続。** " + " ".join(res_mechanisms))
                    # reproduction path
                    rep = []
                    if r.get("親出生", 0) > r.get("死亡", 0):
                        rep.append(f"親として残したコピーは {cnt(r['親出生'])}、死亡は {cnt(r['死亡'])} です。親出生が死亡を上回るため、この型は『生き残っただけ』ではなく、繁殖出口を通ってコピーを作った可能性があります。")
                    elif r.get("親出生", 0) < r.get("死亡", 0):
                        rep.append(f"親として残したコピーは {cnt(r['親出生'])}、死亡は {cnt(r['死亡'])} です。死亡の方が重いため、この型が減っているならかなり自然に説明できます。もし比率が増えているなら、それは他型がさらに減った、または少数の成功個体が支えた可能性があります。")
                    else:
                        rep.append(f"親出生と死亡は大きく離れていません。この型は強く増殖しているというより、他の圧と釣り合っている可能性があります。")
                    if not pd.isna(r.get("交尾率", np.nan)) and not pd.isna(r.get("交尾成功", np.nan)):
                        if r["交尾率"] > avgs.get("交尾率", np.nan) + 0.02 and r["交尾成功"] <= avgs.get("交尾成功", np.nan):
                            rep.append("交尾行動は比較的多いのに成功が伸びていません。この場合、意欲ではなく環境側の出口が詰まっています。近親回避、空きマス不足、相手の位置、資源不足が候補です。")
                        elif r["交尾成功"] > avgs.get("交尾成功", np.nan):
                            rep.append("交尾成功が平均より高めなので、この型は相手・資源・空きマスという繁殖条件を比較的よく通過しています。")
                        elif r["交尾率"] < avgs.get("交尾率", np.nan) - 0.02:
                            rep.append("交尾行動率が低めです。この型が残っているなら、増殖力ではなく生存維持で残っている可能性があります。長期的には、死亡が少ないだけではなく親出生が伸びるかが重要です。")
                    parts.append("**繁殖出口。** " + " ".join(rep))
                    # dangerous / gene interactions
                    danger = []
                    if not pd.isna(r.get("タカ比率", np.nan)):
                        if r["タカ比率"] > avgs.get("タカ比率", np.nan) + 0.06:
                            if contest_net_late > 0:
                                danger.append(f"タカ比率が高く、争奪ネットも正です。この場合、この型の結果は哲学型単独ではなく、タカ遺伝子との結合で資源獲得が増幅された可能性があります。")
                            else:
                                danger.append(f"タカ比率が高いのに争奪ネットは正ではありません。ここではタカ性は利益ではなく、消耗を増やす同伴遺伝子になっている可能性があります。")
                        elif r["タカ比率"] < avgs.get("タカ比率", np.nan) - 0.06:
                            if contest_net_late <= 0:
                                danger.append("ハト寄りで、争奪ネットも弱い/負です。戦闘を避けることが資源消耗を減らし、生存維持に効いた可能性があります。")
                            else:
                                danger.append("ハト寄りですが、争奪ネットは正です。この場合、戦闘利得を取り逃がしている可能性もあり、ハト性が常に有利とは言えません。")
                    if not pd.isna(r.get("捕食傾向比率", np.nan)):
                        if r["捕食傾向比率"] > avgs.get("捕食傾向比率", np.nan) + 0.03:
                            if not pd.isna(pred_success_late) and pred_success_late > 0.5:
                                danger.append("捕食傾向が高く、捕食成功率も高めなので、捕食は資源獲得経路として働いた可能性があります。")
                            else:
                                danger.append("捕食傾向が高い一方、捕食成功率は十分高いとは言えません。この場合、捕食傾向は危険接触や失敗コストを増やす圧かもしれません。")
                    if not pd.isna(r.get("回避率", np.nan)) and r["回避率"] > avgs.get("回避率", np.nan) + 0.03:
                        danger.append("回避率が高めです。回避は死亡を減らす可能性がありますが、同時に資源地帯や交尾機会から遠ざかる可能性もあります。つまり回避は生存には効くが、繁殖には必ずしも効かない二面性を持ちます。")
                    if not danger:
                        danger.append("危険行動や同伴遺伝子だけでは大きな説明が出ていません。この型の変化は、資源配置、繁殖出口、チーム偏りの方で説明される可能性があります。")
                    parts.append("**他の遺伝子との作用。** " + " ".join(danger))
                    team = []
                    if not pd.isna(r.get("赤比率", np.nan)) and not pd.isna(r.get("青比率", np.nan)):
                        if r["赤比率"] > r["青比率"] + 0.08:
                            team.append("この型は赤チームに偏っています。したがって、この型の増減には赤側の資源配置、近くにいる相互作用相手、チーム内のタカ/ハト構成が混ざっている可能性があります。型そのものの効果とチーム環境の効果を分けるには、赤青内訳を見ます。")
                        elif r["青比率"] > r["赤比率"] + 0.08:
                            team.append("この型は青チームに偏っています。つまり、型が強いように見えても、実際には青側にあった資源・密度・交尾相手の条件が効いている可能性があります。")
                        else:
                            team.append("赤青への偏りは大きくありません。この型の変化はチーム色より、型自身の行動傾向や局所環境との相性を優先して見ます。")
                    parts.append("**チーム環境との関係。** " + " ".join(team))
                    # final interpretation
                    support = strength_label(
                        (not pd.isna(r.get("比率変化", np.nan)) and r["比率変化"] > 0),
                        (not pd.isna(r.get("終盤W", np.nan)) and r["終盤W"] >= 1.0),
                        (r.get("親出生", 0) > r.get("死亡", 0)),
                        (not pd.isna(r.get("資源収支", np.nan)) and r["資源収支"] > avgs.get("資源収支", 0)),
                        (not pd.isna(r.get("空腹率", np.nan)) and r["空腹率"] < avgs.get("空腹率", 1)),
                    )
                    if not pd.isna(r.get("比率変化", np.nan)) and r["比率変化"] > 0:
                        conclusion = f"この型の優位性は **{support}** です。比率上昇があり、そこにW、親出生、資源収支、空腹率がどれだけ同じ方向でそろうかで解釈の強さを決めます。そろっているなら自然選択の候補として強い。そろっていないなら、他型の減少、チーム偏り、偶然配置の影響をまだ疑います。"
                    elif not pd.isna(r.get("比率変化", np.nan)) and r["比率変化"] < 0:
                        conclusion = f"この型の劣勢化は **{support}** とは逆方向に読まれます。低下の原因は、資源に届かない、空腹が多い、親出生が少ない、死亡が多い、同伴遺伝子が環境と噛み合わない、のどこかにあるはずです。"
                    else:
                        conclusion = "この型は大きくは動いていません。強い淘汰を受けていないのか、複数の圧が打ち消し合っているのかを分ける必要があります。"
                    parts.append("**この型について今言えること。** " + conclusion)
                    st.markdown(join(parts))

        # ---------- team story ----------
        st.markdown("#### 赤チーム・青チームの差を、結果ではなく原因として読む")
        red_res, blue_res = meanv("赤 平均所持資源", np.nan, late), meanv("青 平均所持資源", np.nan, late)
        red_g, blue_g = meanv("赤 Gini", np.nan, late), meanv("青 Gini", np.nan, late)
        red_h, blue_h = meanv("赤 タカ比率（0-1）", np.nan, late), meanv("青 タカ比率（0-1）", np.nan, late)
        team_lines = []
        if red1 > blue1 * 1.08:
            side, other = "赤", "青"
            side_res, other_res, side_g, other_g, side_h, other_h = red_res, blue_res, red_g, blue_g, red_h, blue_h
        elif blue1 > red1 * 1.08:
            side, other = "青", "赤"
            side_res, other_res, side_g, other_g, side_h, other_h = blue_res, red_res, blue_g, red_g, blue_h, red_h
        else:
            side = None
        if side is None:
            team_lines.append(f"赤は {cnt(red1)} 体、青は {cnt(blue1)} 体で、極端な差はありません。この場合、色そのものを原因にするより、チーム内部にどの行動型が多いか、どちらにタカ/ハトが偏っているか、どちらのGiniが高いかを見ます。")
        else:
            team_lines.append(f"最新数では **{side}チーム** が多いです。しかし、これは原因ではなく結果です。原因として考えるべき経路は三つあります。第一に、{side}側が資源を多く持っているか。第二に、{side}側の資源格差が低く、低資源個体を作りにくいか。第三に、{side}側に有利な行動型やタカ/ハト構成が偏っているか、です。")
            if not pd.isna(side_res) and not pd.isna(other_res):
                if side_res > other_res + 0.5:
                    team_lines.append(f"平均所持資源は{side}側が高めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。この場合、{side}の多さは資源アクセスによって支えられた可能性があります。資源を持てる個体は維持コストを払い、移動し、交尾に参加しやすいため、資源差はそのままコピー差へ接続します。")
                elif side_res < other_res - 0.5:
                    team_lines.append(f"平均所持資源はむしろ{side}側が低めです（{fmt(side_res,2)} 対 {fmt(other_res,2)}）。それでも{side}が多いなら、資源量ではなく死亡回避、交尾出口、または内部遺伝子構成が効いている可能性があります。")
                else:
                    team_lines.append(f"平均所持資源は赤青で大差ありません。この場合、チーム差は資源量だけでは説明できません。格差、密度、タカ/ハト、交尾出口を疑います。")
            if not pd.isna(side_g) and not pd.isna(other_g):
                if side_g < other_g - 0.03:
                    team_lines.append(f"{side}側はGiniが低めです。これは強い説明候補です。平均資源が同じでも、Giniが低い側は低資源で脱落する個体を作りにくく、集団としてコピーを維持しやすくなります。")
                elif side_g > other_g + 0.03:
                    team_lines.append(f"{side}側はGiniが高めです。数では多くても、内部では資源が偏っており、少数の豊かな個体に支えられている可能性があります。長期では不安定になるかもしれません。")
            if not pd.isna(side_h) and not pd.isna(other_h):
                if side_h > other_h + 0.05 and contest_net_late > 0:
                    team_lines.append(f"{side}側はタカ比率が高く、争奪ネットも正です。これは、戦闘的な同伴遺伝子が資源獲得を増幅し、{side}側の維持や繁殖を支えた可能性があります。")
                elif side_h > other_h + 0.05 and contest_net_late <= 0:
                    team_lines.append(f"{side}側はタカ比率が高い一方、争奪ネットは正ではありません。ここではタカ性はコストになり得ます。それでも{side}が多いなら、資源配置や繁殖出口など別の要因がタカの負担を上回っているはずです。")
                elif side_h < other_h - 0.05 and contest_net_late <= 0:
                    team_lines.append(f"{side}側はタカ比率が低く、争奪ネットも弱い/負です。戦闘を避ける構成が消耗を減らし、結果としてコピー維持に有利だった可能性があります。")
        section("チーム差の文章解釈", join(team_lines))

        # ---------- what is interesting ----------
        interesting = []
        interesting.append("このモデルで面白いのは、遺伝子の価値が固定されていないことです。タカ遺伝子は、争奪で得られる資源がコストを上回る世界では利益になります。しかし、争奪の失敗や消耗が大きい世界では、同じタカ遺伝子が負担になります。捕食遺伝子も同じです。成功率が高ければ資源獲得経路ですが、失敗が多ければ危険接触を増やすだけです。")
        interesting.append("哲学遺伝子も単独で勝つわけではありません。ヒューム型、ストア型、デカルト型、カント型は、それぞれ世界の見方を変える行動評価関数です。しかし、その評価が有利になるかは環境に依存します。資源が局所的なら、見えている資源へ反応する型が有利になるかもしれません。危険が多いなら、回避や自己保存が有利になるかもしれません。過密で出生が詰まるなら、どれだけ資源を取っても子を置けず、繁殖出口を開ける型が残ります。")
        interesting.append("つまり、自然淘汰は単純な強弱表ではありません。遺伝子、環境、他の遺伝子、相互作用相手、空間配置が組み合わさって、その時だけの有利さを作ります。ある型が増えたとき、それはその型が絶対に優れているという意味ではなく、その型がこの世界の資源配置、密度、危険、繁殖条件と一時的に噛み合ったという意味です。この『噛み合い』を読むことが、ネオライフゲームの核心です。")
        section("生物として面白いところ", join(interesting))

        # ---------- claims and tests ----------
        final_lines = []
        if top_ratio:
            final_lines.append(f"観察事実として、比率変化が最も大きいのは **{top_ratio['型']}** です（{delta(top_ratio['比率変化'])}）。ただし、これはまだ『強い』の証明ではありません。強い主張にするには、親出生、死亡、資源収支、空腹率、同伴遺伝子、チーム偏りが同じ方向を示す必要があります。")
        if bottom_ratio:
            final_lines.append(f"観察事実として、比率低下が最も大きいのは **{bottom_ratio['型']}** です（{delta(bottom_ratio['比率変化'])}）。この型が何に削られたかは、死亡が多いのか、親出生が少ないのか、資源に届かないのか、危険行動が失敗しているのかで分けて考えます。")
        final_lines.append("比較実験で最初に潰すべき別解釈は三つです。第一に、同じseedで哲学遺伝子OFFにして、哲学補正そのものが効いたのかを見ること。第二に、通常個体割合を変えて、哲学個体が少数派でも同じ方向に伸びるかを見ること。第三に、捕食OFF、密度依存OFF、局所資源再生OFFを一つずつ試し、どの環境圧を外したときに遺伝子順位が変わるかを見ることです。順位が変わる場所こそ、その遺伝子が受けていた淘汰圧です。")
        section("この結果から次に何を確かめるべきか", join(final_lines))



    def show_public_causal_report_v26():
        """外部向け：結果ではなく、結果を生んだ経路を数珠つなぎで説明する因果レポート。"""
        if len(df) < 3:
            st.info("分析レポートには、少なくとも3世代以上の履歴が必要です。まず数十世代ほど進めてください。")
            return

        # ---------- helpers ----------
        def srs(col, frame=None):
            frame = df if frame is None else frame
            if col not in frame.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(frame[col], errors="coerce")
        def clean(col, frame=None):
            return srs(col, frame).dropna()
        def meanv(col, default=np.nan, frame=None):
            v = clean(col, frame)
            return float(v.mean()) if len(v) else default
        def sumv(col, default=0.0, frame=None):
            v = clean(col, frame)
            return float(v.sum()) if len(v) else default
        def firstv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[0]) if len(v) else default
        def lastv(col, default=np.nan):
            v = clean(col)
            return float(v.iloc[-1]) if len(v) else default
        def first_any(cols, default=np.nan):
            for c in cols:
                if c in df.columns:
                    return firstv(c, default)
            return default
        def last_any(cols, default=np.nan):
            for c in cols:
                if c in df.columns:
                    return lastv(c, default)
            return default
        def mean_any(cols, default=np.nan, frame=None):
            for c in cols:
                if c in (df if frame is None else frame).columns:
                    return meanv(c, default, frame)
            return default
        def corr(a, b):
            if a not in df.columns or b not in df.columns:
                return np.nan
            d = pd.concat([srs(a), srs(b)], axis=1).dropna()
            if len(d) < 6:
                return np.nan
            try:
                return float(d.iloc[:,0].corr(d.iloc[:,1]))
            except Exception:
                return np.nan
        def fmt(x, digits=3, signed=False):
            try:
                if x is None or pd.isna(x):
                    return "—"
                sign = "+" if signed and float(x) >= 0 else ""
                return f"{sign}{float(x):.{digits}f}"
            except Exception:
                return "—"
        def pct(x, digits=1):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return f"{float(x)*100:.{digits}f}%"
            except Exception:
                return "—"
        def cnt(x):
            try:
                if x is None or pd.isna(x):
                    return "—"
                return str(int(round(float(x))))
            except Exception:
                return "—"
        def join(lines):
            return "\n\n".join([str(x).strip() for x in lines if str(x).strip()])
        def strength(*conds):
            n = sum(1 for c in conds if bool(c))
            return "強い" if n >= 4 else ("中程度" if n >= 2 else ("弱い" if n == 1 else "未確定"))
        def comp_word(value, ref, margin):
            if pd.isna(value) or pd.isna(ref):
                return "不明"
            if value > ref + margin:
                return "高い"
            if value < ref - margin:
                return "低い"
            return "近い"
        def top_by(rows, key, reverse=True):
            vals = [r for r in rows if key in r and not pd.isna(r[key])]
            if not vals:
                return None
            return sorted(vals, key=lambda x: x[key], reverse=reverse)[0]
        def section(title, body):
            st.markdown(f"#### {title}")
            st.markdown(body)

        n = len(df)
        third = max(3, n // 3)
        early = df.head(third)
        middle = df.iloc[third:2*third] if n >= third*3 else df.iloc[max(0,n//3):max(1,2*n//3)]
        late = df.tail(third)

        # ---------- global environment ----------
        pop0, pop1 = firstv("個体数（体）", 0), lastv("個体数（体）", 0)
        red1 = last_any(["赤個体数（体）", "個体数（赤体）"], np.nan)
        blue1 = last_any(["青個体数（体）", "個体数（青体）"], np.nan)
        w_early = meanv("個体群全体W（増殖率）", np.nan, early)
        w_mid = meanv("個体群全体W（増殖率）", np.nan, middle)
        w_late = meanv("個体群全体W（増殖率）", np.nan, late)
        births_late = meanv("出生数（体/世代）", np.nan, late)
        deaths_late = meanv("死亡数（体/世代）", np.nan, late)
        births_total = sumv("出生数（体/世代）", 0.0)
        deaths_total = sumv("死亡数（体/世代）", 0.0)
        res_early = meanv("資源総量（単位）", np.nan, early)
        res_late = meanv("資源総量（単位）", np.nan, late)
        bag_early = meanv("平均所持資源（単位/体）", np.nan, early)
        bag_late = meanv("平均所持資源（単位/体）", np.nan, late)
        gini_late = meanv("資源格差Gini（0-1）", np.nan, late)
        gini_early = meanv("資源格差Gini（0-1）", np.nan, early)
        density_late = meanv("平均局所密度（体/近傍）", np.nan, late)
        mate_late = meanv("交尾成立率（0-1）", np.nan, late)
        kin_block_late = meanv("近親交配回避（回/世代）", 0.0, late)
        density_block_late = meanv("過密で抑制された出生候補（回/世代）", 0.0, late)
        pred_attempt_late = meanv("捕食試行（回/世代）", 0.0, late)
        pred_success_late = meanv("捕食成功率（0-1）", np.nan, late)
        contest_gain_late = meanv("争奪で得たV合計（単位/世代）", 0.0, late)
        contest_cost_late = meanv("争奪で支払ったC合計（単位/世代）", 0.0, late)
        contest_net_late = contest_gain_late - contest_cost_late
        c_birth_bag = corr("出生数（体/世代）", "平均所持資源（単位/体）")
        c_death_gini = corr("死亡数（体/世代）", "資源格差Gini（0-1）")
        c_pop_res = corr("個体数（体）", "資源総量（単位）")
        c_birth_density = corr("出生数（体/世代）", "平均局所密度（体/近傍）")
        c_birth_mate = corr("出生数（体/世代）", "交尾成立率（0-1）")

        # ---------- gene table ----------
        labels = [PHILO_LABELS[k] for k in sorted(PHILO_LABELS.keys()) if isinstance(PHILO_LABELS.get(k), str)]
        total_parent = sum(sumv(f"{lab} 親参加:実出生（回/世代）", 0.0) for lab in labels)
        rows = []
        for lab in labels:
            n0 = firstv(f"{lab} 数（体）", np.nan)
            n1 = lastv(f"{lab} 数（体）", np.nan)
            r0 = firstv(f"{lab} 比率（0-1）", np.nan)
            r1 = lastv(f"{lab} 比率（0-1）", np.nan)
            rd = r1-r0 if not pd.isna(r0) and not pd.isna(r1) else np.nan
            row = {
                "型": lab, "初期数": n0, "最新数": n1, "数変化": (n1-n0 if not pd.isna(n0) and not pd.isna(n1) else np.nan),
                "初期比率": r0, "最新比率": r1, "比率変化": rd,
                "平均W": meanv(f"{lab} W", np.nan), "前期W": meanv(f"{lab} W", np.nan, early), "終盤W": meanv(f"{lab} W", np.nan, late),
                "資源収支": meanv(f"{lab} 資源収支ネット（単位/世代）", np.nan, late),
                "採取獲得": meanv(f"{lab} 採取獲得（単位/世代）", np.nan, late),
                "移動支払": meanv(f"{lab} 移動支払（単位/世代）", np.nan, late),
                "維持支払": meanv(f"{lab} 維持支払（単位/世代）", np.nan, late),
                "空腹率": meanv(f"{lab} 空腹個体比率（0-1）", np.nan, late),
                "平均所持資源": meanv(f"{lab} 平均所持資源（単位/体）", np.nan, late),
                "足元資源": meanv(f"{lab} 平均足元資源（単位/マス）", np.nan, late),
                "局所密度": meanv(f"{lab} 平均局所密度（体/近傍）", np.nan, late),
                "出生": sumv(f"{lab} 実出生（体/世代）", 0.0),
                "親出生": sumv(f"{lab} 親参加:実出生（回/世代）", 0.0),
                "死亡": sumv(f"{lab} 死亡（体/世代）", 0.0),
                "交尾試行": meanv(f"{lab} 交尾試行参加（回/世代）", np.nan, late),
                "交尾成功": meanv(f"{lab} 交尾成功参加（回/世代）", np.nan, late),
                "捕食試行": meanv(f"{lab} 捕食試行（回/世代）", np.nan, late),
                "捕食成功": meanv(f"{lab} 捕食成功（回/世代）", np.nan, late),
                "捕食失敗": meanv(f"{lab} 捕食失敗（回/世代）", np.nan, late),
                "タカ比率": meanv(f"{lab} タカ比率（0-1）", np.nan, late),
                "捕食傾向比率": meanv(f"{lab} 捕食傾向比率（0-1）", np.nan, late),
                "赤比率": meanv(f"{lab} 赤比率（0-1）", np.nan, late),
                "青比率": meanv(f"{lab} 青比率（0-1）", np.nan, late),
                "採取率": meanv(f"{lab} 行動率:採取（0-1）", np.nan, late),
                "移動率": meanv(f"{lab} 行動率:移動（0-1）", np.nan, late),
                "交尾率": meanv(f"{lab} 行動率:交尾（0-1）", np.nan, late),
                "回避率": meanv(f"{lab} 行動率:回避（0-1）", np.nan, late),
                "戦闘率": meanv(f"{lab} 行動率:戦闘（0-1）", np.nan, late),
                "捕食率": meanv(f"{lab} 行動率:捕食（0-1）", np.nan, late),
            }
            row["親出生シェア"] = row["親出生"] / max(total_parent, 1e-9)
            score = 0.0
            if not pd.isna(row["比率変化"]): score += row["比率変化"] * 12
            if not pd.isna(row["終盤W"]): score += (row["終盤W"] - 1.0) * 5
            if not pd.isna(row["資源収支"]): score += np.tanh(row["資源収支"] / 45.0)
            if not pd.isna(row["空腹率"]): score -= row["空腹率"]
            score += row["親出生シェア"] * 1.5
            row["表示順スコア"] = score
            rows.append(row)
        gene_df = pd.DataFrame(rows)
        if len(gene_df):
            gene_df = gene_df.sort_values("表示順スコア", ascending=False)
        av = {}
        for key in ["資源収支","採取獲得","移動支払","維持支払","空腹率","平均所持資源","足元資源","局所密度","親出生シェア","交尾成功","タカ比率","捕食傾向比率","採取率","移動率","交尾率","回避率","戦闘率","捕食率"]:
            vals = [r[key] for r in rows if key in r and not pd.isna(r[key])]
            av[key] = float(np.mean(vals)) if vals else np.nan
        top_ratio = top_by(rows, "比率変化", True)
        bottom_ratio = top_by(rows, "比率変化", False)
        top_parent = top_by(rows, "親出生シェア", True)
        top_res = top_by(rows, "資源収支", True)
        high_hunger = top_by(rows, "空腹率", True)

        st.markdown("### 総合分析レポート")
        explain_box(
            "このレポートが目指す説明",
            "ここでは『青が多い』『カント型が増えた』のような結果だけを説明とは呼びません。結果の前に、必ず原因の鎖があります。たとえば、ある型が増えたなら、まず親として子を残したのか、それとも死亡が少なかっただけなのかを分けます。親として子を残したなら、なぜ交尾できたのかを見ます。交尾できたなら、資源を持っていたのか、相手に会える密度だったのか、近親回避や過密で止まらなかったのかを見ます。資源を持っていたなら、なぜ持てたのか、足元資源が多かったのか、採取行動が多かったのか、移動コストが低かったのか、争奪や捕食が資源を増やしたのかを見ます。このように、**結果 → 直接原因 → その原因の原因 → 環境条件 → 遺伝子の作用** まで戻って読みます。"
        )

        # --- 世界の圧力を先に特定 ---
        env_lines = []
        if not pd.isna(res_late) and not pd.isna(bag_late):
            if res_late > res_early and bag_late <= bag_early:
                env_lines.append(f"盤面資源は前期平均 {fmt(res_early,1)} から終盤平均 {fmt(res_late,1)} へ増えていますが、平均所持資源は {fmt(bag_early,2)} から {fmt(bag_late,2)} へ伸びていません。これは『資源があるのに個体の体内資源へ変換されていない』状態です。この場合、結果を作っているのは資源量そのものではなく、資源へ到達する行動、足元資源の局所性、移動コスト、採取判断です。")
            elif res_late < res_early and bag_late < bag_early:
                env_lines.append(f"盤面資源も平均所持資源も下がっています。これは単純な資源不足圧です。資源が少ない環境では、採取効率が高い型、移動コストが低い型、争奪や捕食で不足分を補える型が残りやすくなります。")
            elif bag_late > bag_early:
                env_lines.append(f"平均所持資源が前期 {fmt(bag_early,2)} から終盤 {fmt(bag_late,2)} へ上がっています。この場合、少なくとも一部の個体は環境資源を体内資源へ変換できています。次に見るべきなのは、その資源が全型に広く渡っているのか、特定型だけが取っているのかです。Giniと型別資源収支を見ます。")
        if not pd.isna(gini_late):
            if gini_late > 0.42:
                env_lines.append(f"終盤Giniは {fmt(gini_late)} と高めです。ここでは平均資源ではなく資源格差が淘汰圧になります。つまり、豊かな個体は繁殖できる一方、低資源個体は維持コストで脱落しやすい。したがって、低空腹率の型や、資源収支が安定した型が相対的に残ります。")
            elif gini_late < 0.30:
                env_lines.append(f"終盤Giniは {fmt(gini_late)} と低めです。資源が比較的均されているため、資源格差よりも交尾出口、密度、捕食・争奪の成否が差を作りやすい環境です。")
        if not pd.isna(mate_late):
            if mate_late < 0.18 or density_block_late > births_late or kin_block_late > births_late:
                env_lines.append(f"交尾成立率は {fmt(mate_late)}、近親回避は平均 {fmt(kin_block_late,2)}、過密抑制は平均 {fmt(density_block_late,2)} です。出生数そのものより、交尾条件や空きマス条件が詰まりやすいなら、資源を持つだけではコピーは増えません。交尾行動が多い型ではなく、交尾成功と親出生まで到達する型が有利になります。")
            else:
                env_lines.append(f"交尾成立率は {fmt(mate_late)} で、繁殖出口は完全には詰まっていません。この場合、親出生が多い型は資源や相手探索を実際のコピーへ変換できています。")
        if pred_attempt_late > 0:
            if not pd.isna(pred_success_late) and pred_success_late >= 0.55:
                env_lines.append(f"捕食試行があり、捕食成功率は {fmt(pred_success_late)} です。成功率が高い捕食は、単なる危険行動ではなく追加の資源獲得経路になります。捕食傾向が高い型が伸びるなら、捕食遺伝子が哲学型の結果を媒介している可能性があります。")
            else:
                env_lines.append(f"捕食試行はありますが、捕食成功率は {fmt(pred_success_late)} です。成功率が高くないなら、捕食傾向は資源獲得ではなく失敗コストや危険接触を増やす可能性があります。捕食型が減るなら、捕食が罰として働いた可能性を見ます。")
        if contest_gain_late or contest_cost_late:
            if contest_net_late > 0:
                env_lines.append(f"争奪の終盤ネットは {fmt(contest_net_late,2, signed=True)} です。争奪で得た利得が支払ったコストを上回っているため、タカ性は一部の型にとって資源獲得を増幅する媒介遺伝子になり得ます。")
            else:
                env_lines.append(f"争奪の終盤ネットは {fmt(contest_net_late,2, signed=True)} です。争奪が純利益になっていないなら、タカ性はむしろ消耗の経路です。この環境ではハト性や回避性が死亡圧を下げる可能性があります。")
        section("1. まず世界がどんな淘汰圧を作っているか", join(env_lines) if env_lines else "まだ世界全体の淘汰圧を十分に読める統計がありません。世代数を増やすか、統計履歴をリセットして再実行してください。")

        # --- causal chain table ---
        table_cols = ["型","初期数","最新数","比率変化","終盤W","資源収支","空腹率","親出生シェア","タカ比率","捕食傾向比率","赤比率","青比率"]
        show_cols = [c for c in table_cols if c in gene_df.columns]
        st.markdown("#### 2. 遺伝子型別の原因候補表")
        st.caption("この表は結論ではなく、下の文章で使う証拠です。比率変化は結果、資源収支・空腹率・親出生・同伴遺伝子は原因候補です。")
        st.dataframe(gene_df[show_cols], use_container_width=True, hide_index=True)

        def gene_chain(r):
            lab = str(r.get("型", "不明"))
            lines = []
            rd = r.get("比率変化", np.nan)
            wv = r.get("終盤W", np.nan)
            lines.append(f"**出発点。** {lab} は初期 {cnt(r.get('初期数'))} 体から最新 {cnt(r.get('最新数'))} 体へ変化し、比率変化は {fmt(rd,3, signed=True)}、終盤Wは {fmt(wv)} です。ここまでは結果です。原因を読むには、この変化が『親として子を残したから』なのか、『死亡を避けたから』なのか、『他型が減ったから相対的に上がっただけ』なのかを分けます。")
            # direct reproduction vs survival
            parent_share = r.get("親出生シェア", np.nan)
            hunger = r.get("空腹率", np.nan)
            net = r.get("資源収支", np.nan)
            if not pd.isna(parent_share) and parent_share > av.get("親出生シェア", np.nan) + 0.03:
                lines.append(f"**直接原因候補1：繁殖出口を通過している。** {lab} の親出生シェアは {pct(parent_share)} で平均より高めです。つまり、この型は生き残っているだけでなく、親として子世代へコピーを送っています。では、なぜ親になれたのか。親になるには、所持資源、配偶相手、近親回避、空きマス、密度条件を同時に通る必要があります。ここで交尾成功や資源収支が高いなら、繁殖出口を開ける前段階も説明できます。")
            elif not pd.isna(parent_share) and parent_share < av.get("親出生シェア", np.nan) - 0.03:
                lines.append(f"**直接原因候補1：繁殖出口が弱い。** {lab} の親出生シェアは {pct(parent_share)} で平均より低めです。もしこの型の比率が維持されているなら、親として増やしたというより、死亡が少ない、または他型がより強く減ったために残っている可能性があります。")
            else:
                lines.append("**直接原因候補1：繁殖出口だけでは説明しにくい。** 親出生シェアは平均付近です。この型の増減は、繁殖よりも資源収支・死亡回避・同伴遺伝子・チーム偏りから読む必要があります。")
            # resource why
            res_bits = []
            if not pd.isna(net):
                if net > av.get("資源収支", 0) + 2:
                    res_bits.append(f"資源収支が平均より高い（{fmt(net,2,signed=True)}）ため、環境資源を個体内資源へ変換できています")
                elif net < av.get("資源収支", 0) - 2:
                    res_bits.append(f"資源収支が平均より低い（{fmt(net,2,signed=True)}）ため、資源を取り込む前段階で不利です")
            foot = r.get("足元資源", np.nan)
            if not pd.isna(foot):
                if foot > av.get("足元資源", np.nan) + 0.3:
                    res_bits.append("足元資源が多く、そもそも資源の近くにいる/資源地帯へ入りやすい")
                elif foot < av.get("足元資源", np.nan) - 0.3:
                    res_bits.append("足元資源が少なく、資源地帯への接触が弱い")
            gather = r.get("採取率", np.nan)
            move = r.get("移動率", np.nan)
            if not pd.isna(gather) and gather > av.get("採取率", np.nan) + 0.03:
                res_bits.append("採取行動率が高く、見つけた資源を取りに行く傾向がある")
            if not pd.isna(move) and move > av.get("移動率", np.nan) + 0.03:
                res_bits.append("移動率が高く、探索範囲を広げる一方で移動コストも増やし得る")
            if not pd.isna(hunger):
                if hunger < av.get("空腹率", np.nan) - 0.03:
                    res_bits.append(f"空腹率が低い（{pct(hunger)}）ため、維持コストで死亡する圧を避けています")
                elif hunger > av.get("空腹率", np.nan) + 0.03:
                    res_bits.append(f"空腹率が高い（{pct(hunger)}）ため、資源不足が死亡や繁殖不能へつながる可能性があります")
            if res_bits:
                lines.append("**その原因の原因：資源との結びつき。** " + "。".join(res_bits) + "。つまり、資源経路は『盤面資源が多いから有利』ではなく、資源地帯への接触、採取判断、移動コスト、空腹回避まで連続して初めてコピーへ接続します。")
            else:
                lines.append("**その原因の原因：資源との結びつき。** 資源経路の証拠はまだ薄いです。この型の変化は、資源よりも繁殖出口、危険回避、チーム偏り、または偶然配置で説明されるかもしれません。")
            # interaction genes
            inter = []
            hawk = r.get("タカ比率", np.nan)
            if not pd.isna(hawk):
                if hawk > av.get("タカ比率", np.nan) + 0.06:
                    if contest_net_late > 0:
                        inter.append("タカ比率が高く、争奪ネットも正なので、争奪が資源獲得を増幅した可能性があります。ここでは哲学型そのものではなく、哲学型×タカ遺伝子の結合効果を疑います")
                    else:
                        inter.append("タカ比率が高いのに争奪ネットは正ではありません。タカ性は利益ではなく消耗を増やす同伴遺伝子だった可能性があります")
                elif hawk < av.get("タカ比率", np.nan) - 0.06:
                    if contest_net_late <= 0:
                        inter.append("ハト寄りで、争奪ネットも弱い/負です。戦闘を避けることが消耗を減らし、死亡圧を下げた可能性があります")
                    else:
                        inter.append("ハト寄りですが争奪ネットは正です。戦闘利得を取り逃がしている可能性もあるので、ハト性が有利とは断定できません")
            pg = r.get("捕食傾向比率", np.nan)
            if not pd.isna(pg):
                if pg > av.get("捕食傾向比率", np.nan) + 0.03:
                    if not pd.isna(pred_success_late) and pred_success_late >= 0.55:
                        inter.append("捕食傾向が高く、捕食成功率も高いため、捕食が追加資源の経路になった可能性があります")
                    else:
                        inter.append("捕食傾向が高い一方、捕食成功率が十分ではありません。捕食は危険接触や失敗コストを増やした可能性があります")
            if not inter:
                inter.append("タカ/ハトや捕食傾向からは強い媒介が見えません。この型は資源接触、繁殖出口、チーム配置を中心に読むべきです")
            lines.append("**別遺伝子との作用。** " + "。".join(inter) + "。")
            # team mediator
            team_bits = []
            redr, bluer = r.get("赤比率", np.nan), r.get("青比率", np.nan)
            if not pd.isna(redr) and not pd.isna(bluer):
                if redr > bluer + 0.08:
                    team_bits.append("赤チームに偏っています。したがって、この型の結果には赤側の資源配置・密度・相手分布が混ざります")
                elif bluer > redr + 0.08:
                    team_bits.append("青チームに偏っています。したがって、この型の結果には青側の資源配置・密度・相手分布が混ざります")
                else:
                    team_bits.append("赤青への偏りは大きくありません。チーム環境より、型自身の行動傾向や同伴遺伝子を優先して読みます")
            lines.append("**チーム環境。** " + "。".join(team_bits) + "。")
            # final chain conclusion
            stg = strength((not pd.isna(rd) and rd > 0), (not pd.isna(wv) and wv >= 1), (not pd.isna(parent_share) and parent_share > av.get("親出生シェア", np.nan)), (not pd.isna(net) and net > av.get("資源収支", np.nan)), (not pd.isna(hunger) and hunger < av.get("空腹率", np.nan)))
            if not pd.isna(rd) and rd > 0:
                lines.append(f"**数珠つなぎの暫定結論。** {lab} の増加は {stg} な因果候補です。強く言えるのは、比率上昇だけでなく、親出生・資源収支・空腹率・Wが同じ方向にそろう場合です。そろわない場合は、『強いから増えた』ではなく、『他型が減った』『チームに乗った』『同伴遺伝子が効いた』可能性を残します。")
            elif not pd.isna(rd) and rd < 0:
                lines.append(f"**数珠つなぎの暫定結論。** {lab} の低下は、資源に届かない、出生出口に到達しない、空腹や死亡で削られる、同伴遺伝子が環境と噛み合わない、のどれかで説明されます。上の段落で一番多く出た経路が、次に比較実験で潰すべき原因候補です。")
            else:
                lines.append(f"**数珠つなぎの暫定結論。** {lab} は大きく動いていません。これは中立という意味ではなく、資源経路の利益と危険/繁殖の不利益が打ち消し合っている可能性があります。")
            return join(lines)

        st.markdown("#### 3. 遺伝子型ごとの数珠つなぎ説明")
        for _, rr in gene_df.iterrows():
            r = rr.to_dict()
            expanded = (r.get("型") in [top_ratio.get("型") if top_ratio else None, bottom_ratio.get("型") if bottom_ratio else None, top_parent.get("型") if top_parent else None])
            with st.expander(f"{r.get('型')}：結果から原因までさかのぼる", expanded=expanded):
                st.markdown(gene_chain(r))

        # --- Team causal chain ---
        st.markdown("#### 4. 赤チーム・青チーム差の数珠つなぎ")
        red_res = mean_any(["赤 平均所持資源"], np.nan, late)
        blue_res = mean_any(["青 平均所持資源"], np.nan, late)
        red_g = mean_any(["赤 Gini"], np.nan, late)
        blue_g = mean_any(["青 Gini"], np.nan, late)
        red_h = mean_any(["赤タカ比率（0-1）"], np.nan, late)
        blue_h = mean_any(["青タカ比率（0-1）"], np.nan, late)
        team_lines = []
        if not pd.isna(red1) and not pd.isna(blue1) and max(red1, blue1) > 0:
            if red1 > blue1 * 1.08:
                side, other = "赤", "青"; sr, orr, sg, og, sh, oh = red_res, blue_res, red_g, blue_g, red_h, blue_h
            elif blue1 > red1 * 1.08:
                side, other = "青", "赤"; sr, orr, sg, og, sh, oh = blue_res, red_res, blue_g, red_g, blue_h, red_h
            else:
                side = None
            if side is None:
                team_lines.append(f"赤 {cnt(red1)} 体、青 {cnt(blue1)} 体で、チーム差は極端ではありません。ここでは色そのものより、チーム内にどの遺伝子型が多いか、タカ/ハトや捕食傾向がどちらに偏るかを見ます。")
            else:
                team_lines.append(f"最新では **{side}チーム** が多いです。ただし、これは結論ではなく入口です。なぜ{side}が多いのかは、①資源を持てた、②格差が低く脱落が少なかった、③争奪/捕食/哲学型の構成が環境に合った、④相手や空きマスに恵まれて出生出口を通った、のどれかです。")
                if not pd.isna(sr) and not pd.isna(orr):
                    if sr > orr + 0.5:
                        team_lines.append(f"平均所持資源は{side}側が高めです（{fmt(sr,2)} 対 {fmt(orr,2)}）。では、なぜ高いのか。可能性は、{side}側が資源地帯に多くいる、採取型や資源収支の高い型が{side}に偏っている、争奪や捕食で追加資源を得ている、移動コストが相対的に低い、の4つです。つまり資源差は結果であり、その前には空間配置と遺伝子構成があります。")
                    elif sr < orr - 0.5:
                        team_lines.append(f"平均所持資源はむしろ{side}側が低めです。それでも{side}が多いなら、資源量以外、たとえば死亡回避、低いGini、交尾出口、または有利な同伴遺伝子が効いているはずです。")
                    else:
                        team_lines.append("平均所持資源に大差はありません。したがって、チーム差は資源量だけでは説明できません。格差、交尾出口、タカ/ハト構成、哲学型の偏りへ原因をさかのぼります。")
                if not pd.isna(sg) and not pd.isna(og):
                    if sg < og - 0.03:
                        team_lines.append(f"{side}側のGiniは低めです。これは重要です。Giniが低いということは、資源を持てない個体が少なく、維持コストで脱落しにくいということです。つまり{side}の多さは、平均資源よりも『資源の分配の安定』によって支えられている可能性があります。")
                    elif sg > og + 0.03:
                        team_lines.append(f"{side}側のGiniは高めです。数では多くても、内部では一部の個体に資源が偏っています。この場合、短期的には多く見えても、低資源個体の脱落で不安定化する可能性があります。")
                if not pd.isna(sh) and not pd.isna(oh):
                    if sh > oh + 0.05:
                        if contest_net_late > 0:
                            team_lines.append(f"{side}側はタカ比率が高く、争奪ネットも正です。ここではタカ性が環境に合っています。なぜ合うかというと、争奪で得られる資源が支払うコストを上回っているからです。したがって、{side}の優勢はチーム色そのものではなく、{side}に偏ったタカ遺伝子が資源獲得を増幅した結果かもしれません。")
                        else:
                            team_lines.append(f"{side}側はタカ比率が高いのに、争奪ネットは正ではありません。タカ性はコストになっている可能性があります。それでも{side}が多いなら、資源配置や繁殖出口がタカの負担を上回ったと考えます。")
                    elif sh < oh - 0.05:
                        if contest_net_late <= 0:
                            team_lines.append(f"{side}側はタカ比率が低く、争奪ネットも弱い/負です。これは、戦闘を避ける構成が消耗を減らし、死亡圧を下げたという説明につながります。")
                biased = []
                for r in rows:
                    rr, bb = r.get("赤比率", np.nan), r.get("青比率", np.nan)
                    if side == "赤" and not pd.isna(rr) and not pd.isna(bb) and rr > bb + 0.08:
                        biased.append(str(r.get("型")))
                    if side == "青" and not pd.isna(rr) and not pd.isna(bb) and bb > rr + 0.08:
                        biased.append(str(r.get("型")))
                if biased:
                    team_lines.append(f"{side}側には **{', '.join(biased)}** が偏っています。もしこれらの型が資源収支・親出生・低空腹率で有利なら、{side}の多さはチームの色ではなく、有利な遺伝子型が{side}に多く配置されたことから生じた可能性があります。")
                team_lines.append("最終的に、チーム差を説明するときは『色が強い』とは言いません。色は空間と相互作用相手をまとめるラベルです。資源、格差、タカ/ハト、捕食傾向、哲学型偏りがその色に集まったとき、色の差として見えるのです。")
        section("チーム差の因果鎖", join(team_lines) if team_lines else "赤青チーム差を読む列が不足しています。")

        # --- ranked causal hypotheses ---
        hypotheses = []
        if top_ratio:
            hypotheses.append({"結果": f"{top_ratio['型']}の比率上昇", "直接原因": "親出生が多い/死亡が少ない/他型が減った、のいずれか", "さらに前の原因": "資源収支・空腹率・同伴遺伝子・チーム偏りのうち、上の文章で支持されたもの", "検証": "同じseedで該当環境圧をOFFにし、順位が変わるかを見る"})
        if high_hunger:
            hypotheses.append({"結果": f"{high_hunger['型']}の空腹率が高い", "直接原因": "所持資源を維持コスト以上に保てない", "さらに前の原因": "足元資源が少ない、採取率が低い、移動コストが高い、捕食/争奪が失敗している", "検証": "局所資源再生・移動コスト・捕食OFFで変化を見る"})
        if not pd.isna(c_birth_bag):
            if c_birth_bag > 0.35:
                hypotheses.append({"結果": "出生が資源に強く連動", "直接原因": "資源を持つ個体が交尾・出生へ進みやすい", "さらに前の原因": "採取・争奪・捕食・局所資源接触の差が出生差へ変換されている", "検証": "資源量/局所資源再生を変えた時に優勢型が変わるか"})
            elif abs(c_birth_bag) < 0.2:
                hypotheses.append({"結果": "出生が資源だけでは説明できない", "直接原因": "相手探索・近親回避・空きマス・密度が出口になっている", "さらに前の原因": "資源を持っても繁殖場所や相手に接続できない", "検証": "密度依存OFF・近親回避OFFで出生差を見る"})
        st.markdown("#### 5. いま最も疑うべき因果仮説")
        if hypotheses:
            st.dataframe(pd.DataFrame(hypotheses), use_container_width=True, hide_index=True)
            st.markdown("この表の読み方は、結果から原因へ一段で止めず、さらに前の原因へ戻ることです。『資源を多く持つ』だけでは説明不足です。なぜ資源を持てたのか、どの行動で資源に接触したのか、どの別遺伝子がその行動を増幅したのか、どの環境条件なら同じ効果が消えるのか、までが一つの説明になります。")
        else:
            st.info("まだ因果仮説を作るには履歴が足りません。")

        # --- big picture ---
        ending = []
        ending.append("このモデルで面白いのは、遺伝子の価値が固定されていないことです。タカ遺伝子は、争奪が利益になる世界では資源獲得の道具になりますが、争奪が損になる世界では死亡や消耗への道になります。捕食遺伝子も、成功率が高いと資源への近道ですが、成功率が低いと危険への近道です。哲学遺伝子も、思想名そのものが強いのではなく、資源配置・密度・危険・相手探索と噛み合った時だけ強くなります。")
        ending.append("だから、ネオライフゲームの説明は『どれが勝ったか』では終わりません。勝ったように見える遺伝子が、どんな環境の中で、どの行動を通じて、どのボトルネックを抜けたのかを読む必要があります。生物の面白さは、強いものがいつも強いのではなく、環境と噛み合ったものがその時だけ強くなるところにあります。")
        section("生物として面白いところ", join(ending))

    # -------------------------
    # グラフ表示（単位ごとに分けて“読みやすく”）
    # -------------------------
    xcol = "世代（回）"

    # -------------------------
    # 分析ダッシュボード
    # -------------------------
    st.markdown("### 分析ダッシュボード")
    explain_box(
        "この画面の目的",
        "個体の勝敗ではなく、遺伝子コピーがどの環境条件によって増減したかを読む画面です。上部では現在値、下部の分析レポートでは原因経路を整理します。"
    )

    pop_now = _latest_num("個体数（体）")
    pop_delta = _trend_delta("個体数（体）")
    W_pop_now = _latest_num("個体群全体W（増殖率）", 0.0)
    birth_now = _latest_num("出生数（体/世代）")
    death_now = _latest_num("死亡数（体/世代）")
    birth_death_ratio = birth_now / max(death_now, 1.0)
    res_now = _latest_num("資源総量（単位）")
    res_delta = _trend_delta("資源総量（単位）")
    philo_div = _latest_num("哲学遺伝子多様度（Simpson）")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("個体数", f"{int(pop_now)}体", _fmt_delta(pop_delta, 0))
    m2.metric("個体群W", f"{W_pop_now:.3f}", "1.0が維持線")
    m3.metric("出生/死亡", f"{birth_death_ratio:.2f}", "1以上で増加圧")
    m4.metric("資源総量", f"{int(res_now)}", _fmt_delta(res_delta, 0))
    m5.metric("哲学遺伝子多様度", f"{philo_div:.3f}", "0低・1高")

    status_cols = st.columns(2)
    with status_cols[0]:
        if W_pop_now < 0.90:
            st.warning("個体群Wが1を大きく下回っています。死亡圧・維持コスト・交尾相手探索・資源回収のどこかが強すぎる可能性があります。")
        elif W_pop_now > 1.10:
            st.info("個体群Wが1を大きく上回っています。増殖圧が強い状態です。密度依存や資源制限が効くか確認してください。")
        else:
            st.success("個体群Wはおおむね維持線付近です。自然淘汰を観察しやすい状態に近いです。")
    with status_cols[1]:
        if res_delta > 0 and pop_delta < 0:
            st.warning("資源は増えているのに個体数が減っています。環境資源と個体行動の接続、つまり探索・採取・繁殖が弱い可能性があります。")
        elif res_delta < 0 and pop_delta > 0:
            st.info("個体数は増えていますが資源が減っています。増殖が資源ストックを食いつぶしていないか確認してください。")
        else:
            st.info("資源と個体数の増減方向が大きく矛盾していません。次は遺伝子別Wを見ます。")

    show_public_causal_report_v26()
    show_v20_comparison_mode()

    if st.checkbox("補助分析ログを表示する", value=False, help="開発・検算用です。外部説明では上の分析レポートを中心に見せてください。"):
        show_auto_interpretation()
        show_whole_run_summary()
        show_deep_causal_interpretation()
        show_philosophy_gene_flow_summary()
        show_v19_lineage_flow_summary()
        show_v21_deep_gene_causality()
        show_v22_environment_gene_report()
        show_philo_summary_table()

    tab_gene, tab_pop, tab_resource, tab_action = st.tabs(["遺伝子", "個体群", "資源", "行動"])
    with tab_gene:
        philo_ratio_cols = [f"{lab} 比率（0-1）" for lab in PHILO_LABELS.values()]
        philo_W_cols = [f"{lab} W" for lab in PHILO_LABELS.values()]
        plot_stacked_area(
            "行動型頻度（通常個体＋哲学個体）の積み上げ推移",
            xcol,
            philo_ratio_cols,
            "頻度（0-1）",
            "帯が太くなるほど、その哲学遺伝子が集団内で増えています。ただし全体個体数が落ちているときはWも必ず併読します。"
        )
        plot_latest_bar(
            "最新世代：行動型W",
            philo_W_cols,
            "W（コピー増殖率）",
            "1を超える型は前世代よりコピーを増やし、1未満の型はコピーを減らしています。"
        )
        plot_latest_bar(
            "最新世代：行動型別 資源収支ネット",
            [f"{lab} 資源収支ネット（単位/世代）" for lab in PHILO_LABELS.values()],
            "資源収支ネット（単位/世代）",
            "採取・捕食・戦闘獲得から、移動・維持・戦闘損失を引いた値です。繁殖への余剰を作れているかを見ます。"
        )
    with tab_pop:
        plot_lines(
            "個体群の増減",
            xcol,
            ["個体数（体）", "出生数（体/世代）", "死亡数（体/世代）"],
            "体 / 体世代",
        )
        plot_latest_bar(
            "最新世代：行動型別 空腹個体比率",
            [f"{lab} 空腹個体比率（0-1）" for lab in PHILO_LABELS.values()],
            "空腹比率（0-1）",
            "高い型は資源獲得に失敗しやすいか、消費が大きすぎる可能性があります。"
        )
    with tab_resource:
        plot_lines(
            "資源ストックと主要フロー",
            xcol,
            ["資源総量（単位）", "資源自然発生総量（単位/世代）", "採取総量（単位/世代）", "維持コスト支払総量（単位/世代）"],
            "資源量（単位）",
        )
        plot_latest_bar(
            "最新世代：資源フロー",
            ["資源自然発生総量（単位/世代）", "採取総量（単位/世代）", "維持コスト支払総量（単位/世代）", "移動コスト支払総量（単位/世代）", "子へ分配した資源（単位/世代）"],
            "資源量（単位/世代）",
            "発生・採取が維持・移動・繁殖投資を上回らないと、長期的には個体群が崩れます。"
        )
    with tab_action:
        plot_lines(
            "行動イベントの推移",
            xcol,
            ["交尾試行（回/世代）", "交尾成立（回/世代）", "捕食試行（回/世代）", "捕食成功（回/世代）", "戦闘回数（回/世代）", "争奪イベント数（回/世代）"],
            "回数（回/世代）",
        )
        plot_latest_bar(
            "最新世代：行動型別 捕食試行",
            [f"{lab} 捕食試行（回/世代）" for lab in PHILO_LABELS.values()],
            "捕食試行（回/世代）",
            "捕食が多い型は短期資源を得る可能性がありますが、失敗コスト・集団崩壊・非搾取型との比較が必要です。"
        )

    st.markdown("---")

    st.markdown("### 1) 資源ストック（盤面に存在する量）")
    st.caption("盤面に残っている資源量です。ここが増えているのに個体が減るなら、資源を発見・回収・繁殖へ変換できていません。")
    plot_area("資源総量の推移", xcol, "資源総量（単位）", "資源総量（単位）")
    plot_lines(
        "資源マス数/割合の推移",
        xcol,
        ["資源マス数（マス）", "資源マス割合（0-1）"],
        "（マス） / （0-1）"
    )

    st.markdown("### 2) 資源フロー（世代あたりの増減：単位/世代）")
    st.caption("世代ごとの資源の出入りです。自然発生・採取が供給側、維持・移動・繁殖投資が消費側です。")
    plot_lines(
        "資源フロー（単位/世代）",
        xcol,
        [
            "資源自然発生総量（単位/世代）",
            "採取総量（単位/世代）",
            "戦闘移転資源（単位/世代）",
            "子へ分配した資源（単位/世代）",
            "出生コスト支払総量（単位/世代）",
            "維持コスト支払総量（単位/世代）",
            "移動コスト支払総量（単位/世代）",
            "争奪で得たV合計（単位/世代）",
            "争奪で支払ったC合計（単位/世代）",
        ],
        "資源量（単位/世代）"
    )

    st.markdown("### 3) 個体イベント（体/世代）")
    st.caption("出生が死亡を継続的に下回ると、どの遺伝子が一時的に増えても集団は崩壊します。")
    plot_lines(
        "出生・死亡・出生失敗（体/世代）",
        xcol,
        ["出生数（体/世代）", "死亡数（体/世代）", "出生失敗（空きなし）（体/世代）"],
        "個体数（体/世代）"
    )

    st.markdown("### 4) 行動イベント（回/世代）")
    plot_lines(
        "戦闘・交尾・争奪（回/世代）",
        xcol,
        [
            "戦闘回数（回/世代）",
            "交尾試行（回/世代）",
            "交尾成立（回/世代）",
            "争奪イベント数（回/世代）",
            "タカ勝利数（回/世代）",
            "タカ同士争い（回/世代）",
            "弱者勝利（回/世代）",
            "捕食試行（回/世代）",
            "捕食成功（回/世代）",
            "近親交配回避（回/世代）",
        ],
        "回数（回/世代）"
    )

    st.markdown("### 5) 比率（0-1）")
    plot_lines(
        "比率の推移（0-1）",
        xcol,
        [
            "タカ比率（0-1）",
            "移動成功率（0-1）",
            "交尾成立率（0-1）",
            "弱者勝率（0-1）",
            "資源格差Gini（0-1）",
            "捕食成功率（0-1）",
            "捕食傾向比率（0-1）",
        ],
        "比率（0-1）"
    )
    st.markdown("### 5.1) 遺伝子の世代ごとの流れ")
    st.caption("この研究の中心です。頻度だけでなく、必ずWと個体群全体Wを一緒に見ます。")
    plot_lines(
        "哲学遺伝子の頻度変化",
        xcol,
        [f"{lab} 比率（0-1）" for lab in PHILO_LABELS.values()],
        "遺伝子頻度（0-1）"
    )
    plot_lines(
        "哲学遺伝子のコピー増殖率W",
        xcol,
        [f"{lab} W" for lab in PHILO_LABELS.values()] + ["個体群全体W（増殖率）"],
        "W（今世代コピー数 / 前世代コピー数）"
    )
    plot_lines(
        "争奪・捕食遺伝子の適応度W",
        xcol,
        [
            "タカ 適応度W（コピー増殖率）",
            "ハト 適応度W（コピー増殖率）",
            "非捕食 適応度W（コピー増殖率）",
            "捕食 適応度W（コピー増殖率）",
        ],
        "W（コピー増殖率）"
    )
    plot_lines(
        "遺伝子多様度",
        xcol,
        [
            "争奪遺伝子多様度（Simpson）",
            "捕食遺伝子多様度（Simpson）",
            "哲学遺伝子多様度（Simpson）",
        ],
        "Simpson多様度"
    )

    st.markdown("### 5.1.1) 行動型別の状態（通常個体＋哲学個体）")
    plot_lines(
        "行動型別：平均所持資源",
        xcol,
        [f"{lab} 平均所持資源（単位/体）" for lab in PHILO_LABELS.values()],
        "平均所持資源（単位/体）"
    )
    plot_lines(
        "行動型別：空腹個体比率",
        xcol,
        [f"{lab} 空腹個体比率（0-1）" for lab in PHILO_LABELS.values()],
        "空腹個体比率（0-1）"
    )
    plot_lines(
        "行動型別：平均局所密度",
        xcol,
        [f"{lab} 平均局所密度（体/近傍）" for lab in PHILO_LABELS.values()],
        "局所密度（体/近傍）"
    )

    st.markdown("### 5.1.2) 行動型別の行動フロー")
    plot_lines(
        "行動型別：出生予約",
        xcol,
        [f"{lab} 出生予約（体/世代）" for lab in PHILO_LABELS.values()],
        "出生予約（体/世代）"
    )
    plot_lines(
        "行動型別：実出生と死亡",
        xcol,
        [f"{lab} 実出生（体/世代）" for lab in PHILO_LABELS.values()] + [f"{lab} 死亡（体/世代）" for lab in PHILO_LABELS.values()],
        "個体数（体/世代）"
    )
    plot_lines(
        "行動型別：資源収支ネット",
        xcol,
        [f"{lab} 資源収支ネット（単位/世代）" for lab in PHILO_LABELS.values()],
        "資源収支ネット（単位/世代）"
    )
    plot_lines(
        "行動型別：捕食試行",
        xcol,
        [f"{lab} 捕食試行（回/世代）" for lab in PHILO_LABELS.values()],
        "捕食試行（回/世代）"
    )

    st.markdown("### 5.2) 遺伝子別の平均（タカ vs ハト）")
    plot_lines(
        "所持資源：タカ vs ハト",
        xcol,
        ["タカ 平均所持資源（単位/体）", "ハト 平均所持資源（単位/体）"],
        "平均所持資源（単位/体）"
    )
    plot_lines(
        "肉体強度：タカ vs ハト",
        xcol,
        ["タカ 平均肉体強度（値/体）", "ハト 平均肉体強度（値/体）"],
        "平均肉体強度（値/体）"
    )
    plot_lines(
        "認識半径：タカ vs ハト",
        xcol,
        ["タカ 平均認識半径（マス）", "ハト 平均認識半径（マス）"],
        "平均認識半径（マス）"
    )
    plot_lines(
        "年齢：タカ vs ハト",
        xcol,
        ["タカ 平均年齢（世代）", "ハト 平均年齢（世代）"],
        "平均年齢（世代）"
    )
    plot_lines(
        "資源格差Gini：タカ vs ハト",
        xcol,
        ["タカ 資源格差Gini（0-1）", "ハト 資源格差Gini（0-1）"],
        "Gini（0-1）"
    )

    st.markdown("### 5.3) チーム内のタカ比率（赤/青）")
    plot_lines(
        "赤/青チーム内タカ比率",
        xcol,
        ["赤タカ比率（0-1）", "青タカ比率（0-1）"],
        "タカ比率（0-1）"
    )


    st.markdown("### 6) 分布（最新世代の“個体内ばらつき”）")
    w_now = st.session_state.world
    bag_now = w_now.get("bag", np.array([], dtype=np.int32))
    age_now = w_now.get("age", np.array([], dtype=np.int32))
    str_now = w_now.get("strength", np.array([], dtype=np.int32))
    vis_now = w_now.get("vision", np.array([], dtype=np.int32))

    col1, col2 = st.columns(2)
    with col1:
        plot_hist("所持資源の分布", bag_now, "所持資源（単位）")
        plot_hist("年齢の分布", age_now, "年齢（世代）")
    with col2:
        plot_hist("肉体強度の分布", str_now, "肉体強度（値）")
        plot_hist("認識半径の分布", vis_now, "認識半径（マス）")

    st.markdown("### 7) 表（直近の統計）")
    st.dataframe(dff, use_container_width=True)