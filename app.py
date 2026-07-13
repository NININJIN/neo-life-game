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
st.caption("個体が資源を集め、移動し、交尾し、争奪や捕食の圧を受けながら、どの遺伝子がどの条件で残るのかを観察する進化シミュレーションです。")

st.info("このシミュレーションでは、個体が資源を集め、移動し、交尾し、争奪や捕食の圧を受けながら、どの遺伝子がどの環境で残るのかを観察できます。まずはリセットしてから、1世代ずつ進めるか、自動実行で変化を見てください。")

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

# 哲学型判断遺伝子
# ここで扱う「デカルト型・ヒューム型・カント型」は、実在する遺伝子ではない。
# 哲学者の思想全体を遺伝子化するのでもない。
# それぞれの哲学に特徴的な「限られた情報からどう判断するか」を抽出し、
# 親から子へ継承される仮想的な判断形質としてモデル化する。
# 0 デカルト型：十分に確実な差が出るまで判断を保留しやすい。
# 1 ヒューム型：局所観察と過去に似た状況での成功を重く見る。
# 2 カント型：先天的な規則で情報を整理し、経験と合わせて判断する。
# 3 通常個体：哲学型判断遺伝子を持たず、基本判断アルゴリズムで動く対照群。
NORMAL_PHILO_VALUE = 3
PHILO_TYPE_COUNT = 4
PHILOSOPHY_VALUES = (0, 1, 2)

PHILO_LABELS = {
    0: "デカルト型",
    1: "ヒューム型",
    2: "カント型",
    3: "通常個体",
}

PHILO_THEORY = {
    0: "デカルト型：判断の確実性を重視する。行動候補の差が小さいときは、急いで動くより待機・回避を選びやすい。誤認は減るが、資源や交尾の機会を逃すことがある。",
    1: "ヒューム型：経験と局所観察を重視する。見えている資源、足元の環境、過去に成功しやすかった状況へ反応しやすい。環境が安定していれば強いが、急変には遅れることがある。",
    2: "カント型：生得的な判断規則と経験を組み合わせる。危険、近さ、資源の偏り、持続可能性を一定の枠組みで整理する。未経験でも動けるが、枠組みが環境に合わないと誤る。",
    3: "通常個体：哲学型判断遺伝子を持たない対照群。資源、危険、移動コストなどを基本アルゴリズムで直接評価して行動する。",
}


def philo_active_values():
    """サイドバーでONになっている哲学型の値を返す。全OFFならヒューム型だけを使う。"""
    flags = {
        0: bool(globals().get('philo_enable_descartes', True)),
        1: bool(globals().get('philo_enable_hume', True)),
        2: bool(globals().get('philo_enable_kant', True)),
    }
    vals = [k for k, v in flags.items() if v]
    return vals if vals else [1]


def philo_choice_values_probs():
    """初期生成・古い世界の補修用。ONの型だけから、重みに従って選ぶ。"""
    vals = philo_active_values()
    weight_map = {
        0: int(globals().get('philo_weight_descartes', 25)),
        1: int(globals().get('philo_weight_hume', 25)),
        2: int(globals().get('philo_weight_kant', 25)),
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
    """通常個体と哲学個体を初期割合に従って、できるだけ厳密に生成する。"""
    n = int(n)
    if n <= 0:
        return np.array([], dtype=np.int8)

    vals, probs = philo_choice_values_probs()
    normal_pct = float(globals().get('initial_normal_pct', 25))
    normal_n = int(round(n * np.clip(normal_pct, 0.0, 100.0) / 100.0))
    normal_n = max(0, min(n, normal_n))
    philo_n = n - normal_n

    arr_parts = []
    if normal_n > 0:
        arr_parts.append(np.full(normal_n, int(NORMAL_PHILO_VALUE), dtype=np.int8))

    if philo_n > 0:
        raw = probs * philo_n
        counts = np.floor(raw).astype(int)
        rest = int(philo_n - counts.sum())
        if rest > 0:
            order = np.argsort(-(raw - counts))
            for k in order[:rest]:
                counts[int(k)] += 1
        philo_vals = []
        for v, c in zip(vals, counts):
            if int(c) > 0:
                philo_vals.append(np.full(int(c), int(v), dtype=np.int8))
        if philo_vals:
            arr_parts.append(np.concatenate(philo_vals))

    arr = np.concatenate(arr_parts) if arr_parts else np.array([], dtype=np.int8)
    rng.shuffle(arr)
    return arr.astype(np.int8)

PHILO_STAT_KEYS = [
    "stat_philo_birth_reserved", "stat_philo_birth_real", "stat_philo_death",
    "stat_philo_gather_gain", "stat_philo_move_cost", "stat_philo_upkeep_cost",
    "stat_philo_mate_attempt", "stat_philo_mate_success",
    "stat_philo_predation_attempt", "stat_philo_predation_success", "stat_philo_predation_fail",
    "stat_philo_predation_gain", "stat_philo_battle_gain", "stat_philo_battle_cost",
    "stat_philo_parent_offspring_reserved", "stat_philo_parent_offspring_real",
    "stat_philo_decision_hold", "stat_philo_opportunity_loss",
    "stat_philo_memory_used", "stat_philo_memory_positive",
    "stat_philo_kant_frame_match", "stat_philo_kant_frame_mismatch",
]

PHILO_ACTION_LABELS = {
    0: "待機",
    1: "移動",
    2: "採取",
    3: "戦闘",
    4: "回避",
    5: "交尾",
    6: "捕食",
}

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
    st.subheader("実験環境")
    environment_scenario = st.selectbox(
        "環境タイプ",
        ["手動設定", "安定環境", "不確実環境", "急変環境"],
        index=0,
        help="データ収集用の大枠です。手動設定では各スライダーをそのまま使います。急変環境では指定世代以降、豊かなバイオームの位置づけを反転させます。",
    )
    environment_shift_generation = st.slider("急変環境：変化が起きる世代", 20, 500, 120, 10)

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
    st.subheader("哲学型判断遺伝子")
    enable_philo_gene = st.checkbox("哲学型判断遺伝子を行動に反映する", value=True)
    philo_effect = st.slider("判断遺伝子の影響強度", 0.0, 2.0, 1.0, 0.05)
    initial_normal_pct = st.slider("初期：通常個体割合（%）", 0, 100, 25, 1)
    st.caption("初期設定は、通常個体・デカルト型・ヒューム型・カント型を各25％にします。通常個体は哲学型判断遺伝子を持たない対照群です。")
    st.caption("ここでいう遺伝子は実在の遺伝子ではなく、判断方法を親から子へ継承される仮想形質として扱うためのモデル上の名前です。")

    with st.expander("三つの哲学型のON/OFF・初期重み", expanded=True):
        st.caption("OFFにした型は初期個体にも、古い個体群の補修にも使われません。全OFFなら安全のためヒューム型だけにします。")
        philo_enable_descartes = st.checkbox("デカルト型を使う", value=True)
        philo_weight_descartes = st.slider("初期重み：デカルト型", 0, 100, 25, 1)
        philo_enable_hume = st.checkbox("ヒューム型を使う", value=True)
        philo_weight_hume = st.slider("初期重み：ヒューム型", 0, 100, 25, 1)
        philo_enable_kant = st.checkbox("カント型を使う", value=True)
        philo_weight_kant = st.slider("初期重み：カント型", 0, 100, 25, 1)

    with st.expander("操作的定義", expanded=False):
        for _k, _v in PHILO_THEORY.items():
            st.markdown(f"- **{PHILO_LABELS[_k]}**：{_v}")
        st.caption("思想家本人の完全再現ではなく、哲学に特徴的な判断原理だけを取り出し、行動評価関数へ変換したモデルです。")

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
    reset_btn = a.button("↻ リセット", width="stretch")
    step_gen_btn = b.button("▶ 1世代", width="stretch")
    step_10_btn = c.button("▶▶ 10世代", width="stretch")
    step_50_btn = d.button("▶▶▶ 50世代", width="stretch")
    step_btn = e.button("1フェーズ", width="stretch")
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

    # チーム別イベント統計。赤=0 / 青=1。低密度化による繁殖ネットワーク崩壊を読むために使う。
    for key in ["stat_team_birth_reserved", "stat_team_birth_real", "stat_team_death", "stat_team_mate_attempt", "stat_team_mate_success"]:
        if key not in w or np.asarray(w.get(key, [])).size != 2:
            w[key] = np.zeros(2, dtype=np.int32)
        else:
            w[key] = np.asarray(w[key], dtype=np.int32)

    # ヒューム型の経験参照用。各バイオームで過去に資源が取れたかを、軽い移動平均として持つ。
    # 個体ごとの長大な記憶ではなく、まずは「環境への経験的重み」が行動に効くかを見るための安全な実装。
    k_now = int(globals().get('biome_k', 3))
    mem = np.asarray(w.get("hume_biome_memory", np.zeros(k_now, dtype=np.float32)), dtype=np.float32)
    if mem.size != k_now:
        new_mem = np.zeros(k_now, dtype=np.float32)
        m = min(mem.size, k_now)
        if m > 0:
            new_mem[:m] = mem[:m]
        mem = new_mem
    w["hume_biome_memory"] = mem

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
    """哲学型判断遺伝子が、同じ環境入力をどう評価するかを変える。

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
        # デカルト型：不確実な行動を抑え、危険・失敗コストに強く反応する。
        # 正確さは増えるが、採取・交尾・捕食の機会を逃しやすい。
        raw = (0.95, 1.45, -3.0, -0.5, -3.0, 2.5)
    elif p == 1:
        # ヒューム型：見えている資源と近傍経験を重く評価する。
        # 安定環境では素早く資源へ向かえるが、環境急変では過去の成功に引きずられやすい。
        raw = (1.35, 0.95, -1.0, 0.0, -1.0, -0.2)
    elif p == 2:
        # カント型：生得的な整理規則を使い、危険・繁殖・持続可能性を同時に見る。
        # 単純な中間ではなく、行動を一定の枠組みで整理する型として扱う。
        raw = (1.05, 1.20, -4.5, 2.5, -4.0, 1.2)
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
        "stat_team_birth_reserved": np.zeros(2, dtype=np.int32),
        "stat_team_birth_real": np.zeros(2, dtype=np.int32),
        "stat_team_death": np.zeros(2, dtype=np.int32),
        "stat_team_mate_attempt": np.zeros(2, dtype=np.int32),
        "stat_team_mate_success": np.zeros(2, dtype=np.int32),
        "hume_biome_memory": np.zeros(int(biome_k), dtype=np.float32),
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
        bool(philo_enable_descartes), bool(philo_enable_hume), bool(philo_enable_kant),
        int(philo_weight_descartes), int(philo_weight_hume), int(philo_weight_kant),
        bool(enable_predation), int(predation_gene_init_pct),
        int(predation_hunger_threshold), float(predation_gain_rate), int(predation_fail_cost),
        str(environment_scenario), int(environment_shift_generation),
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
    # 実験環境プリセット：急変環境では途中から「どのバイオームが豊かか」を反転させる。
    try:
        if str(environment_scenario) == "急変環境" and int(st.session_state.gen) >= int(environment_shift_generation):
            biome_factor = biome_factor[::-1].copy()
    except Exception:
        pass

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
    # 不確実環境：資源発生の揺らぎを少し増やす。認識の誤差そのものではなく、環境側の予測しにくさとして入れる。
    try:
        if str(environment_scenario) == "不確実環境":
            noise = rng.normal(1.0, 0.35, size=(H, W)).astype(np.float32)
            prob_map = prob_map * np.clip(noise, 0.15, 2.0)
        elif str(environment_scenario) == "安定環境":
            prob_map = prob_map * 1.05
    except Exception:
        pass
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
            child_team_real = int(child.get("team", 0))
            if child_team_real in (0, 1):
                w["stat_team_birth_real"][child_team_real] += 1

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
    hume_biome_memory = np.asarray(w.get("hume_biome_memory", np.zeros(k, dtype=np.float32)), dtype=np.float32)
    if hume_biome_memory.size != k:
        hume_biome_memory = np.zeros(k, dtype=np.float32)
        w["hume_biome_memory"] = hume_biome_memory

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
        if philo_index(my_philo) == 1 and float(np.max(np.abs(hume_biome_memory))) > 0.05:
            w["stat_philo_memory_used"][1] += 1
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
            # ヒューム型：過去に資源を得やすかったバイオームを少し高く評価する。
            if philo_index(my_philo) == 1:
                mem_v = float(hume_biome_memory[int(biome_id[ty, tx])])
                res_score += 1.15 * mem_v
                if mem_v > 0.10:
                    w["stat_philo_memory_positive"][1] += 1

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

        # デカルト型は、行動候補の利得が小さく、確信が弱いときに判断を保留しやすい。
        # これは方法的懐疑の完全再現ではなく、「疑わしい情報をすぐ行動根拠にしない」性質だけを抽出したもの。
        if enable_philo_gene and philo_index(my_philo) == 0 and best_a not in (0, 4):
            certainty_threshold = 1.8 + 0.6 * float(hunger01)
            if float(best_u) < certainty_threshold:
                w["stat_philo_decision_hold"][0] += 1
                if int(best_a) in (1, 2, 5, 6) and float(best_u) > 0.0:
                    w["stat_philo_opportunity_loss"][0] += 1
                best_y, best_x, best_a = y0, x0, 0

        # カント型：生得的な判断枠組みが、実際に選ばれた行動と噛み合ったかを粗く記録する。
        # 危険が見えているなら回避/待機、資源が見えているなら採取/移動を「枠組みと一致」と見る。
        if enable_philo_gene and philo_index(my_philo) == 2:
            danger_here = 0
            for oy, ox in neigh1:
                yy = (y0 + oy) % H
                xx = (x0 + ox) % W
                j = pos_to_idx.get((yy, xx))
                if j is not None and j != i and int(team[j]) != my_team:
                    danger_here += max(1, int(strength[j]) - my_str)
            if danger_here > 0:
                if int(best_a) in (0, 4):
                    w["stat_philo_kant_frame_match"][2] += 1
                else:
                    w["stat_philo_kant_frame_mismatch"][2] += 1
            elif visible_total_res > 0:
                if int(best_a) in (1, 2):
                    w["stat_philo_kant_frame_match"][2] += 1
                elif int(best_a) in (0, 4):
                    w["stat_philo_kant_frame_mismatch"][2] += 1

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
    biome_id = w["biome_id"]  # ヒューム型の経験更新で参照するため、行動フェーズ内でも明示的に取得
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
                team_pair = int(team[a])
                if team_pair in (0, 1):
                    w["stat_team_mate_attempt"][team_pair] += 1
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
                if int(child_team) in (0, 1):
                    w["stat_team_birth_reserved"][int(child_team)] += 1

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
                if team_pair in (0, 1):
                    w["stat_team_mate_success"][team_pair] += 1
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
            if ph_i == 1:
                bidx = int(biome_id[y, x])
                mem = np.asarray(w.get("hume_biome_memory", np.zeros(int(biome_k), dtype=np.float32)), dtype=np.float32)
                if mem.size == int(biome_k):
                    # 成功したバイオームを少し高く、失敗したバイオームを少し低く評価する。
                    target = float(take)
                    mem[bidx] = 0.90 * float(mem[bidx]) + 0.10 * target
                    w["hume_biome_memory"] = mem
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
        "設定:環境タイプ": str(environment_scenario),
        "設定:急変世代": int(environment_shift_generation),
        "環境変化後世代（回）": max(0, int(st.session_state.gen) - int(environment_shift_generation)) if str(environment_scenario) == "急変環境" else 0,

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
        "デカルト型 数（体）": int(philo_counts[0]),
        "ヒューム型 数（体）": int(philo_counts[1]),
        "カント型 数（体）": int(philo_counts[2]),
        "デカルト型 比率（0-1）": float(philo_counts[0]) / max(n, 1),
        "ヒューム型 比率（0-1）": float(philo_counts[1]) / max(n, 1),
        "カント型 比率（0-1）": float(philo_counts[2]) / max(n, 1),
        "デカルト型 W": float(W_philo[0]),
        "ヒューム型 W": float(W_philo[1]),
        "カント型 W": float(W_philo[2]),
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
        "赤 出生予約（体/世代）": int(w.get("stat_team_birth_reserved", np.zeros(2, dtype=np.int32))[0]),
        "青 出生予約（体/世代）": int(w.get("stat_team_birth_reserved", np.zeros(2, dtype=np.int32))[1]),
        "赤 実出生（体/世代）": int(w.get("stat_team_birth_real", np.zeros(2, dtype=np.int32))[0]),
        "青 実出生（体/世代）": int(w.get("stat_team_birth_real", np.zeros(2, dtype=np.int32))[1]),
        "赤 死亡（体/世代）": int(w.get("stat_team_death", np.zeros(2, dtype=np.int32))[0]),
        "青 死亡（体/世代）": int(w.get("stat_team_death", np.zeros(2, dtype=np.int32))[1]),
        "赤 交尾試行（回/世代）": int(w.get("stat_team_mate_attempt", np.zeros(2, dtype=np.int32))[0]),
        "青 交尾試行（回/世代）": int(w.get("stat_team_mate_attempt", np.zeros(2, dtype=np.int32))[1]),
        "赤 交尾成立（回/世代）": int(w.get("stat_team_mate_success", np.zeros(2, dtype=np.int32))[0]),
        "青 交尾成立（回/世代）": int(w.get("stat_team_mate_success", np.zeros(2, dtype=np.int32))[1]),
        
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
        row[f"{ph_label} タカ比率（0-1）"] = float((gene[mask] == 0).mean()) if int(mask.sum()) > 0 else 0.0
        row[f"{ph_label} 捕食傾向比率（0-1）"] = float((gene_pred[mask] == 1).mean()) if int(mask.sum()) > 0 else 0.0

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
        row[f"{ph_label} 判断保留（回/世代）"] = int(w["stat_philo_decision_hold"][ph_i])
        row[f"{ph_label} 機会損失候補（回/世代）"] = int(w["stat_philo_opportunity_loss"][ph_i])
        row[f"{ph_label} 経験参照（回/世代）"] = int(w["stat_philo_memory_used"][ph_i])
        row[f"{ph_label} 経験が資源方向を後押し（回/世代）"] = int(w["stat_philo_memory_positive"][ph_i])
        row[f"{ph_label} 枠組み一致（回/世代）"] = int(w["stat_philo_kant_frame_match"][ph_i])
        row[f"{ph_label} 枠組み不一致候補（回/世代）"] = int(w["stat_philo_kant_frame_mismatch"][ph_i])

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
    for key in ["stat_team_birth_reserved", "stat_team_birth_real", "stat_team_death", "stat_team_mate_attempt", "stat_team_mate_success"]:
        w[key] = np.zeros(2, dtype=np.int32)
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
    w["stat_team_death"] = np.bincount(team[dead].astype(np.int32), minlength=2).astype(np.int32)
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
        label="表示を選ぶ",
        options=VIEW_TABS,
        horizontal=True,
        key="view_tab",
        label_visibility="collapsed",
    )

# -------------------------
# 各ビュー表示（元の tabs 内容をそのまま移植）
# -------------------------
if view == "環境":
    st.image(upscale_with_grid(base), width="stretch")

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

    st.image(np.array(pil), width="stretch")

elif view == "②認識":
    img = base
    if show_perception:
        img = overlay_perception(img)
    st.image(upscale_with_grid(img), width="stretch")

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
        st.image(big2, width="stretch")
    else:
        st.image(big, width="stretch")

elif view == "④行動":
    big = upscale_with_grid(base)
    prev_y, prev_x = w.get("last_prev", (ys, xs))
    act = w.get("last_act", np.zeros(n_agents, dtype=np.int8))
    if len(prev_x) == n_agents and n_agents > 0:
        big2 = draw_thinking(big, prev_y, prev_x, ys, xs, act)
        st.image(big2, width="stretch")
    else:
        st.image(big, width="stretch")

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

    st.image(np.array(pil), width="stretch")

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
        width="stretch"
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
            st.altair_chart(ch, width="stretch")
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
            st.altair_chart(ch, width="stretch")
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
            st.altair_chart(ch, width="stretch")
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
            st.altair_chart(ch, width="stretch")
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
            st.altair_chart((bars + text).properties(title=title, height=max(180, 28 * len(cols))), width="stretch")
        else:
            st.bar_chart(data.set_index("指標"))

    def show_reading_guide():
        with st.expander("数値の読み方・この研究での意味", expanded=False):
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
            st.dataframe(pd.DataFrame(guide_rows), width="stretch", hide_index=True)

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
                width="stretch",
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
            st.dataframe(pd.DataFrame(cause_rows), width="stretch", hide_index=True)

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
                    width="stretch",
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
                }), width="stretch", hide_index=True)


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
            }), width="stretch", hide_index=True)

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

        st.markdown("### 0.6) v19/v20 親子遺伝子フロー：どの型が、どの経路で増えたか")
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
        st.dataframe(parent_df, width="stretch", hide_index=True, column_config={
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
        st.dataframe(mat_parent_child.style.format("{:.0f}"), width="stretch")
        st.markdown("#### C. 親組み合わせ行列")
        st.caption("どの型同士のペアが実出生につながったかです。対角線は同型同士、対角線以外は混合ペアです。")
        st.dataframe(mat_pair.style.format("{:.0f}"), width="stretch")
        st.markdown("#### D. コピー元→子フロー行列")
        st.caption("子が実際にどの型のコピーとして生まれたかです。現段階では突然変異を入れていないので基本的に対角線に出ます。将来、突然変異や文化的変換を入れたときに重要になります。")
        st.dataframe(mat_source_child.style.format("{:.0f}"), width="stretch")

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
        st.dataframe(action_df, width="stretch", hide_index=True, column_config={
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
        st.dataframe(team_df, width="stretch", hide_index=True, column_config={
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
            st.dataframe(corr_df, width="stretch", hide_index=True, column_config={
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
            st.dataframe(gene_df, width="stretch", hide_index=True, column_config={
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
            st.dataframe(cdf, width="stretch", hide_index=True)
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
        st.dataframe(pressure_df, width="stretch", hide_index=True)



    def show_v20_comparison_mode():
        """v20：同じseedで条件だけを変え、観察された差分を因果候補として読む。"""
        st.markdown("### 0.7) v20 比較実験モード：同じseedで条件だけを変える")
        explain_box(
            "なぜ比較実験が必要か",
            "一つのランだけでは、遺伝子頻度の変化が本当にその遺伝子の効果なのか、初期配置・資源配置・偶然の出生死亡で起きたのかを分けにくいです。v20では、同じseedを使ったまま一つの条件だけを変えて走らせます。すると、初期配置の偶然をかなりそろえたうえで、哲学遺伝子・通常個体割合・捕食・密度依存・局所資源再生・近親回避がどの程度結果を変えたかを比較できます。これは因果の証明そのものではありませんが、単なる相関よりずっと強い因果候補になります。"
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

        with st.expander("v20 比較実験を実行する", expanded=False):
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
            run_compare = st.button("同じseedで比較実験を実行", width="stretch", key="v20_run_compare")
            st.caption("軽量化のため初期値は40世代×1反復・条件3つにしています。必要なときだけ世代数・条件・反復数を増やしてください。")

            if run_compare:
                specs = [("基準", {})] + [(name, scenario_options[name]) for name in selected_scenarios]
                total_runs = len(specs) * int(compare_repeats)
                if total_runs > 18:
                    st.warning(f"比較条件が多く、{total_runs}本の内部ランになります。Cloudでは重くなりやすいので、条件数か反復数を減らすのがおすすめです。")
                try:
                    with st.spinner("v20比較実験を実行中です。現在の世界はあとで復元します。"):
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
        st.dataframe(agg, width="stretch", hide_index=True, column_config=fmt_cols)

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
            st.dataframe(ph_agg, width="stretch", hide_index=True, column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in philo_delta_cols})

        # テキスト解釈：なぜその差が出たと読めるか。
        st.markdown("#### v20-E. 比較から読める因果候補")
        causal_lines = _v20_causal_text(agg)
        for line in causal_lines:
            st.markdown(f"- {line}")

        with st.expander("v20 生データ：各seed反復ごとの結果", expanded=False):
            st.dataframe(raw_df, width="stretch", hide_index=True)

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



    def show_public_story_report():
        """外部公開向けの、文章で読ませる分析レポート。
        目的は「どちらが多いか」ではなく、何がその結果を作ったのかを、
        環境・行動・遺伝子の順に戻って説明すること。
        """
        def has(col: str) -> bool:
            return col in df.columns

        def ser(col: str):
            if col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce").dropna()

        def last(col: str, default=0.0) -> float:
            s = ser(col)
            return float(s.iloc[-1]) if len(s) else float(default)

        def first(col: str, default=0.0) -> float:
            s = ser(col)
            return float(s.iloc[0]) if len(s) else float(default)

        def mean(col: str, frame=None, default=0.0) -> float:
            if frame is None:
                s = ser(col)
            else:
                if col not in frame.columns:
                    return float(default)
                s = pd.to_numeric(frame[col], errors="coerce").dropna()
            return float(s.mean()) if len(s) else float(default)

        def delta(col: str, default=0.0) -> float:
            s = ser(col)
            return float(s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else float(default)

        def recent_frame(k: int = 50):
            return df.tail(min(k, len(df))).copy()

        def corr(a: str, b: str):
            if a not in df.columns or b not in df.columns:
                return np.nan
            d = pd.concat([pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")], axis=1).dropna()
            if len(d) < 4:
                return np.nan
            return float(d.iloc[:, 0].corr(d.iloc[:, 1]))

        def fmt(v, nd=2):
            try:
                if np.isnan(float(v)):
                    return "不足"
            except Exception:
                return "不足"
            return f"{float(v):.{nd}f}"

        def sign_word(v, eps=1e-9):
            if v > eps:
                return "増えている"
            if v < -eps:
                return "減っている"
            return "ほぼ変わっていない"

        def strongest(items):
            items = [x for x in items if x[0] is not None and not np.isnan(x[0])]
            if not items:
                return None
            return sorted(items, key=lambda x: abs(float(x[0])), reverse=True)[0]

        rf = recent_frame(50)
        n_all = len(df)
        if n_all < 2:
            st.info("まだ十分な履歴がありません。数世代進めると、環境・行動・遺伝子の関係が読み取れるようになります。")
            return

        pop0, pop1 = first("個体数（体)"), last("個体数（体)")
        if not has("個体数（体)"):
            pop0, pop1 = first("個体数（体）"), last("個体数（体）")
        pop_change = pop1 - pop0
        W_recent = mean("個体群全体W（増殖率）", rf, last("個体群全体W（増殖率）", 1.0))
        births_recent = mean("出生数（体/世代）", rf)
        deaths_recent = mean("死亡数（体/世代）", rf)
        bd = births_recent / max(deaths_recent, 1e-9)

        res0, res1 = first("資源総量（単位）"), last("資源総量（単位）")
        bag0, bag1 = first("平均所持資源（単位/体）"), last("平均所持資源（単位/体）")
        gini0, gini1 = first("資源格差Gini（0-1）"), last("資源格差Gini（0-1）")
        gather_recent = mean("採取総量（単位/世代）", rf)
        spawn_recent = mean("資源自然発生総量（単位/世代）", rf)
        upkeep_recent = mean("維持コスト支払総量（単位/世代）", rf)
        move_recent = mean("移動コスト支払総量（単位/世代）", rf)
        child_recent = mean("子へ分配した資源（単位/世代）", rf)
        energy_simple = gather_recent - upkeep_recent - move_recent - child_recent
        use_ratio = gather_recent / max(spawn_recent, 1e-9)

        mate_attempt = mean("交尾試行（回/世代）", rf)
        mate_success = mean("交尾成立（回/世代）", rf)
        mate_rate = mate_success / max(mate_attempt, 1e-9)
        over_birth = mean("過密で抑制された出生候補（回/世代）", rf)
        kin_avoid = mean("近親交配回避（回/世代）", rf)
        pred_attempt = mean("捕食試行（回/世代）", rf)
        pred_success = mean("捕食成功（回/世代）", rf)
        pred_rate = pred_success / max(pred_attempt, 1e-9)
        contest_events = mean("争奪イベント数（回/世代）", rf)
        contest_gain = mean("争奪で得たV合計（単位/世代）", rf)
        contest_cost = mean("争奪で支払ったC合計（単位/世代）", rf)
        contest_net = contest_gain - contest_cost

        def profile(label: str):
            cnt0 = first(f"{label} 数（体）", np.nan)
            cnt1 = last(f"{label} 数（体）", np.nan)
            if label == "通常個体":
                cnt0 = first("通常個体数（体）", cnt0)
                cnt1 = last("通常個体数（体）", cnt1)
            ratio0 = first(f"{label} 比率（0-1）", np.nan)
            ratio1 = last(f"{label} 比率（0-1）", np.nan)
            if label == "通常個体":
                ratio0 = first("通常個体割合（0-1）", ratio0)
                ratio1 = last("通常個体割合（0-1）", ratio1)
            return {
                "型": label,
                "初期数": cnt0,
                "最新数": cnt1,
                "数変化": cnt1 - cnt0 if not np.isnan(cnt0) and not np.isnan(cnt1) else np.nan,
                "初期比率": ratio0,
                "最新比率": ratio1,
                "比率変化": ratio1 - ratio0 if not np.isnan(ratio0) and not np.isnan(ratio1) else np.nan,
                "平均W": mean(f"{label} W", rf, last(f"{label} W", np.nan)),
                "資源収支": mean(f"{label} 資源収支ネット（単位/世代）", rf, np.nan),
                "平均所持資源": mean(f"{label} 平均所持資源（単位/体）", rf, np.nan),
                "足元資源": mean(f"{label} 平均足元資源（単位/マス）", rf, np.nan),
                "局所密度": mean(f"{label} 平均局所密度（体/近傍）", rf, np.nan),
                "空腹率": mean(f"{label} 空腹個体比率（0-1）", rf, np.nan),
                "実出生": mean(f"{label} 実出生（体/世代）", rf, 0.0),
                "死亡": mean(f"{label} 死亡（体/世代）", rf, 0.0),
                "親出生": mean(f"{label} 親参加:実出生（回/世代）", rf, np.nan),
                "交尾成功": mean(f"{label} 交尾成功参加（回/世代）", rf, 0.0),
                "採取率": mean(f"{label} 行動率:採取（0-1）", rf, np.nan),
                "移動率": mean(f"{label} 行動率:移動（0-1）", rf, np.nan),
                "交尾率": mean(f"{label} 行動率:交尾（0-1）", rf, np.nan),
                "回避率": mean(f"{label} 行動率:回避（0-1）", rf, np.nan),
                "捕食率": mean(f"{label} 行動率:捕食（0-1）", rf, np.nan),
                "タカ比率": mean(f"{label} タカ比率（0-1）", rf, np.nan),
                "捕食傾向": mean(f"{label} 捕食傾向比率（0-1）", rf, np.nan),
                "赤内比率": mean(f"赤×{label} 比率（赤内0-1）", rf, np.nan),
                "青内比率": mean(f"青×{label} 比率（青内0-1）", rf, np.nan),
            }

        labels = []
        if has("通常個体数（体）") or has("通常個体割合（0-1）"):
            labels.append("通常個体")
        for lab in PHILO_LABELS.values():
            if lab != "通常個体":
                labels.append(lab)
        profiles = [profile(lab) for lab in labels]
        valid_profiles = [p for p in profiles if not np.isnan(p.get("最新比率", np.nan))]
        dominant = sorted(valid_profiles, key=lambda p: p.get("比率変化", -999), reverse=True)[0] if valid_profiles else None
        weakest = sorted(valid_profiles, key=lambda p: p.get("比率変化", 999))[0] if valid_profiles else None

        def explain_gene(p):
            lab = p["型"]
            lines = []
            change = p.get("比率変化", np.nan)
            count_change = p.get("数変化", np.nan)
            if np.isnan(change):
                return f"**{lab}** は、必要な列がまだ不足しているため詳しい読み取りができません。"
            if change > 0.03:
                opening = f"**{lab}** は比率を **{p['初期比率']:.3f}→{p['最新比率']:.3f}** へ上げています。"
            elif change < -0.03:
                opening = f"**{lab}** は比率を **{p['初期比率']:.3f}→{p['最新比率']:.3f}** へ下げています。"
            else:
                opening = f"**{lab}** の比率は **{p['初期比率']:.3f}→{p['最新比率']:.3f}** で大きくは動いていません。"
            if not np.isnan(count_change):
                opening += f" 実数では **{int(p['初期数'])}→{int(p['最新数'])}体** です。"
            lines.append(opening)

            routes = []
            # 資源ルート
            if not np.isnan(p["資源収支"]):
                if p["資源収支"] > 0:
                    why = []
                    if not np.isnan(p["足元資源"]) and p["足元資源"] > mean("平均足元資源（単位/マス）", rf, 0.0):
                        why.append("足元資源が比較的多い場所にいる")
                    if not np.isnan(p["採取率"]) and p["採取率"] > 0.15:
                        why.append("採取行動を取りやすい")
                    if not np.isnan(p["タカ比率"]) and p["タカ比率"] > mean("タカ比率（0-1）", rf, 0.0) and contest_net > 0:
                        why.append("タカ的な争奪が資源獲得として働いている")
                    if not np.isnan(p["捕食傾向"]) and p["捕食傾向"] > 0.15 and pred_rate > 0.3:
                        why.append("捕食が追加資源として成立している")
                    routes.append("資源面では、直近の資源収支が正です。" + (" その前段として、" + "、".join(why) + "ことが疑えます。" if why else " ただし、採取・争奪・捕食のどれが主因かは追加比較が必要です。"))
                else:
                    why = []
                    if not np.isnan(p["足元資源"]) and p["足元資源"] < mean("平均足元資源（単位/マス）", rf, 0.0):
                        why.append("周囲の資源が薄い")
                    if not np.isnan(p["移動率"]) and p["移動率"] > 0.20:
                        why.append("移動に資源を使いやすい")
                    if not np.isnan(p["空腹率"]) and p["空腹率"] > 0.30:
                        why.append("低資源個体を抱えやすい")
                    routes.append("資源面では、直近の資源収支が弱いか負です。" + (" その前段として、" + "、".join(why) + "ことが考えられます。" if why else " 原因は足元資源・採取率・移動支払の分解が必要です。"))

            # 繁殖ルート
            if p["実出生"] > p["死亡"]:
                routes.append(f"繁殖面では、直近の実出生が死亡を上回っています（出生 {p['実出生']:.1f} / 死亡 {p['死亡']:.1f}）。この型は、資源を持つだけでなく子の発生まで届いている可能性があります。")
            elif p["実出生"] < p["死亡"]:
                if p["交尾成功"] > 0 and p["実出生"] <= 0.1:
                    routes.append("繁殖面では、交尾や親参加があっても実出生に届いていない可能性があります。ここでは相手不足ではなく、空きマス・密度依存・近親回避・出生時資源のどこかで止まっているかもしれません。")
                else:
                    routes.append(f"繁殖面では、直近の死亡が実出生を上回っています（出生 {p['実出生']:.1f} / 死亡 {p['死亡']:.1f}）。この型が減るなら、単に弱いというより、コピーを増やす出口に届いていないことが大きいです。")

            # 生存ルート
            if not np.isnan(p["空腹率"]):
                if p["空腹率"] < 0.15:
                    routes.append("生存面では、空腹率が低めです。これは死亡を避ける下地になります。資源獲得が多い場合だけでなく、無駄な移動・戦闘・捕食失敗を避けている場合にも起こります。")
                elif p["空腹率"] > 0.35:
                    routes.append("生存面では、空腹率が高めです。低資源個体が多い型は、少しの維持コストや移動コストで死亡に近づくため、Wが下がりやすくなります。")

            # 遺伝子間作用
            interactions = []
            if not np.isnan(p["タカ比率"]):
                if p["タカ比率"] > mean("タカ比率（0-1）", rf, 0.0) + 0.05:
                    if contest_net > 0:
                        interactions.append("タカ比率が高く、争奪の収支も正に近いため、衝突が資源獲得として働いた可能性があります")
                    else:
                        interactions.append("タカ比率が高い一方で争奪収支が弱いため、攻撃性がコストとして返っている可能性があります")
                elif p["タカ比率"] < mean("タカ比率（0-1）", rf, 0.0) - 0.05:
                    interactions.append("ハト寄りなので、戦闘コストを避けて生存側で残っている可能性があります")
            if not np.isnan(p["捕食傾向"]):
                if p["捕食傾向"] > 0.20:
                    if pred_rate > 0.4:
                        interactions.append("捕食傾向があり、捕食成功率も一定以上なので、捕食が資源獲得経路になっている可能性があります")
                    else:
                        interactions.append("捕食傾向はあるが成功率が弱いので、捕食が危険接触や行動コストになっている可能性があります")
            if interactions:
                routes.append("他の遺伝子との組み合わせでは、" + "。また、".join(interactions) + "。")

            # チーム媒介
            if not np.isnan(p["赤内比率"]) and not np.isnan(p["青内比率"]):
                diff = p["青内比率"] - p["赤内比率"]
                if abs(diff) > 0.05:
                    side = "青" if diff > 0 else "赤"
                    routes.append(f"チーム環境では、この型は{side}側に偏っています。したがって、この型の増減は型そのものだけでなく、{side}側の資源配置・密度・チーム内のタカ/ハト構成に支えられている可能性があります。")

            if not routes:
                routes.append("この型については、比率変化は読めますが、資源・繁殖・死亡のどの経路が主因かはまだ弱いです。世代を進めるか、比較実験で条件をそろえる必要があります。")
            return "\n\n".join([lines[0]] + routes)

        # チーム説明
        red_now, blue_now = last("赤個体数（体）", np.nan), last("青個体数（体）", np.nan)
        red0, blue0 = first("赤個体数（体）", np.nan), first("青個体数（体）", np.nan)
        red_bag, blue_bag = mean("赤 平均所持資源", rf, np.nan), mean("青 平均所持資源", rf, np.nan)
        red_gini, blue_gini = mean("赤 Gini", rf, np.nan), mean("青 Gini", rf, np.nan)
        red_hawk, blue_hawk = mean("赤タカ比率（0-1）", rf, np.nan), mean("青タカ比率（0-1）", rf, np.nan)
        team_paragraphs = []
        if not np.isnan(red_now) and not np.isnan(blue_now):
            side = "赤" if red_now > blue_now else "青" if blue_now > red_now else "赤青ほぼ同数"
            team_paragraphs.append(f"最新世代では、赤は **{int(red_now)}体**、青は **{int(blue_now)}体** です。ここで重要なのは、{side}が多いという結果そのものではなく、その前にある資源・格差・行動の差です。")
            path = []
            if not np.isnan(red_bag) and not np.isnan(blue_bag):
                richer = "赤" if red_bag > blue_bag else "青" if blue_bag > red_bag else "どちらもほぼ同じ"
                path.append(f"平均所持資源は赤 **{red_bag:.2f}**、青 **{blue_bag:.2f}** で、資源保持では **{richer}** が上です。もし多いチームと資源保持が同じ向きなら、そのチームは資源を出生と生存へ変換しやすかったと読めます。逆向きなら、資源以外の死亡回避・繁殖出口・空間配置が効いています。")
            if not np.isnan(red_gini) and not np.isnan(blue_gini):
                lower = "赤" if red_gini < blue_gini else "青" if blue_gini < red_gini else "どちらもほぼ同じ"
                path.append(f"Giniは赤 **{red_gini:.3f}**、青 **{blue_gini:.3f}** です。Giniが低い **{lower}** は、平均資源が同じでも低資源個体を作りにくく、死亡圧を下げやすい側です。")
            if not np.isnan(red_hawk) and not np.isnan(blue_hawk):
                hawk_side = "赤" if red_hawk > blue_hawk else "青" if blue_hawk > red_hawk else "どちらもほぼ同じ"
                if contest_net > 0:
                    path.append(f"タカ比率は赤 **{red_hawk:.3f}**、青 **{blue_hawk:.3f}** で、よりタカ寄りなのは **{hawk_side}** です。直近の争奪収支が正寄りなので、タカ寄りであることは資源獲得として働く可能性があります。")
                else:
                    path.append(f"タカ比率は赤 **{red_hawk:.3f}**、青 **{blue_hawk:.3f}** で、よりタカ寄りなのは **{hawk_side}** です。ただし争奪収支は強くないため、タカ寄りはむしろコストになっている可能性があります。")
            # team-composition bias
            bias_lines = []
            for p in valid_profiles:
                if np.isnan(p["赤内比率"]) or np.isnan(p["青内比率"]):
                    continue
                d = p["青内比率"] - p["赤内比率"]
                if abs(d) > 0.06:
                    bias_lines.append((abs(d), p["型"], "青" if d > 0 else "赤", d))
            if bias_lines:
                bias_lines.sort(reverse=True)
                _, lab, side_bias, d = bias_lines[0]
                path.append(f"チーム内の遺伝子構成では、**{lab}** が **{side_bias}側** に偏っています。この場合、チーム差は色そのものではなく、『そのチームにどの行動型が集まったか』によって作られている可能性があります。")
            team_paragraphs.append("\n\n".join(path) if path else "チーム差を分解する列がまだ不足しています。赤青の数だけでなく、赤/青の平均資源、Gini、タカ比率、哲学型の偏りを追加で見ます。")

        # 環境説明
        env_lines = []
        if W_recent < 0.97:
            env_lines.append(f"直近50世代の平均Wは **{W_recent:.3f}** で、世界全体には縮小圧があります。これは、どれか一つの遺伝子が弱いというより、出生まで届く経路よりも死亡・資源消費・繁殖失敗の経路が強くなっている状態です。")
        elif W_recent > 1.03:
            env_lines.append(f"直近50世代の平均Wは **{W_recent:.3f}** で、世界全体には増殖圧があります。この場合、有利な遺伝子は『死なない遺伝子』だけでなく、『資源を出生へ変換できる遺伝子』です。")
        else:
            env_lines.append(f"直近50世代の平均Wは **{W_recent:.3f}** で、世界はおおむね維持線付近です。この状態では、全体崩壊よりも、型ごとの小さな差が見えやすくなります。")
        if res1 > res0 and bag1 <= bag0:
            env_lines.append(f"資源総量は **{int(res0)}→{int(res1)}** と増えている一方、平均所持資源は **{bag0:.2f}→{bag1:.2f}** で伸びていません。これは、資源が世界に存在することと、個体がそれを使えることが別問題であることを示します。認識半径、移動コスト、足元資源、採取行動のいずれかが、環境資源を体内資源へ変換する経路を弱めています。")
        elif res1 < res0 and bag1 > bag0:
            env_lines.append(f"資源総量は **{int(res0)}→{int(res1)}** と減っていますが、平均所持資源は **{bag0:.2f}→{bag1:.2f}** と増えています。個体が環境ストックを回収できている状態ですが、長く続けば資源枯渇が次の淘汰圧になります。")
        else:
            env_lines.append(f"資源総量は **{int(res0)}→{int(res1)}**、平均所持資源は **{bag0:.2f}→{bag1:.2f}** です。この二つが同じ方向に動くなら、環境資源と個体行動は比較的つながっています。逆方向なら、資源配置と行動の間にずれがあります。")
        env_lines.append(f"直近50世代では、自然発生した資源に対する採取の比率は **{use_ratio:.2f}**、採取から維持・移動・子への分配を引いた簡易収支は **{energy_simple:.1f}** です。これが弱いと、採取だけでは生存と繁殖を支えにくくなり、争奪・捕食・低コスト行動が重要になります。")
        env_lines.append(f"交尾は、試行 **{mate_attempt:.1f}** に対して成立 **{mate_success:.1f}**、成立率 **{mate_rate:.2f}** です。交尾が少ない場合は、資源以前に相手探索や行動選択で止まっています。交尾はあるのに出生が少ない場合は、空きマス・密度依存・近親回避・出生時資源が出口を塞いでいます。")
        if over_birth > 0 or kin_avoid > 0:
            env_lines.append(f"過密による出生抑制は **{over_birth:.1f}/世代**、近親回避は **{kin_avoid:.1f}/世代** です。これは、繁殖意欲や交尾行動だけではコピーは増えず、空間と系統距離が出生の最後の門になっていることを意味します。")
        if contest_events > 0:
            env_lines.append(f"争奪イベントは **{contest_events:.1f}/世代**、争奪の簡易収支は **{contest_net:.1f}** です。収支が正なら攻撃性は資源獲得になり、負なら攻撃性は消耗になります。したがってタカ遺伝子の価値は、タカそのものではなく、争奪が得になる環境かどうかで変わります。")
        if pred_attempt > 0:
            env_lines.append(f"捕食は試行 **{pred_attempt:.1f}**、成功 **{pred_success:.1f}**、成功率 **{pred_rate:.2f}** です。成功率が高ければ捕食傾向は資源獲得の遺伝子として働きますが、低ければ危険接触を増やす遺伝子になります。")

        with st.expander("分析レポート：この世界で何が起きているか", expanded=True):
            st.markdown("### この世界を読むための前提")
            st.markdown(
                "このシミュレーションで見ているのは、個体の勝ち負けそのものではありません。"
                "個体は遺伝子を運ぶ場であり、重要なのは、どの遺伝子がどの環境でコピーを残しやすくなるかです。"
                "そのため、説明は『数が多いから強い』では止めません。資源に触れたのか、資源を持てたのか、交尾できたのか、子が生まれたのか、死ななかったのか、そこまで戻って読みます。"
            )

            st.markdown("### 環境が作っている圧")
            for line in env_lines:
                st.markdown(line)

            if dominant and weakest:
                st.markdown("### いま伸びている型、押されている型")
                st.markdown(
                    f"直近までの頻度変化だけを見ると、最も伸びているのは **{dominant['型']}**、最も押されているのは **{weakest['型']}** です。"
                    "ただし、ここでも頻度は結果です。下では、その型が資源で勝ったのか、死亡を避けたのか、繁殖出口に届いたのか、別の遺伝子やチーム環境に支えられたのかを分けます。"
                )
                st.markdown("#### 伸びている型の読み取り")
                st.markdown(explain_gene(dominant))
                if weakest["型"] != dominant["型"]:
                    st.markdown("#### 押されている型の読み取り")
                    st.markdown(explain_gene(weakest))

            st.markdown("### 各行動型の詳しい読み取り")
            st.caption("ここでは、各型について、結果から一つ前の行動、そのさらに前にある環境との接触まで戻って説明します。")
            for p in valid_profiles:
                with st.expander(p["型"], expanded=False):
                    st.markdown(explain_gene(p))

            if team_paragraphs:
                st.markdown("### 赤チーム・青チームの差はどこから来ているか")
                for para in team_paragraphs:
                    st.markdown(para)

            st.markdown("### この結果から次に確かめたいこと")
            next_checks = []
            if res1 > res0 and bag1 <= bag0:
                next_checks.append("資源が盤面にあるのに個体が使えていない可能性があるため、局所資源再生OFF、移動コスト変更、認識半径変更を比べる。")
            if mate_rate < 0.25 or bd < 0.9:
                next_checks.append("繁殖が出口で止まっている可能性があるため、近親回避OFF、密度依存OFF、初期密度変更を比べる。")
            if contest_events > 0:
                next_checks.append("タカ/ハトの価値が争奪収支に左右されているため、争奪コストCと報酬Vを変えて、タカが本当に有利か確かめる。")
            if pred_attempt > 0:
                next_checks.append("捕食傾向が資源獲得か危険接触かを分けるため、捕食OFFと捕食成功率周辺の指標を見る。")
            if dominant and not np.isnan(dominant.get("青内比率", np.nan)) and abs(dominant.get("青内比率", 0)-dominant.get("赤内比率", 0)) > 0.05:
                next_checks.append(f"{dominant['型']} が片方のチームに偏っているため、同じseedでチーム初期配置を変え、型そのものの効果とチーム環境の効果を分ける。")
            if not next_checks:
                next_checks.append("現在のランは極端な単一原因より、資源・繁殖・死亡が同時に効いている可能性があります。seedを変えて同じ傾向が残るか確認します。")
            for item in next_checks:
                st.markdown("- " + item)

            st.caption("ここでの説明は、単独ランから読める最も筋の通る解釈です。最終的な因果主張にするには、比較実験モードで条件を一つずつ変え、同じ傾向が残るか確認します。")

        with st.expander("分析に使った主要な数値", expanded=False):
            rows = []
            for p in valid_profiles:
                rows.append({
                    "型": p["型"],
                    "初期比率": p["初期比率"],
                    "最新比率": p["最新比率"],
                    "比率変化": p["比率変化"],
                    "平均W": p["平均W"],
                    "資源収支": p["資源収支"],
                    "平均所持資源": p["平均所持資源"],
                    "足元資源": p["足元資源"],
                    "空腹率": p["空腹率"],
                    "実出生": p["実出生"],
                    "死亡": p["死亡"],
                    "採取率": p["採取率"],
                    "交尾率": p["交尾率"],
                    "回避率": p["回避率"],
                    "タカ比率": p["タカ比率"],
                    "捕食傾向": p["捕食傾向"],
                    "赤内比率": p["赤内比率"],
                    "青内比率": p["青内比率"],
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).style.format({
                    "初期比率": "{:.3f}", "最新比率": "{:.3f}", "比率変化": "{:+.3f}",
                    "平均W": "{:.3f}", "資源収支": "{:.2f}", "平均所持資源": "{:.2f}",
                    "足元資源": "{:.2f}", "空腹率": "{:.3f}", "実出生": "{:.2f}", "死亡": "{:.2f}",
                    "採取率": "{:.3f}", "交尾率": "{:.3f}", "回避率": "{:.3f}",
                    "タカ比率": "{:.3f}", "捕食傾向": "{:.3f}", "赤内比率": "{:.3f}", "青内比率": "{:.3f}",
                }), width="stretch", hide_index=True)


    # -------------------------
    # グラフ表示（単位ごとに分けて“読みやすく”）
    # -------------------------
    xcol = "世代（回）"

    # -------------------------
    # 0) 研究ダッシュボード：まずここを見れば状態が分かる
    # -------------------------
    st.markdown("### 0) 研究ダッシュボード")
    explain_box(
        "この画面の目的",
        "個体の強さではなく、遺伝子コピーがどの環境圧で増減したかを見るための読み取り画面です。まず W・遺伝子頻度・出生/死亡・資源収支を見て、自然淘汰が成立しているかを確認します。"
    )
    show_reading_guide()

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

    show_public_story_report()
    show_philo_summary_table()
    if st.checkbox("補助的な詳細分析を表示", value=False):
        show_auto_interpretation()
        show_whole_run_summary()
        show_deep_causal_interpretation()
        show_philosophy_gene_flow_summary()
        show_v19_lineage_flow_summary()
        show_v20_comparison_mode()
    else:
        st.caption("細かい検算用の表や比較実験は、必要なときだけ『補助的な詳細分析を表示』から開けます。")

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
    st.dataframe(dff, width="stretch")