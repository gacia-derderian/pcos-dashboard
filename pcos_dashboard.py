import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# COLOR SYSTEM — LIGHT THEME, DARK GREY TEXT
# ─────────────────────────────────────────────
PCOS_COLOR = "#C2185B"
CTRL_COLOR = "#1565C0"
BG         = "#F4F5F8"
CARD_BG    = "#FFFFFF"
BORDER     = "#E3E6EE"
TEXT_MAIN  = "#1E2233"
TEXT_SUB   = "#4A5168"
TEXT_MUTED = "#7B8299"
ACCENT     = "#C2185B"
GRID       = "#EAEDF4"
WARN       = "#B5780A"

st.set_page_config(page_title="PCOS Symptom Profiling Dashboard",
                   page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{BG};color:{TEXT_MAIN};}}
.main{{background:{BG};}}
.block-container{{padding:1.2rem 1.6rem 1.6rem 1.6rem;max-width:1700px;}}
section[data-testid="stSidebar"]{{display:none;}}

.card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:1.1rem 1.3rem;
       margin-bottom:0.7rem;box-shadow:0 1px 3px rgba(20,24,40,0.04);}}
.kpi{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:0.9rem 1.1rem;
      text-align:center;box-shadow:0 1px 3px rgba(20,24,40,0.04);}}
.kv{{font-size:1.85rem;font-weight:800;color:{ACCENT};line-height:1.1;}}
.kl{{font-size:0.72rem;color:{TEXT_SUB};margin-top:4px;text-transform:uppercase;letter-spacing:.05em;font-weight:700;}}
.ks{{font-size:0.78rem;color:{TEXT_MUTED};margin-top:2px;}}

.sh{{font-size:1.05rem;font-weight:800;color:{TEXT_MAIN};border-left:4px solid {ACCENT};
     padding-left:.7rem;margin:1rem 0 .6rem 0;background:rgba(194,24,91,0.05);
     padding-top:.45rem;padding-bottom:.45rem;border-radius:0 8px 8px 0;}}
.shsub{{font-size:0.78rem;font-weight:500;color:{TEXT_SUB};margin-left:.7rem;margin-top:1px;margin-bottom:.5rem;}}

.div{{border:none;border-top:1px solid {BORDER};margin:0.9rem 0;}}

.badge{{display:inline-block;font-size:.72rem;font-weight:700;padding:.16rem .6rem;border-radius:20px;margin-right:4px;}}
.bp{{background:rgba(194,24,91,.10);color:#9B1349;border:1px solid rgba(194,24,91,.3);}}
.bn{{background:rgba(21,101,192,.10);color:#0D3F87;border:1px solid rgba(21,101,192,.3);}}

.rl{{font-size:.8rem;font-weight:700;text-align:center;padding:.25rem .8rem;border-radius:20px;display:inline-block;margin-top:5px;}}
.rlo{{background:#E3F3E5;color:#1B5E20;}}
.rlm{{background:#FDF0DC;color:#8C4A00;}}
.rlh{{background:#FBE0E0;color:#7F1010;}}

h1,h2,h3{{color:{TEXT_MAIN}!important;}}

.filter-bar{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;
     padding:0.8rem 1.3rem 0.2rem 1.3rem;margin-bottom:0.8rem;box-shadow:0 1px 3px rgba(20,24,40,0.04);}}
.filter-title{{font-size:0.82rem;font-weight:800;color:{TEXT_MAIN};margin-bottom:0.4rem;}}
.filter-count{{font-size:0.76rem;color:{ACCENT};font-weight:700;}}

label, .stCheckbox label p, .stSelectbox label p, .stSlider label p {{color:{TEXT_SUB} !important;font-weight:700 !important;}}
div[data-testid="stMarkdownContainer"] p {{color:{TEXT_SUB};}}
.story-tag{{display:inline-block;background:{TEXT_MAIN};color:#FFFFFF;font-size:0.68rem;font-weight:700;
     padding:0.12rem 0.5rem;border-radius:6px;margin-bottom:0.4rem;letter-spacing:0.03em;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PASSWORD GATE
# ─────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown(f"""
    <div style="max-width:420px;margin:8vh auto;text-align:center;
         background:{CARD_BG};border:1px solid {BORDER};border-radius:20px;padding:3rem 2.5rem;
         box-shadow:0 4px 16px rgba(20,24,40,0.08);">
      <div style="font-size:2.8rem;margin-bottom:.5rem;">🩺</div>
      <div style="font-size:1.6rem;font-weight:800;color:{TEXT_MAIN};margin-bottom:.4rem;">PCOS Symptom Profiling</div>
      <div style="font-size:.9rem;color:{TEXT_SUB};margin-bottom:2rem;">Healthcare Analytics Dashboard · MSBA382 · OSB</div>
    </div>""", unsafe_allow_html=True)
    _,c,_ = st.columns([1,1.4,1])
    with c:
        pw = st.text_input("pw", type="password", label_visibility="collapsed", placeholder="🔒  Enter password")
        if st.button("Access Dashboard", use_container_width=True):
            if pw == "pcos2026":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.markdown(f"<div style='text-align:center;color:{TEXT_SUB};font-size:.78rem;margin-top:1rem;'>Kerala Hospital Cohort · n=541 · Kottarathil (2020)</div>", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "C:/Users/gacia.derderian/Desktop/AUB/MSBA382 - HEALTHCARE - SAMAR HAJJ/Individual Project Presentation/PCOS Dataset/PCOS DATA/"
    df_main = pd.read_excel(path + "PCOS_data_without_infertility.xlsx", sheet_name="Full_new")
    df_inf  = pd.read_csv(path + "PCOS_infertility.csv")
    df_main.columns = df_main.columns.str.strip()
    df_inf.columns  = df_inf.columns.str.strip()
    df = pd.merge(df_main,
                  df_inf[['Patient File No.','AMH(ng/mL)']].rename(columns={'AMH(ng/mL)':'AMH_inf'}),
                  on='Patient File No.', how='left')
    df['AMH(ng/mL)'] = df['AMH(ng/mL)'].combine_first(df['AMH_inf'])
    drop_cols = [c for c in ['AMH_inf','Sl. No','Patient File No.','Unnamed: 44'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df.columns = df.columns.str.strip()
    df['Cycle(R/I)']             = df['Cycle(R/I)'].map({2:0,4:1}).fillna(0)
    df['Fast food (Y/N)']        = df['Fast food (Y/N)'].fillna(1.0)
    df['Marraige Status (Yrs)']  = df['Marraige Status (Yrs)'].fillna(df['Marraige Status (Yrs)'].median())
    df['AMH(ng/mL)']             = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
    df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
    df['BMI']                    = pd.to_numeric(df['BMI'], errors='coerce')
    df['TSH (mIU/L)']            = pd.to_numeric(df['TSH (mIU/L)'], errors='coerce')
    SC = ['Weight gain(Y/N)','hair growth(Y/N)','Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)','Cycle(R/I)']
    df['symptom_score'] = df[SC].sum(axis=1)
    return df

@st.cache_resource
def train_model(_df):
    SC = ['Weight gain(Y/N)','hair growth(Y/N)','Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)','Cycle(R/I)']
    X = _df[SC].astype(float)
    y = _df['PCOS (Y/N)'].astype(int)
    Xt,Xe,yt,ye = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    m = LogisticRegression(class_weight='balanced',random_state=42,max_iter=1000)
    m.fit(Xt,yt)
    return m

df_full = load_data()
mdl     = train_model(df_full)

SC = ['Weight gain(Y/N)','hair growth(Y/N)','Skin darkening (Y/N)','Hair loss(Y/N)','Pimples(Y/N)','Cycle(R/I)']
SL = ['Weight Gain','Hair Growth','Skin Darkening','Hair Loss','Pimples/Acne','Irregular Cycle']
SL3 = ['Wt Gain','Hirsutism','Skin Darkening','Hair Loss','Pimples','Irregular Cycle']

def fig_layout(fig, title="", height=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT_SUB, size=11),
        title=dict(text=title, font=dict(size=13, color=TEXT_MAIN, family="Inter"), x=0),
        margin=dict(l=10,r=10,t=38,b=10), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT_SUB),
                    orientation="h", y=1.15, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", zerolinecolor=GRID, showline=False,
                   tickfont=dict(color=TEXT_SUB, size=10), title_font=dict(color=TEXT_SUB, size=11)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=False,
                   tickfont=dict(color=TEXT_SUB, size=10), title_font=dict(color=TEXT_SUB, size=11))
    )

def section_header(tag, title, subtitle):
    st.markdown(f'<span class="story-tag">{tag}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="shsub">{subtitle}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
header_html = (
    '<div style="display:flex;align-items:center;justify-content:space-between;'
    f'background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;'
    'padding:1rem 1.5rem;margin-bottom:0.8rem;box-shadow:0 1px 3px rgba(20,24,40,0.04);">'
    '<div>'
    f'<div style="font-size:1.4rem;font-weight:800;color:{TEXT_MAIN};">🩺 PCOS Symptom Profiling Dashboard</div>'
    f'<div style="font-size:0.82rem;color:{TEXT_SUB};margin-top:2px;font-weight:500;">Can visible symptoms alone identify PCOS? &middot; Kerala Hospital Cohort &middot; n=541</div>'
    '</div>'
    '<div style="text-align:right;">'
    '<span class="badge bp">PCOS+ n=177</span>'
    '<span class="badge bn">PCOS&minus; n=364</span>'
    f'<div style="font-size:0.72rem;color:{TEXT_SUB};margin-top:5px;font-weight:600;">MSBA382 &middot; OSB &middot; 2026</div>'
    '</div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER BAR  (single level — fine)
# ─────────────────────────────────────────────
st.markdown(f'<div class="filter-bar">', unsafe_allow_html=True)
st.markdown(f'<div class="filter-title">🎛️ Filters — adjust to cross-filter every chart below</div>', unsafe_allow_html=True)

f1,f2,f3,f4,f5,f6 = st.columns(6)
with f1: age_r  = st.slider("Age (years)", 20, 48, (20,48))
with f2: wt_r   = st.slider("Weight (Kg)", 31, 108, (31,108))
with f3: ht_r   = st.slider("Height (Cm)", 137, 180, (137,180))
with f4: bmi_r  = st.slider("BMI", 12.0, 39.0, (12.0,39.0))
with f5: preg_opt = st.selectbox("Pregnant (Y/N)", ["All","Yes","No"])
with f6: tsh_r  = st.slider("TSH (mIU/L)", 0.0, 65.0, (0.0,65.0))

mask = (
    df_full['Age (yrs)'].between(age_r[0], age_r[1]) &
    df_full['Weight (Kg)'].between(wt_r[0], wt_r[1]) &
    df_full['Height(Cm)'].between(ht_r[0], ht_r[1]) &
    df_full['BMI'].between(bmi_r[0], bmi_r[1]) &
    df_full['TSH (mIU/L)'].between(tsh_r[0], tsh_r[1])
)
if preg_opt == "Yes": mask &= df_full['Pregnant(Y/N)'] == 1
elif preg_opt == "No": mask &= df_full['Pregnant(Y/N)'] == 0

df = df_full[mask].copy()
n_filtered = len(df); n_total = len(df_full)
pct_kept = round(n_filtered/n_total*100,1) if n_total else 0

st.markdown(
    f'<div class="filter-count">Showing {n_filtered} of {n_total} patients ({pct_kept}%) '
    f'&middot; PCOS+ {len(df[df["PCOS (Y/N)"]==1])} &middot; PCOS&minus; {len(df[df["PCOS (Y/N)"]==0])}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if n_filtered < 10:
    st.warning("⚠️ Fewer than 10 patients match the current filters. Widen the ranges for meaningful statistics.")
    st.stop()

pos = df[df['PCOS (Y/N)']==1]
neg = df[df['PCOS (Y/N)']==0]
has_pos = len(pos) > 0
has_neg = len(neg) > 0

# ─────────────────────────────────────────────
# KPI ROW  (single level — fine)
# ─────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
def kpi(col,val,lbl,sub=""):
    col.markdown(f'<div class="kpi"><div class="kv">{val}</div><div class="kl">{lbl}</div><div class="ks">{sub}</div></div>',
                 unsafe_allow_html=True)

pcos_rate = round(len(pos)/n_filtered*100,1) if n_filtered else 0
avg_score = round(df['symptom_score'].mean(),2) if n_filtered else 0
high_risk_n = len(df[df['symptom_score']>=4])
high_risk_rate = round(len(df[(df['symptom_score']>=4)&(df['PCOS (Y/N)']==1)])/high_risk_n*100,1) if high_risk_n else 0

kpi(k1, f"{n_filtered}", "Patients", f"{pct_kept}% of cohort")
kpi(k2, f"{pcos_rate}%", "PCOS Prevalence", f"{len(pos)} of {n_filtered}")
kpi(k3, f"{avg_score}", "Avg Symptom Score", "out of 6")
kpi(k4, f"{high_risk_rate}%", "PCOS Rate (Score ≥4)", f"n={high_risk_n}")
kpi(k5, "0.889", "Model AUC-ROC", "symptom-only")

st.markdown("<div class='div'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TWO-COLUMN GRID — exactly ONE level of nesting throughout
# LEFT:  Step 1 (prevalence) + Step 3 (triads)
# RIGHT: Step 2 (co-occurrence + burden) + Step 4 (risk screener)
# ═══════════════════════════════════════════════════════════════
colL, colR = st.columns([1.05, 1], gap="medium")

# ───────────── LEFT COLUMN ─────────────
with colL:

    # STEP 1 — THE SIGNAL: symptom prevalence
    section_header("STEP 1 · THE SIGNAL", "Symptom Prevalence by PCOS Status",
                   "Every observable symptom shows a statistically significant gap between groups.")

    sd = {}
    for c,l in zip(SC,SL):
        pv_ = round(pos[c].mean()*100,1) if has_pos else 0
        nv_ = round(neg[c].mean()*100,1) if has_neg else 0
        if has_pos and has_neg:
            chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(df['PCOS (Y/N)'],df[c]))
        else:
            p = np.nan
        sd[l] = {'pos':pv_,'neg':nv_,'p':p}
    ss = sorted(sd.items(), key=lambda x: x[1]['pos']-x[1]['neg'], reverse=True)
    ls=[s[0] for s in ss]; pv=[s[1]['pos'] for s in ss]; nv=[s[1]['neg'] for s in ss]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='PCOS+',x=ls,y=pv,marker_color=PCOS_COLOR,marker_line_width=0,
        text=[f"{v}%" for v in pv],textposition='outside',textfont=dict(size=10,color=TEXT_MAIN)))
    fig1.add_trace(go.Bar(name='PCOS−',x=ls,y=nv,marker_color=CTRL_COLOR,marker_line_width=0,
        text=[f"{v}%" for v in nv],textposition='outside',textfont=dict(size=10,color=TEXT_MAIN)))
    fig_layout(fig1,"Symptom Prevalence (%) by Diagnosis Status",330)
    fig1.update_layout(barmode='group',bargap=0.25,bargroupgap=0.08,
        yaxis=dict(title="Prevalence (%)",range=[0,108],gridcolor=GRID,tickfont=dict(color=TEXT_SUB)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT_SUB, size=9)))
    for lbl,d in ss:
        if not np.isnan(d['p']):
            pt = "p<0.001" if d['p']<0.001 else f"p={d['p']:.3f}"
            fig1.add_annotation(x=lbl,y=max(d['pos'],d['neg'])+9,text=f"★ {pt}",
                showarrow=False,font=dict(size=8,color=WARN),xanchor="center")
    st.plotly_chart(fig1,use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # STEP 3 — SO WHAT: top symptom triads (the headline finding)
    section_header("STEP 3 · SO WHAT", "Top 3-Symptom Triads",
                   "The most clinically actionable, high-specificity symptom combinations.")

    triad_results = []
    if has_pos and has_neg:
        for i in range(len(SC)):
            for j in range(i+1,len(SC)):
                for k in range(j+1,len(SC)):
                    c1_,c2_,c3_ = SC[i],SC[j],SC[k]
                    pp = round(((pos[c1_]==1)&(pos[c2_]==1)&(pos[c3_]==1)).mean()*100,1)
                    pn = round(((neg[c1_]==1)&(neg[c2_]==1)&(neg[c3_]==1)).mean()*100,1)
                    ratio = round(pp/pn,1) if pn>0 else (999 if pp>0 else 0)
                    triad_results.append((f"{SL3[i]} + {SL3[j]} + {SL3[k]}", pp, pn, ratio))
        triad_results.sort(key=lambda x: x[1], reverse=True)
        top5 = triad_results[:5]
    else:
        top5 = []

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if top5:
        max_pp = max(t[1] for t in top5) or 1
        for tri,pp,pn,ratio in top5:
            bw=int(pp/max(max_pp,1)*100)
            ratio_txt = f"{ratio}×" if ratio < 999 else "∞"
            st.markdown(f"""<div style="margin-bottom:.8rem;">
              <div style="font-size:.78rem;font-weight:700;color:{TEXT_MAIN};margin-bottom:3px;">{tri}</div>
              <div style="display:flex;align-items:center;gap:8px;">
                <div style="flex:1;background:{BORDER};border-radius:4px;height:8px;overflow:hidden;">
                  <div style="width:{bw}%;background:{PCOS_COLOR};height:100%;border-radius:4px;"></div>
                </div>
                <div style="font-size:.74rem;color:{TEXT_SUB};white-space:nowrap;font-weight:500;">
                  <span style="color:#9B1349;font-weight:700;">{pp}%</span> PCOS+ ·
                  <span style="color:#0D3F87;font-weight:700;">{pn}%</span> PCOS− ·
                  <span style="color:{WARN};font-weight:800;">{ratio_txt}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:{TEXT_SUB};font-size:0.82rem;">Not enough PCOS+ and PCOS− patients in this selection to compute triads.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model performance summary card under triads
    st.markdown(f"""<div class="card">
      <div style="font-size:.85rem;font-weight:800;color:{TEXT_MAIN};margin-bottom:.6rem;">📊 Model Performance (Full Cohort)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;text-align:center;">
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">83.5%</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">Accuracy</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">77.8%</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">Sensitivity</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">86.3%</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">Specificity</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">0.889</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">AUC-ROC</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">73.7%</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">Precision</div></div>
        <div><div style="font-size:1rem;font-weight:800;color:{ACCENT};">75.7%</div><div style="font-size:.68rem;color:{TEXT_SUB};font-weight:600;">F1-Score</div></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ───────────── RIGHT COLUMN ─────────────
with colR:

    # STEP 2 — THE PATTERN: co-occurrence heatmap (tabs, not nested columns)
    section_header("STEP 2 · THE PATTERN", "Symptom Co-occurrence",
                   "Symptoms cluster together rather than appearing in isolation.")

    t1,t2 = st.tabs(["🔴 PCOS+ Co-occurrence","🔵 PCOS− Co-occurrence"])
    def heatmap(subdf,title,color):
        n=len(SC); mat=np.zeros((n,n))
        if len(subdf)>0:
            for i in range(n):
                for j in range(n):
                    mat[i,j]=round(((subdf[SC[i]]==1)&(subdf[SC[j]]==1)).mean()*100,1)
        m2=np.where(np.tril(np.ones((n,n),dtype=bool)),mat,np.nan)
        r,g,b=int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig=go.Figure(go.Heatmap(z=m2,x=SL,y=SL,
            colorscale=[[0,"rgba(0,0,0,0)"],[0.01,f"rgba({r},{g},{b},0.12)"],[1,color]],
            showscale=True,
            text=[[f"{m2[i,j]:.0f}%" if not np.isnan(m2[i,j]) else "" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",textfont=dict(size=10,color="white"),
            hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>Co-occurrence: %{z:.1f}%<extra></extra>",
            colorbar=dict(tickfont=dict(color=TEXT_SUB),thickness=9)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter",color=TEXT_SUB,size=11),height=300,
            title=dict(text=title,font=dict(size=12,color=TEXT_MAIN),x=0),margin=dict(l=10,r=10,t=38,b=10),
            xaxis=dict(tickangle=-30,gridcolor="rgba(0,0,0,0)",tickfont=dict(color=TEXT_SUB,size=9)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",autorange="reversed",tickfont=dict(color=TEXT_SUB,size=9)))
        return fig
    with t1: st.plotly_chart(heatmap(pos,f"Co-occurrence in PCOS+ (%) — n={len(pos)}",PCOS_COLOR),use_container_width=True)
    with t2: st.plotly_chart(heatmap(neg,f"Co-occurrence in PCOS− (%) — n={len(neg)}",CTRL_COLOR),use_container_width=True)

    # Burden score chart directly below (full width of right column — no extra nesting)
    sdf=df.groupby('symptom_score')['PCOS (Y/N)'].agg(['mean','count']).reset_index()
    sdf['pct']=(sdf['mean']*100).round(1)
    sdf['sl']=sdf['symptom_score'].astype(int).astype(str)
    fig3=make_subplots(specs=[[{"secondary_y":True}]])
    fig3.add_trace(go.Bar(x=sdf['sl'],y=sdf['count'],name='N Patients',
        marker_color=BORDER,marker_line_width=0,opacity=0.9,
        hovertemplate="Score %{x}: %{y} patients<extra></extra>"),secondary_y=True)
    fig3.add_trace(go.Scatter(x=sdf['sl'],y=sdf['pct'],name='PCOS Rate (%)',mode='lines+markers',
        line=dict(color=PCOS_COLOR,width=3),
        marker=dict(size=9,color=sdf['pct'],
            colorscale=[[0,CTRL_COLOR],[0.5,WARN],[1,PCOS_COLOR]],
            line=dict(color="white",width=2)),
        hovertemplate="Score %{x}: PCOS rate = %{y:.1f}%<extra></extra>"),secondary_y=False)
    fig3.add_vline(x=3.5,line_dash="dash",line_color=WARN,line_width=1.5,
        annotation_text="≥80% threshold",annotation_position="top left",
        annotation_font=dict(size=9,color=WARN))
    fig3.add_hline(y=50,line_dash="dot",line_color=TEXT_MUTED,line_width=1)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=290,
        font=dict(family="Inter",color=TEXT_SUB,size=11),
        title=dict(text="PCOS Rate by Symptom Burden Score (0–6)",font=dict(size=12,color=TEXT_MAIN),x=0),
        margin=dict(l=10,r=10,t=38,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=TEXT_SUB),orientation="h",y=1.15,x=0.5,xanchor="center"),
        xaxis=dict(title="Symptom Burden Score",gridcolor="rgba(0,0,0,0)",tickfont=dict(color=TEXT_SUB)))
    fig3.update_yaxes(title_text="PCOS Rate (%)",range=[0,108],gridcolor=GRID,secondary_y=False,
                       tickfont=dict(color=TEXT_SUB))
    fig3.update_yaxes(title_text="N Patients",gridcolor="rgba(0,0,0,0)",secondary_y=True,
                       tickfont=dict(color=TEXT_SUB))
    st.plotly_chart(fig3,use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # STEP 4 — THE PROOF: interactive risk screener (gauge only, no inner columns)
    section_header("STEP 4 · THE PROOF", "Interactive PCOS Risk Screener",
                   "A symptom-only model (AUC-ROC = 0.889) estimates PCOS probability live.")

    st.markdown(f"""<div class="card" style="margin-bottom:.5rem;">
      <div style="font-size:.8rem;color:{TEXT_SUB};">Select the symptoms present in the patient below.</div>
    </div>""", unsafe_allow_html=True)

    wg = st.checkbox("⚖️  Weight Gain")
    hg = st.checkbox("💇  Hirsutism (Hair Growth)")
    sd2 = st.checkbox("🌑  Skin Darkening")
    hl = st.checkbox("🪮  Hair Loss")
    pm = st.checkbox("😣  Pimples / Acne")
    ci = st.checkbox("📅  Irregular Menstrual Cycle")

    feats=np.array([[int(wg),int(hg),int(sd2),int(hl),int(pm),int(ci)]])
    prob=mdl.predict_proba(feats)[0][1]
    scr=feats.sum(); pct=round(prob*100,1)
    if prob<0.35:   rc,rt="rlo","LOW RISK"
    elif prob<0.65: rc,rt="rlm","MODERATE RISK"
    else:           rc,rt="rlh","HIGH RISK"

    fig8=go.Figure(go.Indicator(mode="gauge+number",value=pct,
        number=dict(suffix="%",font=dict(size=32,color=TEXT_MAIN)),
        gauge=dict(axis=dict(range=[0,100],tickcolor=TEXT_SUB,tickfont=dict(color=TEXT_SUB)),
            bar=dict(color=PCOS_COLOR if prob>=0.65 else WARN if prob>=0.35 else CTRL_COLOR,thickness=0.7),
            bgcolor="rgba(0,0,0,0)",
            steps=[dict(range=[0,35],color="rgba(21,101,192,0.08)"),
                   dict(range=[35,65],color="rgba(181,120,10,0.08)"),
                   dict(range=[65,100],color="rgba(194,24,91,0.08)")],
            threshold=dict(line=dict(color=TEXT_MAIN,width=2),thickness=0.8,value=pct))))
    fig8.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter",color=TEXT_MAIN),
        height=190,margin=dict(l=20,r=20,t=10,b=10))
    st.plotly_chart(fig8,use_container_width=True)
    st.markdown(f"""<div style="text-align:center;margin-top:-.5rem;">
      <span class="rl {rc}">{rt}</span>
      <div style="font-size:.78rem;color:{TEXT_SUB};margin-top:5px;">
        Burden score: <b style="color:{TEXT_MAIN};">{int(scr)}/6</b> ·
        Probability: <b style="color:{TEXT_MAIN};">{pct}%</b>
      </div>
      <div style="font-size:.72rem;color:{TEXT_MUTED};margin-top:3px;">⚠️ Screening tool only · Not a clinical diagnosis</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""<div style="text-align:center;color:{TEXT_SUB};font-size:.74rem;
     margin-top:1.2rem;padding:0.8rem;border-top:1px solid {BORDER};font-weight:500;">
  PCOS Symptom Profiling Dashboard · MSBA382 Healthcare Analytics · OSB 2026 ·
  Data: Kottarathil (2020), Kerala Hospital Cohort, n=541 ·
  ⚠️ For educational and research purposes only
</div>""", unsafe_allow_html=True)
