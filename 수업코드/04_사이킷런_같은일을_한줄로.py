# =====================================================================
#  04. 사이킷런 — 01~03 에서 손으로 한 일을 '한 줄'로, 그리고 언제 쓰나
# =====================================================================
# 사이킷런(scikit-learn, 코드에선 sklearn) = 파이썬 머신러닝 도구 상장
# 우리가 01~03에서 손으로 짠 것(표준화 경사하강 회귀 분류 채점)이 전부 함수로 들어 있다.
# 왜 손으로 먼저 했나? 도구가 '안에서 무슨 일을 하는지' 알아야 결과를 읽고, 이상할 때 고칠 수 있어서.
# 언제 쓰나? 표(엑셀 같은)데이터 + 선형모델 트리 같은 전통 머신러닝. 실무 첫 선택은 거의 이것

# "손으로 한 값 == 사이킷러 값"을 눈으로 확인하고, 규칙 3개를 몸에 익힌다.
# 규칙1) X는 (설비 수, 센서 수) 표 모양, y는 한줄
# 규칙2) 만들기 -> fit(학습) -> predict(예측) 세 동사
# 규칙3) 학습 결과는 이름 끝에 밑줄: coef_, intercept_

import os
import numpy as np
import pandas as pd

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "수업용데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
feature_name = ["공기온도", "회전수", "토크", "공구마모"]

# =====================================================================
# 0. 사이킷런에서 꺼내 쓰는 것들 — 미리 한눈에
# =====================================================================
# ★ 문법 ★ from sklearn.어느상자 import 도구
#           사이킷런은 큰 도구함이라 통째로 부르지 않고, 필요한 서랍에서 필요한 것만 꺼냅니다.
#           서랍 이름(sklearn 뒤에 오는 것)만 알면 어디서 뭘 꺼낼지 감이 옵니다.
#
#   서랍 이름                뜻              여기서 꺼내는 것
#   ---------------------   -------------   --------------------------------------------
#   sklearn.linear_model    직선 계열 모델   LinearRegression, LogisticRegression, Ridge
#   sklearn.tree            트리 하나        DecisionTreeClassifier/Regressor, export_text   (07)
#   sklearn.ensemble        트리 여러 개     RandomForest..., HistGradientBoosting...        (07)
#   sklearn.neighbors       가까운 이웃      KNeighborsRegressor                             (§3-1)
#   sklearn.preprocessing   전처리          StandardScaler, PolynomialFeatures
#   sklearn.model_selection 나누기/고르기    train_test_split, cross_val_score, GridSearchCV
#   sklearn.metrics         채점            confusion_matrix, classification_report, recall/precision/f1_score
#   sklearn.pipeline        이어 붙이기      make_pipeline
#
# 하나씩 무슨 일을 하는지 (여기서 다 보고 가면 뒤가 편합니다)
#
#  [모델] — 만들고(생성) → fit(학습) → predict(예측) 세 동사는 전부 똑같습니다
#   LinearRegression()        숫자를 맞히는 직선 모델. 01·02 에서 손으로 짠 그것 (회귀)
#   LogisticRegression()      종류를 맞히는 모델. 03 의 sigmoid + 로그손실이 이 안에 (분류)
#                             기본으로 규제가 켜져 있고, max_iter 는 "몇 걸음까지 걸을까"
#   Ridge(alpha=1.0)          가중치가 커지면 벌점을 주는 선형회귀. alpha 가 벌점 세기 (과적합 처방)
#   KNeighborsRegressor(1)    "제일 비슷한 것 하나 찾아 그 답을 그대로 말하기". §3-1 누수 시연용
#
#  [전처리] — fit 으로 기준을 외우고, transform 으로 값을 바꿉니다
#   StandardScaler()          02 의 표준화. fit 하면 학습용 평균·표준편차를 외우고,
#                             transform 하면 그 자로 값을 바꿉니다. ★ fit 은 학습용에만 ★
#   PolynomialFeatures(4)     특징끼리 곱하고 제곱해 개수를 불립니다. §6-1 에서 과적합 만들 때만 사용
#
#  [나누기 · 고르기]
#   train_test_split(...)     02 의 "섞고 70:30 자르기" 한 줄. random_state 로 매번 같게,
#                             stratify=y 로 고장 비율을 양쪽에 똑같이 (03 에서 손으로 한 일)
#   cross_val_score(...)      학습용을 cv 조각으로 나눠 번갈아 채점하고 점수 목록을 줍니다.
#                             test 는 건드리지 않습니다 (§7)
#   GridSearchCV(...)         후보값을 전부 교차검증으로 돌려 제일 좋은 걸 고르고 다시 학습까지 (§7)
#
#  [채점]
#   confusion_matrix(y, 예측)      네 칸 표. 사이킷런 순서는 [[TN, FP], [FN, TP]] 로 03 과 자리가 다릅니다
#   classification_report(...)     정밀도·재현율·f1 을 한 표로. 불균형이면 정확도 대신 이걸 봅니다
#   recall_score(y, 예측)          재현율 = 실제 고장 중 몇 개를 잡았나 (놓치면 안 될 때)
#   precision_score(y, 예측)       정밀도 = 고장이라 외친 것 중 몇 개가 진짜였나 (헛경보가 곤란할 때)
#   f1_score(y, 예측)              재현율과 정밀도를 하나로 합친 점수 (07 에서 모델 비교할 때)
#
#  [이어 붙이기]
#   make_pipeline(A, B)       A 를 거쳐 B 로 가는 한 덩어리 모델. fit 하면 A 는 학습용으로만
#                             기준을 잡고, predict 하면 변환만 합니다 → 누수를 구조적으로 차단 (§3)
#
# 공통 규칙 세 가지 (이것만 외우면 나머지는 문서 보고 씁니다)
#   1) X 는 (행 수, 열 수) 2차원 표, y 는 1차원 한 줄
#   2) 만들기 → .fit(X, y) → .predict(X). 전처리 도구는 .fit → .transform
#   3) 학습으로 알아낸 값은 이름 끝에 밑줄: .coef_, .intercept_, .mean_, .feature_importances_

# =====================================================================
# 1. 회귀 — 01 의 "공기온도 → 공정온도" 를 세 줄로
# =====================================================================
from sklearn.linear_model import LinearRegression

x = df["공기온도"].values
y = df["공정온도"].values
X1 = x.reshape(-1, 1)
# 규칙 1: 센서가 하나여도 (200, 1) '세로 표'로 세운다.
# reshape(-1, 1)의 -1 = "행 수는 알아서 맞춰"

print("[1] x.shape", x.shape, "→ X1.shape", X1.shape)
# 왜 사이킷런은 "행 = 설비, 열 = 센서" 표만 받는다. 센서 1개면 열이 1개인 표. 한 줄짜리 배열은 안 받음

# 규칙 2: 만들고 학습
model = LinearRegression()  # 만들기 - 빈 선형회귀 모델. 아직 아무것도 모름
model.fit(X1, y)  # 학습 - 01에서 300걸음 걸은 그 일이 이 한줄

# 규칙 3: 학습해서 얻은 값은 끝에 밑줄(_). coef_ = w 들, intercept_ = b
print(f"    w = {model.coef_[0]:.4f}  b = {model.intercept_:.4f}")
# ③ 예측. 새 데이터도 '표 모양'이라 대괄호 두 겹 [[ ]]
print("    공기온도 300 → 예측:", round(model.predict([[300.0]])[0], 2))

# .score = 회귀면 R², 분류면 정확도
# 사이킷런 LinearRegression은 걷지 않고 '공식'으로 단번에 푼다. 그래서 lr epoch가 없음
# 선형회귀(와 그 사촌 Ridge)는 공식이 있고, 로지스틱 회귀 부터는 사이킷런도 속으로 걷는다.
# (max_iter가 그 흔적)
print("    R²:", round(model.score(X1, y), 4), "(01 의 0.7989)")

# 가장 흔한 에러 미리 보기 X를 한 줄(1차원)로 넣으면
try:
    LinearRegression().fit(x, y)
except ValueError as e:
    print("[에러]", str(e).splitlines()[0], "-> reshape(-1, 1) 하라는 뜻")

# =====================================================================
# 2. 나누기 + 표준화 + 다변수 회귀 — 02 를 도구로
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[feature_name].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 02의 "섞고 70:30으로 자르기"가 이 한줄. 돌려주는 순서 (X학습, X시험, y학습, y시험)을 외울 것. 순서 바꾸면 조용히 망한다.
# random_state=42 02의 RandomState(42)와 같은 역할. 단, 섞는 방식이 달라 02와 '같은 140대'는 아니다.
print("\n[2] train", X_train.shape, "/ test", X_test.shape)

scaler = StandardScaler()  # 표준화 도구
scaler.fit(X_train)  # 학습용의 평균·표준편차를 '외운다'  (02 의 mu, sd 계산)
Z_train = scaler.transform(X_train)  # 외운 값으로 변환 (02의 (X-mu) / sd)
Z_test = scaler.transform(X_test)  # 시험용도 학습용 눈금으로 '변환만'

print(
    "    scaler 가 외운 평균:",
    scaler.mean_.round(1),
    "← 02 의 mu 역할 (분할이 달라 값은 조금 다름)",
)

reg = LinearRegression().fit(
    Z_train, y_train
)  # 만들기와 fit 을 한 줄에 붙여 쓰기도 합니다
print("    표준화 가중치:", dict(zip(feature_name, reg.coef_.round(3).tolist())))

# dict(zip(이름, 값)) = 이름: 값 짝 사전
# 02와 분할이 달라 숫자는 조금 다르지만 그림은 같다: 공기온도가 압도적
print(
    f"    train R² {reg.score(Z_train, y_train):.4f} / test R² {reg.score(Z_test, y_test):.4f}"
)
# train 0.816 / test 0.743 -> 차이 0.073. 02의 경보선(0.005 정상 / 0.10 의심) 사이
# 시험용이 60대뿐이라 흔들리는 정도.
# 과적합인지 확인하려면 교차검증에서 다시 본다.

# =====================================================================
# 3. Pipeline — "표준화는 학습용으로만" 을 도구가 대신 지키게
# =====================================================================
from sklearn.pipeline import make_pipeline

# 파이프라인 = 전처리(표준화 등)와 모델을 순서대로 이어 붙여 '하나의 모델'처럼 다루는 것.
# 왜? fit 할 땐 학습용으로 스케일러를 맞추고, predict 할 땐 변환만 하는 걸 도구가 알아서 해줌
# 사람이 실수로 scalar.fit(X_test)를 못하게 구조로 막는 것
# 그리고 새 데이터도 '원래 눈금' 그대로 넣으면 된다.

pipe = make_pipeline(
    StandardScaler(), LinearRegression()
)  # 왼쪽부터 순서대로: 표준화 → 선형회귀
pipe.fit(X_train, y_train)  # 원래 눈금 X 를 그냥 넣는다. 안에서 표준화까지 함
print("\n[3] Pipeline test R²:", round(pipe.score(X_test, y_test), 4), "(위와 같음)")
print(
    "    새 설비 [공기 300, 회전 1500, 토크 40, 마모 100] → 공정온도",
    round(pipe.predict([[300, 1500, 40, 100]])[0], 2),
)

# 함정 누수(leakage)가 얼마나 속이는지 눈으로봄
# 누수 = 시험용 데이터의 정보가 학습 쪽으로 새어 들어가는 것. 점수가 부풀러지고, 현장에서 무너짐
# 제일 흔한 사고: 00에서 배운 '중복 행 제거'를 깜빡한 경우
# 같은 측정이 두 번 들어 있으면,  무작위로 나눌 때 쌍둥이가 학습용과 시험용에 갈라져 들어간다.
# 그럼 모델은 시험 문제를 이미 학습용에서 본 셈

from sklearn.neighbors import KNeighborsRegressor

# KNeighborsRegressor(1) = "제일 비슷한 설비 한 대를 찾아 그 설비의 정답을 그대로 말하는"
# 외우기에 특화돼 있어서 누수를 드러내기 좋다

neighbors = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))

X중복 = np.vstack([X, X])  # 일부러 모든 행을 두 번씩 (중복 제거를 깜빡한 상황)
y중복 = np.concatenate([y, y])
d_tr, d_te, dy_tr, dy_te = train_test_split(
    X중복, y중복, test_size=0.3, random_state=42
)
print("\n[3-1] 중복 행을 안 지우고 나누면 (누수)")
print(
    f"    중복 있음 test R² {neighbors().fit(d_tr, dy_tr).score(d_te, dy_te):.4f}   <- 훌륭해 보인다"
)
print(
    f"    중복 없음 test R² {neighbors().fit(X_train, y_train).score(X_test, y_test):.4f}   <- 이게 이 모델의 진짜 실력"
)
print(
    "    같은 모델, 같은 데이터입니다. 중복을 안 지운 것 하나로 점수가 네 배가 됐어요."
)

# 현장에 넣으면 어떤 점수가 나오냐? 0.19이다. 0.82는 존재하지 않는 실력
# 막는 법 두 가지:
# 1) 나누기 전에 중복을 지운다. 같은 설비/같은 시각이 두 번 있으면 의심할 것
# 2) 표준화 같은 전처리는 반드시 학습용으로만 fit한다.
# 3) 현장 배포 형태가 바로 이거.

# =====================================================================
# 4. 분류 — 03 의 로지스틱 회귀를 도구로
# =====================================================================
from sklearn.linear_model import LogisticRegression

yc = df["고장여부"].values  # 분류 정답: 0/1
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, yc, test_size=0.3, random_state=3, stratify=yc
)  # stratify=yc = 고장 비율을 양쪽에 똑같이. 03 2번에서 손으로 한 일이 단어 하나
print(
    "\n[4] 분류 — 학습용 고장", yc_train.sum(), "대 / 시험용 고장", yc_test.sum(), "대"
)

clf = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
)  # max_iter = 최대 걸음 수. 기본 100 이 모자라면 경고가 떠서 1000 으로
clf.fit(Xc_train, yc_train)  # 03 의 sigmoid + 로그손실 + 경사하강 2000 바퀴가 이 한 줄

p = clf.predict_proba(Xc_test)[:, 1]
판정 = clf.predict(
    Xc_test
)  # 0.5 기준 판정. 임계값을 바꾸려면 p 를 직접 자릅니다: (p >= 0.2)
print(
    "    시험용 고장 확률 상위 5:", np.sort(p)[::-1][:5].round(3)
)  # 정렬 → [::-1] 뒤집어 큰 순 (00 미리보기 ⑥) → 앞 5개
print(
    "    정확도:",
    round(clf.score(Xc_test, yc_test), 3),
    "← 03 에서 배웠듯 이것만 보면 속는다",
)


# =====================================================================
# 5. 채점 도구 — 03 에서 손으로 센 네 칸·재현율을 함수로
# =====================================================================
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
)

print(
    "\n[5] 혼동행렬 (사이킷런은 [[정상→정상, 정상→고장], [고장→정상, 고장→고장]] 순서)"
)
print(
    confusion_matrix(yc_test, 판정)
)  # 인자 순서: (실제, 판정). 바꾸면 표가 뒤집힙니다
tn, fp, fn, tp = confusion_matrix(yc_test, 판정).ravel()
print(f"    잡음(TP) {tp}  놓침(FN) {fn}  헛경보(FP) {fp}  통과(TN) {tn}")

print("\n    classification_report — 정밀도·재현율을 한 표에 (고장 행을 보세요)")
print(
    classification_report(yc_test, 판정, target_names=["정상", "고장"], zero_division=0)
)

print("    임계값을 내리면 (03 8번을 도구로)")
for th in [0.5, 0.3, 0.2, 0.1]:
    판정_th = (p >= th).astype(int)
    print(
        f"      임계값 {th}: 재현율 {recall_score(yc_test, 판정_th, zero_division=0):.2f}"
        f"  정밀도 {precision_score(yc_test, 판정_th, zero_division=0):.2f}"
    )
clf_b = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
)
clf_b.fit(Xc_train, yc_train)
판정_b = clf_b.predict(Xc_test)
print(
    f"    class_weight='balanced' 로 다시 학습 → 임계값 0.5 에서 재현율 {recall_score(yc_test, 판정_b):.2f}"
    f"  정밀도 {precision_score(yc_test, 판정_b, zero_division=0):.2f}  (놓침 대신 헛경보를 택한 것)"
)


# =====================================================================
# 6. 규제 — 과적합 처방을 옵션 하나로 (Ridge)
# =====================================================================
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
# 규제 = 가중치가 너무 커지지 않게 손실에 '벌점'을 더 하는 것. 과적합(외우기)를 누른다.
# 02 5-1에서 과적합 모델은 가중치가 커졌었다. 그거 억누르면 외우기가 어려워짐
# Ridge(릿지) = 벌점을 붙인 선형회귀.
# Lasso(라쏘) = 별점 방식이 달라 쓸모없는 가중치를 아예 0으로 만드는 버전
# 왜 하나? 센서는 많고 데이터는 적을 때 모델이 학습 데이터를 통째로 외우는 걸 막으려고

print("\n[6] Ridge alpha 별 점수 — 이 데이터(센서 4개)에선")
for a in [0.01, 1, 10, 100]:
    r = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train)
    print(
        f"    alpha={a:<5} → train {r.score(X_train, y_train):.4f}  test {r.score(X_test, y_test):.4f}"
    )

# alpha 10까지는 거의 그대로, 100이면 너무 눌러 둘다 나빠짐
# 이 데이터는 손잡이가 5개뿐이라 외울 힘이 없어서 규제가 할 일이 없다.
# 없는 병에 약을 세게 쓰면 탈 난다.

# 그럼 규제가 진짜 필요한 상황을 만들어 본다. 센서 4개를 곱하고 제곱해 특징 70개로 부풀리면
# 외울 힘이 생김.
print(
    "\n[6-1] 특징을 70개로 부풀리면 (PolynomialFeatures) — 과적합이 생기고, Ridge 가 고친다"
)
for 이름, 모델 in [
    ("규제 없음(LinearRegression)", LinearRegression()),
    ("Ridge(alpha=10)", Ridge(alpha=10)),
]:
    poly = make_pipeline(
        StandardScaler(), PolynomialFeatures(4), StandardScaler(), 모델
    ).fit(X_train, y_train)
    print(
        f"    {이름:26s} → train {poly.score(X_train, y_train):.3f}  test {poly.score(X_test, y_test):.3f}"
    )

# 규제 없음: train 0.89 / test 0.43 -> 차이 0.46 전형적 과적합 학습 점수는 올랐는데 시험 점수는 무너짐
# 규제 있음: train 0.87 / test 0.65 -> 학습 점수를 조금 양보하고 시험 점수를 되찾음. 이게 규제가 하는 일
# 그래도 원래 4개 특징(0.74)보다 못한다. 이 데이터는 직선이 정답이라 부풀릴 이유가 없었던 것.
# 정리: train >> test 보이면 규제
# 둘 다 낮으면 -> alpha 줄여 / 특징 더 / 더 유연한 모델

# =====================================================================
# 7. 손잡이(alpha 등) 고르기 — test 를 훔쳐보지 않고: 교차검증
# =====================================================================
# 하이퍼파라미터 = 학습으로 정해지는 게 아니라 '사람이 정하는' 손잡이. alpha, 학습률, 에폭, 임계값
# 위처럼 test 점수를 보며 alpha를 고르면, 그 test 점수는 이미 '고르는 데 쓴 점수'라 부풀려짐
# 교차검증(CV) = 학습용을 5조각으로 나눠,
from sklearn.model_selection import cross_val_score, GridSearchCV

cv = cross_val_score(
    make_pipeline(StandardScaler(), LinearRegression()), X_train, y_train, cv=5
)
print("\n[7] 교차검증 5조각 R²:", cv.round(3), "→ 평균", round(cv.mean(), 4))
# 조각마다 0.71~0.85로 흔들린다. train/test 차이가 0.073은 이 흔들림 안에 있음
# 과적합 아님, 데이터가 적어서 튀는 것
# 이렇게 교차검증은 "그 차이가 진짜 과적합인지, 우연인지"를 가려줌

# GridSearchCV = 후보값들을 전부 교차검증으로 돌려 최고를 고르고, 그 값으로 다시 학습까지 해 주는 역할
grid = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5,
)
grid.fit(X_train, y_train)  # 후보 5개 × 5조각 = 25번 학습
print(
    "    교차검증으로 고른 alpha:",
    grid.best_params_,
    "/ CV 평균 R²:",
    round(grid.best_score_, 4),
)
print("    그 모델의 test R² (이제 딱 한 번):", round(grid.score(X_test, y_test), 4))

# 손으로 했던 이유 -> 에러가 나거나 점수가 이상하거나 -> 모델들 안이 어떻게 생겨 먹었는지는 알 수 있음
# 안을 들여다 볼 수 는 있다.
# 순서가 중요하다: 학습용 안에서 교차검증으로 손잡이를 고르고 -> 마지막에 test를 '딱 한 번' 본다.

# =====================================================================
# 8. 저장 — 내일 다시 쓰려면
# =====================================================================
import joblib

joblib.dump(
    clf, "고장분류기.joblib"
)  # 파이프라인(스케일러+모델) 통째로 파일로. 학습 결과가 다 들어감
다시 = joblib.load(
    "고장분류기.joblib"
)  # 내일 이 한 줄로 불러오면 재학습 없이 바로 예측
print(
    "\n[8] 저장 후 불러와 예측 — 새 설비 [공기 299, 회전 1400, 토크 55, 마모 240] 고장 확률:",
    round(다시.predict_proba([[299, 1400, 55, 240]])[0, 1], 3),
)

# 파이프라인째 저장하는 이유: 스케일러가 외운 평균 표준편차까지 같이 저장돼야 새 데이터를 같은 눈금으로 변환
# 모델만 저장하면 내일 표준화를 다르게 해서 엉뚱한 답이 나온다.

# =====================================================================
# 9. 정리 — 손코드 ↔ 사이킷런 대응표, 그리고 언제 쓰나
# =====================================================================
print("""
[9] 손으로 한 것 ↔ 사이킷런
    01 표준화 (x−평균)/표준편차     ↔  StandardScaler().fit(train) / .transform()
    02 섞어서 70:30 자르기          ↔  train_test_split(X, y, test_size=0.3, random_state=…, stratify=y)
    01·02 경사하강 300·500걸음      ↔  LinearRegression().fit(X, y)   (공식으로 단번에)
    03 sigmoid + 로그손실 + 경사하강 ↔  LogisticRegression().fit(X, y)  (속으로 걸음, max_iter)
    03 확률 / 0.5 판정              ↔  .predict_proba(X)[:, 1] / .predict(X)
    03 네 칸·재현율 손으로 세기      ↔  confusion_matrix / classification_report / recall_score
    w, b                           ↔  .coef_, .intercept_
    (없음) 과적합 처방               ↔  Ridge(alpha) / Lasso(alpha)
    (없음) 불균형 처방               ↔  class_weight="balanced" (+ 임계값 조정)
    (없음) 손잡이 고르기             ↔  cross_val_score / GridSearchCV

    사이킷런은 언제?  표 데이터, 몇백~몇십만 행, 선형모델·트리·부스팅 → 실무 기본. fit 한 줄. 이 과정 데이터는 전부 여기.
    파이토치는 언제?  학습 루프를 내 손으로 쥐어야 할 때 — 아주 큰 데이터, 손실을 내 맘대로 바꿀 때,
                     그리고 다음 과정에서 배울 '신경망'(층 쌓기) → 05 파일에서 같은 문제를 파이토치로 풀어 봅니다.
""")

# =====================================================================
# 실습 — Ridge 말고 Lasso 는?
# =====================================================================
# [문제] 9번 대응표에 Ridge 와 함께 Lasso 가 적혀 있었습니다.
# Lasso(alpha=...) 로 바꿔서 alpha 0.01 / 0.1 / 1.0 의 train, test 점수를 보세요.
#
#   힌트: from sklearn.linear_model import Lasso 부터. 나머지는 6번의 Ridge 코드와 같습니다.
from sklearn.linear_model import Lasso

print("\n[실습] Lasso alpha 별 점수 — 이 데이터(센서 4개)에선")
for a in [0.01, 1, 10, 100]:
    r = make_pipeline(StandardScaler(), Lasso(alpha=a)).fit(X_train, y_train)
    print(
        f"    alpha={a:<5} → train {r.score(X_train, y_train):.4f}  test {r.score(X_test, y_test):.4f}"
    )
