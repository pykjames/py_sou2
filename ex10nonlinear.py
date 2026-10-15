import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 선형회귀와 다항회귀 degree=2 비교
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(7)
torch.manual_seed(7)

# 1. 다항회귀에 적합한 예제 CSV 데이터 생성
# 광고비가 증가하면 매출도 증가하지만,어느 정도 이후에는 증가폭이 둔화되는 곡선 데이터
ad_cost = np.linspace(0, 100, 80)

# sales = -0.06 * 광고비² + 7.5 * 광고비 + 40 + noise
# 제곱항 때문에 직선이 아니라 곡선 형태의 데이터가 생성됨
sales = (
    -0.06 * (ad_cost ** 2) + 7.5 * ad_cost + 40
        + np.random.normal(0, 25, size=len(ad_cost))
)

df = pd.DataFrame({"광고비": ad_cost,"매출": sales})
df.to_csv("ad_sales_poly.csv", index=False, encoding="utf-8-sig")
print("CSV 파일 생성 완료: ad_sales_poly.csv")

# 2. CSV 데이터 읽기
df = pd.read_csv("ad_sales_poly.csv")
print(df.head(3))
print(df.info())

# 3. 결측치 제거
df = df[["광고비", "매출"]].copy()
df = df.dropna()
print("결측치 제거 후 데이터 크기:", df.shape)

# 4. feature, label 분리
X = df[["광고비"]].values.astype(np.float32)
y = df[["매출"]].values.astype(np.float32)

# 5. 산점도로 데이터 확인
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7)
plt.xlabel("Advertising Cost")
plt.ylabel("Sales")
plt.title("Advertising Cost vs Sales")
plt.grid(True)
plt.show()

# 6. train/test split : sklearn 없이 직접 섞고 분리
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
train_size = int(len(X) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
print("X:", X_train.shape, X_test.shape)
print("y:", y_train.shape, y_test.shape)

# 7. Scaling : train 데이터 기준으로 평균과 표준편차 계산
# test 데이터는 train 기준 평균/표준편차로 변환
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# 8. 다항 특성 생성 함수
# degree=2이면 [x, x^2] 형태의 입력 데이터 생성
def make_poly_features(X_scaled, degree=2):
    features = [X_scaled ** d for d in range(1, degree + 1)]
    return np.concatenate(features, axis=1).astype(np.float32)

X_train_poly = make_poly_features(X_train_scaled, degree=2)
X_test_poly = make_poly_features(X_test_scaled, degree=2)
print("선형회귀 입력 shape:", X_train_scaled.shape)
print("다항회귀 입력 shape:", X_train_poly.shape)

# 9. numpy 데이터를 PyTorch Tensor로 변환
X_train_scaled_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled_t = torch.tensor(X_test_scaled, dtype=torch.float32)

X_train_poly_t = torch.tensor(X_train_poly, dtype=torch.float32)
X_test_poly_t = torch.tensor(X_test_poly, dtype=torch.float32)

y_train_scaled_t = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_scaled_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# 10. 평가 함수 : R² score, MSE, RMSE 직접 계산
def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(name, y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score_np(y_true, y_pred)

    print(f"\n[{name}]")
    print("MSE :", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R²  :", round(r2, 3))

# 11. PyTorch 선형회귀 모델 정의 : 입력 feature 수가 1개이므로 Linear(1, 1)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 12. PyTorch 다항회귀 모델 정의 : 입력 feature가 [x, x^2] 이므로 Linear(2, 1)
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

linear_model = LinearRegressionModel()
poly_model = PolynomialRegressionModel()
print("\n선형회귀 모델")
print(linear_model)
print("\n다항회귀 모델")
print(poly_model)

# 13. PyTorch 학습 함수 : Keras의 compile() + fit() 역할을 직접 수행
def train_model(model, X_train_t, y_train_t, epochs=2000, lr=0.01):
    # 손실 함수 : 평균 제곱 오차
    criterion = nn.MSELoss()

    # optimizer : Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        model.train()  # 학습 모드
        y_pred = model(X_train_t)  # 예측
        loss = criterion(y_pred, y_train_t)  # 손실 계산
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트
        loss_history.append(loss.item())

        # 중간 학습 상태 출력
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:4d} / {epochs}, loss: {loss.item():.6f}")

    return loss_history


# 14. 선형회귀 모델 학습 : y_train이 아니라 y_train_scaled를 사용하여 학습
print("\n선형회귀 모델 학습 시작")
linear_loss_history = train_model(
    linear_model, X_train_scaled_t, y_train_scaled_t,
    epochs=2000, lr=0.01
)

# 15. 다항회귀 모델 학습 : 입력 데이터는 [x, x^2] 형태의 다항 특성
# y_train_scaled를 사용하여 학습
print("\n다항회귀 모델 학습 시작")
poly_loss_history = train_model(
    poly_model, X_train_poly_t, y_train_scaled_t,
    epochs=2000, lr=0.01
)

# 16. 예측 : PyTorch에서는 예측 시 model.eval()과 torch.no_grad() 사용
linear_model.eval()
poly_model.eval()

with torch.no_grad():
    # 예측값은 scaled y 기준으로 나옴
    y_pred_linear_scaled_t = linear_model(X_test_scaled_t)
    y_pred_poly_scaled_t = poly_model(X_test_poly_t)

# Tensor를 numpy 배열로 변환
y_pred_linear_scaled = y_pred_linear_scaled_t.numpy()
y_pred_poly_scaled = y_pred_poly_scaled_t.numpy()

# 원래 매출 단위로 복원
y_pred_linear = y_pred_linear_scaled * y_std + y_mean
y_pred_poly = y_pred_poly_scaled * y_std + y_mean

# 17. 성능 비교
evaluate_model("선형회귀", y_test, y_pred_linear)
evaluate_model("다항회귀 degree=2", y_test, y_pred_poly)

# 18. 학습 손실 시각화
plt.figure(figsize=(8, 5))
plt.plot(linear_loss_history, label="Linear Regression Loss")
plt.plot(poly_loss_history, label="Polynomial Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# 19. 예측 곡선 시각화
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1).astype(np.float32)

X_plot_scaled = (X_plot - X_mean) / X_std
X_plot_poly = make_poly_features(X_plot_scaled, degree=2)
X_plot_scaled_t = torch.tensor(X_plot_scaled, dtype=torch.float32)
X_plot_poly_t = torch.tensor(X_plot_poly, dtype=torch.float32)

linear_model.eval()
poly_model.eval()

with torch.no_grad():
    # 예측값은 scaled y 기준으로 나옴
    y_plot_linear_scaled_t = linear_model(X_plot_scaled_t)
    y_plot_poly_scaled_t = poly_model(X_plot_poly_t)

y_plot_linear_scaled = y_plot_linear_scaled_t.numpy()
y_plot_poly_scaled = y_plot_poly_scaled_t.numpy()

# 원래 매출 단위로 복원
y_plot_linear = y_plot_linear_scaled * y_std + y_mean
y_plot_poly = y_plot_poly_scaled * y_std + y_mean

plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, alpha=0.5, label="Train Data")
plt.scatter(X_test, y_test, alpha=0.8, label="Test Data")
plt.plot(X_plot, y_plot_linear, label="Linear Regression")
plt.plot(X_plot, y_plot_poly, label="Polynomial Regression degree=2")
plt.xlabel("Advertising Cost")
plt.ylabel("Sales")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

# 20. 학습된 계수 확인
# PyTorch nn.Linear의 weight, bias 확인
# weight shape: 선형회귀: (1, 1),  다항회귀: (1, 2)
linear_w = linear_model.linear.weight.detach().numpy()
linear_b = linear_model.linear.bias.detach().numpy()

poly_w = poly_model.linear.weight.detach().numpy()
poly_b = poly_model.linear.bias.detach().numpy()

print("\n선형회귀 식 - scaled 기준")
print(f"y_scaled = {linear_w[0][0]:.3f} * x_scaled + {linear_b[0]:.3f}")

print("\n다항회귀 식 - scaled 기준")
print(
    f"y_scaled = {poly_w[0][1]:.3f} * x_scaled^2 "
    f"+ {poly_w[0][0]:.3f} * x_scaled "
    f"+ {poly_b[0]:.3f}"
)
