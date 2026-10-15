# 실습) 문제2) 피마 인디언 당뇨병 데이터셋을 이용한 이진분류
# label:  Outcome  -   0 = 당뇨 아님,  1 = 당뇨 환자
# feature: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(12)
torch.manual_seed(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. 데이터 읽기 : 컬럼명이 없으므로 직접 지정
url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv"

# 칼럼명 : 인터넷에서 얻기
names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

df = pd.read_csv(url, names=names)
print(df.head())
print(df.info())
print(df['Outcome'].value_counts())   # 0:500, 1:268

# 3. feature, label 분리
x = df.drop('Outcome', axis=1).values.astype(np.float32)
y = df['Outcome'].values.astype(np.float32)
print("x shape:", x.shape)
print("y shape:", y.shape)


# 4. train / test split
# stratify=y:  Outcome 0과 1의 비율을 train/test에 비슷하게 유지
# 주의: 먼저 train/test를 나누고, 그 다음 train 기준으로 scaler를 fit 하는 것이 좋음
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=12, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 5. 스케일링
# train 데이터 기준으로 평균/표준편차 계산, test 데이터는 transform만 수행  
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
x_test_scaled = scaler.transform(x_test).astype(np.float32)

# 6. train 데이터를 train / validation으로 분리
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train_scaled, y_train, test_size=0.2, random_state=12, stratify=y_train
)
print("train:", x_train_part.shape, y_train_part.shape)
print("val  :", x_val.shape, y_val.shape)
print("test :", x_test_scaled.shape, y_test.shape)

# 7. NumPy 배열을 PyTorch Tensor로 변환
# BCEWithLogitsLoss는 예측값과 정답의 shape이 같아야 함
# y는 (데이터수,)가 아니라 (데이터수, 1) 형태로 변환
x_train_tensor = torch.tensor(x_train_part, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_part.reshape(-1, 1), dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# 8. DataLoader 생성 : # 학습 데이터를 32개씩 미니배치로 제공하고, 매 epoch마다 데이터 순서를 섞음
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 9. 모델 저장 폴더 준비
MODEL_DIR = './pimamodel/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 10. 평가 함수 정의
def evaluate_model(model, x_data, y_data, criterion):
    """
    PyTorch 평가 함수 - model.eval():평가 모드로 전환
    torch.no_grad(): 평가 시에는 gradient 계산이 필요 없으므로 비활성화
    BCEWithLogitsLoss를 사용하므로 모델 출력값 logits에 sigmoid를 따로 넣지 않고 loss 계산
    정확도 계산 시에는 logits -> sigmoid -> 0.5 기준 분류
    """

    model.eval()
    # Tensor를 CPU 또는 GPU로 이동한다. 모델과 입력/정답 Tensor는 같은 device에 있어야 한다.
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    with torch.no_grad():
        logits = model(x_data)
        loss = criterion(logits, y_data).item()

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        acc = (pred == y_data).float().mean().item()

    return loss, acc

# 11. 공통 학습 함수
def train_model(
    model, train_loader,  x_val_tensor, y_val_tensor, model_path, epochs=200, patience=15, lr=0.001
):
    """
    PyTorch 학습 함수
    TensorFlow의 다음 기능을 직접 구현: model.fit(), EarlyStopping, ModelCheckpoint
    EarlyStopping 기준: val_loss가 patience 횟수만큼 개선되지 않으면 학습 중단
    ModelCheckpoint 기준: val_loss가 가장 낮을 때 모델 가중치 저장
    """

    model = model.to(device)

    # 이진분류 손실 함수 : 모델 마지막에 Sigmoid를 넣지 않았으므로 BCEWithLogitsLoss 사용
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)   # Adam optimizer

    best_val_loss = np.inf
    patience_count = 0

    history = { 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [] }

    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)             # 1) forward: 예측값 계산
            loss = criterion(logits, batch_y)       # 2) loss 계산
            optimizer.zero_grad()                # 3) 이전 gradient 초기화
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_x.size(0)

            prob = torch.sigmoid(logits)   # accuracy 계산
            pred = (prob >= 0.5).float()

            train_correct += (pred == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate_model( model, x_val_tensor, y_val_tensor, criterion )

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] "
            f"loss: {train_loss:.4f} "
            f"accuracy: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} "
            f"val_accuracy: {val_acc:.4f}"
        )

        # ModelCheckpoint : val_loss가 가장 낮아졌을 때 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0

            torch.save(model.state_dict(), model_path)
            print(f"  -> best model 저장: {model_path}")
        else:
            patience_count += 1

        # EarlyStopping
        if patience_count >= patience:
            print(f"\nEarlyStopping 발생: {epoch + 1} epoch에서 학습 중단")
            break

    # 가장 성능이 좋았던 모델 가중치 복원
    # torch.load()는 저장된 가중치를 파일에서 읽고, load_state_dict()는 그 가중치를 현재 모델에 적용. 따라서 EarlyStopping 후 가장 성능이 좋았던 모델 상태로 복원할 수 있다.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 자연스럽고 좋다. 이유는 바로 앞에서 최적 가중치를 다시 불러왔고, 그 다음 단계가 보통 평가나 예측이기 때문.

    return model, history, criterion

# 12. Sequential API에 해당하는 PyTorch 모델
print('--- 1. Sequential 방식 모델 ---')
# TensorFlow Sequential API와 비슷한 방식 : 층을 순서대로 쌓는 단순 구조
seq_model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),

    nn.Linear(24, 12),
    nn.ReLU(),

    nn.Linear(12, 1)
)
print(seq_model)

seq_model_path = './pimamodel/seq_best_pytorch.pth'

seq_model, seq_hist, seq_criterion = train_model(
    model=seq_model,
    train_loader=train_loader,
    x_val_tensor=x_val_tensor,
    y_val_tensor=y_val_tensor,
    model_path=seq_model_path,
    epochs=200,
    patience=15,
    lr=0.001
)

seq_loss, seq_acc = evaluate_model(
    seq_model, x_test_tensor, y_test_tensor, seq_criterion
)
print(f'\nSequential 방식 모델 평가 - 손실: {seq_loss:.4f}, 정확도: {seq_acc:.4f}')

# 13. 커스텀 모델 클래스 방식
print('\n--- 2. 클래스 방식에 해당하는 nn.Module 모델 ---')
# PyTorch에서는 일반적으로 메이커가 제공한 nn.Module 클래스를 __init__()에서 layer를 정의하고
# forward()에서 데이터 흐름을 직접 작성함
class ClassLikeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(8, 24)
        self.dense2 = nn.Linear(24, 12)
        self.output = nn.Linear(12, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.output(x)
        return x


c_model = ClassLikeModel()
print(c_model)

c_model_path = './pimamodel/class_best_pytorch.pth'

c_model, c_hist, c_criterion = train_model(
    model=c_model,
    train_loader=train_loader,
    x_val_tensor=x_val_tensor,
    y_val_tensor=y_val_tensor,
    model_path=c_model_path,
    epochs=200,
    patience=15,
    lr=0.001
)

c_loss, c_acc = evaluate_model(
    c_model, x_test_tensor, y_test_tensor, c_criterion
)
print(f'\n 커스텀 모델 클래스 방식 모델 평가 - 손실: {c_loss:.4f}, 정확도: {c_acc:.4f}')

# 14. loss, accuracy 시각화
plt.figure(figsize=(14, 10))

# Sequential Loss
plt.subplot(2, 2, 1)
plt.plot(seq_hist['loss'], label='train loss')
plt.plot(seq_hist['val_loss'], label='val loss')
plt.title('Sequential Style - Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

# Sequential Accuracy
plt.subplot(2, 2, 2)
plt.plot(seq_hist['accuracy'], label='train acc')
plt.plot(seq_hist['val_accuracy'], label='val acc')
plt.title('Sequential Style - Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)

# 커스텀 모델 클래스 Loss
plt.subplot(2, 2, 3)
plt.plot(c_hist['loss'], label='train loss')
plt.plot(c_hist['val_loss'], label='val loss')
plt.title('Class Style - Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

# 커스텀 모델 클래스 Accuracy
plt.subplot(2, 2, 4)
plt.plot(c_hist['accuracy'], label='train acc')
plt.plot(c_hist['val_accuracy'], label='val acc')
plt.title(' 커스텀 모델 클래스 Style - Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 15. 예측 함수 정의
def predict_model(model, x_data):
    model.eval()

    x_data = x_data.to(device)

    with torch.no_grad():
        logits = model(x_data)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).int()

    return prob.cpu().numpy(), pred.cpu().numpy()

# 16. Sequential 방식 모델 예측
new_data = x_test_tensor[:5]

print('\n--- Sequential 방식 모델 예측 ---')
seq_prob, seq_pred = predict_model(seq_model, new_data)
print('예측 확률 : ', seq_prob.ravel())
print('예측 결과 : ', seq_pred.ravel())

# 17. 커스텀 모델 클래스 방식 모델 예측
print('\n--- 커스텀 모델 클래스 방식 모델 예측 ---')
c_prob, c_pred = predict_model(c_model, new_data)
print('예측 확률 : ', c_prob.ravel())
print('예측 결과 : ', c_pred.ravel())
print('\n실제 정답 : ', y_test[:5].astype(int))
