# 택시 요금 예측 모델 : y = 2x + 5  기본 요금 5000원 + 거리당 2000원 증가
def taxi_fare_model(distance):
    # distance :  이동 거리 → 택시 요금 예측
    weight = 2     # 1km당 2천원
    bias = 5         # 기본요금 5천원

    fare = weight * distance + bias
    return fare

# 실제 사용
distance = 5      # 이동 거리 5km
predicted_fare = taxi_fare_model(distance)
print(f"{distance}km 이동 시 예상 택시 요금: {predicted_fare},000원")  # 5km 이동 시 예상 택시 요금: 15,000원

"""
위 예제에서 순전파 과정은
  - 입력 받기 : 이동 거리 5km -> 계산 : 2 * 5 + 5000 -> 출력 가격 : 15000
즉, 순전파란 입력 데이터가 모델의 연산 흐름을 따라 이동하며, 예측값을 생성하는 과정을 말한다.


둘째, 손실(cost, loss, error) 계산
  - 모델의 예측값과 실제값의 차이를 측정하는 단계다.
  -  모델 학습의  최종 목표는 손실값이 최소화되도록 하는 것이다.
"""
print('---------------')
# "실제 택시 요금이 13,000원이었다"라고 했을 때 모델이 15,000원으로 예측 → 2,000원 차이
# 택시 요금 모델의 오차 보정이 필요해 보인다.
# 택시 요금 역전파 예제로 업그레이드 :  실제값과 예측값의 오차를 이용하여 가중치와 편향을 수정한다.

actual_fare = 13
predicted_fare = 15
cost_error = predicted_fare - actual_fare   # 오차 계산(예측값 - 실제값)   15 - 13 = +2
print(f"오차: {cost_error}천원 (너무 비싸게 예측)")
print("가중치와 편향을 조금 줄여야겠다...")


# 셋째, 역전파(Backpropagation)
#   -  계산된 오차를 기반으로 모델의 퍼하미터(가중치와 편향)을 조정해 출력에 미치는 영향을 계산한다.

# 기존 모델 파라미터
weight = 2;  bias = 5;  distance = 5
learning_rate = 0.01   # 학습률 : 한 번 학습할 때 가중치와 편향을 얼마나 많이 수정할지 결정하는 값.
# 반복적으로 weight와 bias에 조금만 반영하면서 수정할 목적. 이러면 점차 weight와 bias의 값을 얻게됨

# 역전파를 통한 가중치와 편향 업데이트 : 이해를 위해 매우 간단하게 표현한 예제
weight = weight - learning_rate * cost_error * distance    # 2 - 0.01 * 2 * 5 = 1.9 
bias = bias - learning_rate * cost_error                          # 5 - 0.01 * 2 = 4.98
print(f"새로운 weight: {weight:.2f}")       # 1.90 (2에서 줄어 듦)
print(f"새로운 bias: {bias:.2f}")               # 4.98 (5에서 줄어 듦)

# 새로운 가중치와 편향으로 다시 예측
new_predicted_fare = weight * distance + bias   # 14.48  <--- 1.9 * 5 + 4.98
print(f"새로운 예상 택시 요금: {new_predicted_fare:.2f}천원")
    # 기존 예측값 : 15.000원,  실제 요금   : 13.000원,  새로운 예측값: 14.480원


print('\n~~ 택시 요금 모델 반복 학습 (cost를 줄이기 위한 노력) ~~~~~~~~~~~')
actual_fare = 13
weight = 2
bias = 5

for i in range(10):
    predicted_fare = weight * distance + bias   # 1. 현재 모델로 예측
    error = predicted_fare - actual_fare            # 2. 실제값과의 오차
    print(f"{i + 1}회 학습 전 → 예측: {predicted_fare:.2f}천원, 오차: {error:.2f}")

    # 3. 오차를 이용하여 weight와 bias 갱신 (역전파를 통한 갱신)
    weight = weight - learning_rate * error * distance
    bias = bias - learning_rate * error

print("\n반복 완료")
print(f"최종 weight : {weight:.4f}, 최종 bias   : {bias:.4f}")
# 최종 갱신된 weight와 bias로 예측
final_fare = weight * distance + bias
print(f"반복 완료 후 최종 예상 택시 요금 : {final_fare:.2f}천원")
    # ~~ 택시 요금 모델 반복 학습 ~~~~~~~~~~~~~~
    # 1회 학습 전 → 예측: 15.00천원, 오차: 2.00
    # 2회 학습 전 → 예측: 14.48천원, 오차: 1.48
    # ...
    # 9회 학습 전 → 예측: 13.18천원, 오차: 0.18
    # 10회 학습 전 → 예측: 13.13천원, 오차: 0.13
    # 반복 완료
    # 최종 weight : 1.6343, 최종 bias   : 4.9269
    # 반복 완료 후 최종 예상 택시 요금 : 13.10천원


print('\n** pytorch lib 사용 *****************')
import torch

# 택시 요금 모델 ----- y = weight * distance + bias
distance = torch.tensor(5.0)      # 입력값 : 이동 거리 5km
actual_fare = torch.tensor(13.0)  # 실제 택시 요금 : 13천원

# 학습해야 할 값 : requires_grad=True → PyTorch가 미분값(gradient)을 계산하도록 설정
weight = torch.tensor(2.0, requires_grad=True)
bias = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.01   # 학습률

for i in range(10):
    # 1. 순전파(Forward) : 현재 weight와 bias를 이용하여 택시 요금 예측
    predicted_fare = weight * distance + bias
    error = predicted_fare - actual_fare      # 2. 오차 계산
    loss = 0.5 * error ** 2               # 손실 함수 : 1/2을 곱한 제곱 오차

    print(
        f"{i + 1}회 학습 전 → "
        f"예측: {predicted_fare.item():.2f}천원, "
        f"오차: {error.item():.2f}, "
        f"손실: {loss.item():.4f}"
    )

    # 3. 역전파(Backpropagation) : loss를 기준으로 weight와 bias의 기울기를 자동 계산
    # 왜 미분을 할까? 우리가 알고 싶은 것은 단순히 "오차가 2다" 가 아니다.
    # 더 중요한 것은 weight를 조금 바꾸면 loss가 얼마나 변하는가?  
    # bias를 조금 바꾸면 loss가 얼마나 변하는가? 이다.  이것을 알려주는 값이 미분값 = 기울기(gradient)임.
    # 미분은 weight나 bias를 조금 변화시켰을 때 loss가 얼마나 변하는지를 알아보기 위해 사용한다.
    # loss.backward()는 loss부터 거꾸로 계산하면서 각 학습 파라미터의 미분값(gradient)을 자동으로 구해줌.

    loss.backward()
    # PyTorch가 계산한 미분값(gradient)을 확인하려면
    print("weight의 기울기 :", weight.grad.item())   # 기울기 > 0 이면 우로 갈수록 loss 증가, 좌로 이동해야 loss 감소
    print("bias의 기울기   :", bias.grad.item())

    # 4. 가중치와 편향 수정 : gradient 계산 과정은 기록하지 않도록 no_grad 사용
    with torch.no_grad():  # with 블록 안에서 실행되는 계산은 학습 그래프의 자동 미분 기록에서 제외하겠다는 뜻
        weight -= learning_rate * weight.grad   # 가중치와 편향을 실제로 수정하는 단계
        bias -= learning_rate * bias.grad

    # 5. 이전 gradient 초기화 : loss.backward()를 실행할 때 계산한 gradient를 .grad에 덮어쓰는 것이 아니라 누적한다.
    # 그래서 매 학습 반복마다 이전 gradient를 지워줘야 한다.
    weight.grad.zero_()
    bias.grad.zero_()

print("\n학습 완료")
print(f"최종 weight : {weight.item():.4f}")
print(f"최종 bias   : {bias.item():.4f}")

# 학습된 weight와 bias로 다시 예측
final_fare = weight * distance + bias
print(f"최종 예상 택시 요금 : {final_fare.item():.2f}천원")
    # ** pytorch lib 사용 *****************
    # 1회 학습 전 → 예측: 15.00천원, 오차: 2.00, 손실: 2.0000
    # weight의 기울기 : 10.0
    # bias의 기울기   : 2.0
    # 2회 학습 전 → 예측: 14.48천원, 오차: 1.48, 손실: 1.0952
    # weight의 기울기 : 7.399997711181641
    # bias의 기울기   : 1.4799995422363281
    # ...
    # 9회 학습 전 → 예측: 13.18천원, 오차: 0.18, 손실: 0.0162
    # weight의 기울기 : 0.8991909027099609
    # bias의 기울기   : 0.1798381805419922
    # 10회 학습 전 → 예측: 13.13천원, 오차: 0.13, 손실: 0.0089
    # weight의 기울기 : 0.6654071807861328
    # bias의 기울기   : 0.13308143615722656
    # 학습 완료
    # 최종 weight : 1.6343
    # 최종 bias   : 4.9269
    # 최종 예상 택시 요금 : 13.10천원


print('\n** PyTorch 학습 코드 형태로 작성하기 *****************')
import torch
import torch.nn as nn
import torch.optim as optim

# 택시 요금 모델  :  y = weight * distance + bias
distance = torch.tensor([[5.0]])       # 입력값 : 이동 거리 5km
print(distance.shape)   # torch.Size([1, 1])
# 대괄호가 두 번 있는 이유는 nn.Linear 같은 레이어가 입력을 2차원 형태 (배치 크기, 특성 수)로 받기 때문
actual_fare = torch.tensor([[13.0]])  # 실제 택시 요금 : 13천원

# 1. 모델 생성 : 입력 1개(distance) → 출력 1개(fare)
model = nn.Linear(1, 1)

# 편의상 기존 예제와 동일하게 초기값 설정 : weight = 2, bias = 5
# 지금 하는 일은 학습 계산이 아니라 단순히 초기값을 직접 설정하는 작업이므로 자동 미분 기록이 필요 없다.
with torch.no_grad():
    model.weight.fill_(2.0)   # fill_(2.0)은 텐서의 모든 값을 2.0으로 채운다는 뜻
    model.bias.fill_(5.0)       # 메서드 이름 끝에 _가 붙으면 보통 원본 텐서 자체를 직접 변경하는 in-place 연산

loss_fn = nn.MSELoss()   # 2. 손실 함수
# 평균 제곱 오차(Mean Squared Error)를 계산 - 예측값과 실제값이 얼마나 차이 나는지를 제곱해서 평균낸 값

# 3. 옵티마이저 : weight와 bias를 자동으로 수정
    # 역전파로 계산된 gradient를 이용해서 모델의 weight와 bias를 손실이 줄어드는 방향으로 자동 수정해 주는 도구.
    # 쉽게 말하면:
    # loss.backward()가 “어느 방향으로 얼마나 움직여야 하는지” 기울기를 계산하고,
    # optimizer.step()이 그 기울기를 이용해서 실제로 weight와 bias를 수정한다.
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사하강법(Stochastic Gradient Descent)으로 최적화

for i in range(10):
    # 4. 순전파(Forward) : 내부적으로 weight * distance + bias 계산
    predicted_fare = model(distance)

    # 현재 예측 오차
    error = predicted_fare.item() - actual_fare.item()

    # 5. 손실 계산 : 기존 예제의 0.5 * error ** 2와 동일하게 맞춤
    loss = 0.5 * loss_fn(predicted_fare, actual_fare)   # 앞의 직접 구현 코드에서 loss = 0.5 * error ** 2

    print(
        f"{i + 1}회 학습 전 → "
        f"예측: {predicted_fare.item():.2f}천원, "
        f"오차: {error:.2f}, "
        f"손실: {loss.item():.4f}"
    )
   
    optimizer.zero_grad()  # 6. 이전 gradient 초기화
    loss.backward()        # 7. 역전파  
    optimizer.step()       # 8. weight와 bias 수정

print("\n학습 완료")
print(f"최종 weight : {model.weight.item():.4f}")
print(f"최종 bias   : {model.bias.item():.4f}")

# 학습된 모델로 최종 예측
final_fare = model(distance)
print(f"최종 예상 택시 요금 : {final_fare.item():.2f}천원")
    # ** PyTorch 학습 코드 형태로 작성하기 *****************
    # 1회 학습 전 → 예측: 15.00천원, 오차: 2.00, 손실: 2.0000
    # 2회 학습 전 → 예측: 14.48천원, 오차: 1.48, 손실: 1.0952
    # 9회 학습 전 → 예측: 13.18천원, 오차: 0.18, 손실: 0.0162
    # 10회 학습 전 → 예측: 13.13천원, 오차: 0.13, 손실: 0.0089
    # 학습 완료
    # 최종 weight : 1.6343
    # 최종 bias   : 4.9269
    # 최종 예상 택시 요금 : 13.10천원

    # 전체 흐름은 이렇다.
    # distance  ->  model(distance)  ->  예측값  ->  MSELoss  ->  손실값  ->
    # loss.backward()  ->  gradient 계산  ->  optimizer.step()  -> weight, bias 수정
