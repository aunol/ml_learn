# main.py 파일

from sklearn.datasets import make_regression # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 1. 더미 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 2. 데이터셋 분할 (훈련용 데이터와 테스트용 데이터로 나누기)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 정의 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

# 5. 시각화ㄹ
plt.plot(X_test, y_pred, color="red", label="Predicted Line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
