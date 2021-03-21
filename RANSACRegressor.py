import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                'python-machine-learning-book-2nd-edition'
                '/master/code/ch10/housing.data.txt',
                header=None,
                sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
            'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
            'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['RM']].values
y = df[['MEDV']].values

#
# RANSAC
# 1. 랜덤한 일부 샘플을 정상치로 선택해서 모델을 훈련한다.
# 2. 위 훈련된 모델에서 다른 모든 포인트를 테스트하고 
#    사용자지정 허용 오차에 속한 포인트들만 추출해서 정상치에 추가한다.
# 3. 정상치에 추가된 모든 포인트들을 사용해서 모델을 다시 훈련한다.
# 4. 위 훈련된 모델의 오차를 추정한다.
# 5. 성능이 사용자가 지정한 임계값에 도달하거나 지정된 반복횟수에 도달하면 알고리즘 종료.
#

# residual_threshold : 정상치 포인트 판정 기준이 되는 허용 오차.
# if residual_threshold is None -> using MAD(Median Absolute Deviation)

ransac = RANSACRegressor(LinearRegression(), max_trials=100,
                        min_samples=50,
                        loss='absolute_loss',
                        residual_threshold=5.0,
                        random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolors='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolors='white',
            marker='s',label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()