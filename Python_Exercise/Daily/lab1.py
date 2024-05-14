import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


class NaiveBayes:
    def __init__(self):
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        # 获得特征数和样本数
        n_samples, n_features = X.shape
        # 获得类别
        self.classes = np.unique(y)
        # 获得类别的数量
        n_classes = len(self.classes)

        # 计算特征条件概率
        self.feature_prob = np.zeros((n_classes, n_features))

        # 计算类别先验概率
        self.class_prior = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_prior[i] = np.sum(y == c) / n_samples

        for i, cls in enumerate(self.classes):
            # 选择属于类别 c 的样本
            X_cls = X[y == cls]
            # 计算条件概率 P(feature|class)
            self.feature_prob[i] = np.sum(X_cls, axis=0) / np.sum(X_cls)

    def predict(self, X):
        # 获取样本数量
        n_samples, _ = X.shape
        # 初始化概率矩阵
        probs = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            # 计算类别条件概率对数
            class_prob = np.log(self.class_prior[i])
            feature_prob = np.log(self.feature_prob[i] + 1e-9)

            # 计算每个样本的概率
            probs[:, i] = np.sum(X.multiply(feature_prob), axis=1).squeeze() + class_prob

        # 对每个样本预测概率最高的类别
        return self.classes[np.argmax(probs, axis=1)]


# 导入训练、测试数据集
train_data = pd.read_csv("D:/Course/machine learning/Lab1/train.csv")
test_data = pd.read_csv("D:/Course/machine learning/Lab1/test.csv")

# 初始化CountVectorizer
vectorizer = CountVectorizer()

# 向量化训练数据
X_train = vectorizer.fit_transform(train_data["review"])
y_train = train_data['category']

# 向量化测试数据
X_test = vectorizer.transform(test_data["review"])
y_test = test_data['category']

# 实例化和训练朴素贝叶斯分类器
nb = NaiveBayes()
nb.fit(X_train, y_train)

# 进行预测和准确率评估
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()

# 第二问
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class NaiveBayes:
    def __init__(self, alpha=0.1):
        # 引入拉普拉斯平滑参数
        self.alpha = alpha
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        # 获取数据集的样本数和特征数
        n_samples, n_features = X.shape
        # 获取所有类别
        self.classes = np.unique(y)
        class_amounts = len(self.classes)

        # 计算各类别的先验概率
        self.class_prior = np.zeros(class_amounts)
        for i, c in enumerate(self.classes):
            self.class_prior[i] = np.sum(y == c) / n_samples

        # 计算特征的条件概率
        self.feature_prob = np.zeros((class_amounts, n_features))

        for i, cls in enumerate(self.classes):
            # 选择属于类别 c 的样本
            X_cls = X[y == cls]
            feature_counts = np.sum(X_cls, axis=0) + self.alpha
            self.feature_prob[i] = feature_counts / np.sum(feature_counts)

    def predict(self, X):
        n_samples, _ = X.shape
        probs = np.zeros((n_samples, len(self.classes)))
        # 计算每个类别的条件概率的对数值
        for i, c in enumerate(self.classes):
            class_prob = np.log(self.class_prior[i])
            feature_prob = np.log(self.feature_prob[i])
            # 计算每个样本属于每个类别的概率的对数值之和
            probs[:, i] = np.sum(X.multiply(feature_prob), axis=1).squeeze() + class_prob

        # 返回具有最高概率的类别作为预测结果
        return self.classes[np.argmax(probs, axis=1)]


train_data = pd.read_csv("D:/Course/machine learning/Lab1/train.csv")
test_data = pd.read_csv("D:/Course/machine learning/Lab1/test.csv")

# 使用TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["review"])
X_test_tfidf = tfidf_vectorizer.transform(test_data["review"])

# 获取训练和测试数据的类别标签
y_train = train_data['category']
y_test = test_data['category']

# 实例化和训练朴素贝叶斯分类器（调整平滑参数）
nb = NaiveBayes()
nb.fit(X_train_tfidf, y_train)

# 进行预测和准确率评估
y_pred = nb.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()

# 第三问
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_prior = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_prior[i] = np.sum(y == c) / n_samples

        self.feature_prob = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            feature_counts = np.sum(X_c, axis=0) + self.alpha
            self.feature_prob[i] = feature_counts / np.sum(feature_counts)

    def predict(self, X):
        n_samples, _ = X.shape
        probs = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            class_prob = np.log(self.class_prior[i])
            feature_prob = np.log(self.feature_prob[i])
            probs[:, i] = np.sum(X.multiply(feature_prob), axis=1).squeeze() + class_prob

        return self.classes[np.argmax(probs, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        self.alpha = params['alpha']
        return self


#
#nltk.download('stopwords')
#nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # 清理和预处理文本数据
    # 去掉标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 将文本转换为小写
    text = text.lower()
    # 将文本标记为单词
    tokens = word_tokenize(text)
    # 删除停止词
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 将令牌连接回文本
    clean_text = ' '.join(filtered_tokens)
    return clean_text


# 加载和预处理训练和测试数据
train_data = pd.read_csv("D:/Course/machine learning/Lab1/train.csv")
test_data = pd.read_csv("D:/Course/machine learning/Lab1/test.csv")

train_data['clean_review'] = train_data['review'].apply(clean_text)
test_data['clean_review'] = test_data['review'].apply(clean_text)

# 使用TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["clean_review"])
X_test_tfidf = tfidf_vectorizer.transform(test_data["clean_review"])

y_train = train_data['category']
y_test = test_data['category']

# 设置参数网格
param_grid = {'alpha': [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]}

# 实例化朴素贝叶斯分类器
nb = NaiveBayes()
grid_search = GridSearchCV(nb, param_grid, cv=5)

# 使用交叉验证进行参数选择
grid_search.fit(X_train_tfidf, y_train)

# 输出最佳参数
print("Best alpha:", grid_search.best_params_['alpha'])
print()
# 使用最佳参数重新训练模型
best_nb = NaiveBayes(alpha=grid_search.best_params_['alpha'])
best_nb.fit(X_train_tfidf, y_train)

# 进行预测和准确率评估
y_pred = best_nb.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()