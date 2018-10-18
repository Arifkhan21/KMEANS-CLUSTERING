from sklearn.datasets import load_iris
iris = load_iris()
x=iris.data
y=iris.target

print("\nfeatures:\n")
print(iris.feature_names)
print("\n target names:\n")
print(iris.target_names) 

print("\ntarget value:\n")
print(y)


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
KModel = kmeans.fit(iris.data)

KModel.cluster_centers_

y_pred=KModel.predict(x)

print("\nprdicted values by model:\n")
print(y_pred)

from sklearn import metrics
print("\nAccuracy:\n")
print(metrics.accuracy_score(y,y_pred))