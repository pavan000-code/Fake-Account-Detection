from django.shortcuts import render,redirect
import pandas as pd
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score


def index(request):
    return render(request,'index.html')

def home(request):
    global df,X,y
    if request.method == "POST":
        file = request.FILES['myfile']
        df = pd.read_csv(file)
        df = df.rename(
            columns={'edge_followed_by': 'Followed_by', 'edge_follow': 'Follow', 'username_length': 'name_length',
                     'username_has_number': 'has_number', 'full_name_has_number': 'full_name_number',
                     'full_name_length': 'fullname_length', 'is_private': 'private', 'is_joined_recently': 'recent',
                     'has_channel': 'channel', 'is_business_account': 'business_account', 'has_guides': 'guides',
                     'has_external_url': 'external_url', 'is_fake': 'fake'})
        df.drop(['guides'], axis=1, inplace=True)
        df.drop(['channel'], axis=1, inplace=True)
        X = df.drop(['fake'], axis=1)
        y = df['fake']
        return render(request, 'upload.html', {'df': df})
    return render(request,'home.html')


def model(request):
    global x_train, y_train, x_test, y_test,a,auc,avc
    if request.method == 'POST':
        name = request.POST['cars']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        if name == "SVM":
            from sklearn.metrics import accuracy_score
            clf = svm.SVC(kernel='linear')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            a = accuracy_score(y_pred, y_test)
            messages.success(request,"Support Vector Machine Accuracy :")
            return render(request, 'algorithm.html',{'a':a})
        elif name=='NN':
            from sklearn.metrics import accuracy_score
            model = Sequential()
            model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
            model.add(Dense(40))
            model.add(Dense(1, activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x=x_train, y=y_train, verbose=1, epochs=10)
            pre = model.predict(x_test)
            auc= accuracy_score(pre, y_test)
            messages.success(request,"Neural Networks Accuracy")
            return render(request,'algorithm.html',{'a':auc})
        elif name == 'SVM-NN':
            from sklearn.metrics import accuracy_score
            import tensorflow as tf
            model = Sequential()
            model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
            model.add(Dense(40))
            model.add(Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x=x_train, y=y_train, verbose=1, epochs=10)
            pre = model.predict(x_test)
            avc = accuracy_score(pre, y_test)
            messages.success(request, "SVM-ANN accuracy")
            return render(request, 'algorithm.html', {'a': avc})
        else:
            messages.success(request,"You didn't selected any model")
            return render(request,'algorithm.html')
    return render(request,'algorithm.html')

def prediction(request):
    if request.method=='POST':
        Followed_by = request.POST['Followed_by']
        Follow=request.POST['Follow']
        name_length=request.POST['name_length']
        has_number=request.POST['has_number']
        full_name_number=request.POST['full_name_number']
        fullname_length=request.POST['fullname_length']
        private=request.POST['private']
        recent=request.POST['recent']
        business_account=request.POST['business_account']
        external_url=request.POST['external_url']
        c=list((Followed_by,Follow,name_length,has_number,full_name_number,fullname_length,private,recent,business_account,external_url))
        clf = svm.SVC(kernel='linear')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        n=clf.predict([c])
        if n==[1]:
            messages.success(request,"It is a fake account")
            return render(request, 'prediction.html')
        else:
            messages.success(request,"it is not a fake account")
            return render(request, 'prediction.html')
    return render(request,'prediction.html')


def graphs(request):
    x = ['Support Vector Machine','Neural Networks','SVM_NN']

    y = [a,auc,avc]

    graph = sns.barplot(x, y)
    plt.title('Performance Comparision')
    graph.set(ylabel="Accuracy")
    plt.show()
    return redirect('/')






















