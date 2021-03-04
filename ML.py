


#import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import IPython, graphviz, re
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

path = r"C:\Users\medad\לימודים\תואר שני\שנה ב\סמינר על נוף וסביבה\ML_data.xlsx"

df = pd.read_excel(path)


# 'Death'
df_all = df[['Temp','Humidity','Death']]

for i in ['Temp','Humidity']:
    df_all[i] = pd.to_numeric(df[i],errors = 'coerce')




def RandomForestClassifier(data_traning,csv = ''):
    
    X = data_traning.iloc[:,:-1].values
    y = data_traning.iloc[:,-1] .values
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(X, y)  

    
    print(clf.feature_importances_) # מראה את כוח הקורולציה של המשתנים
    pred = clf.predict(X)
    
    df2    = pd.DataFrame(pred,columns = ["predict"])
    result = pd.concat([data_traning,df2], axis = 1)
    
    if csv != '':
        result.to_csv(csv)
    
    return result



result = RandomForestClassifier(df_all,csv = '')
    
    
def draw_tree(t, col_names, size=9, ratio=0.5, precision=3):

    s=export_graphviz(t, out_file=None, feature_names=col_names, filled=True,special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',f'Tree {{ size={size}; ratio={ratio}',s)))
    

def DecisionTree(data,max_depth = 3 ,new_predict = None):
    
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    reger_1 = DecisionTreeRegressor(max_depth=max_depth)
    reger_1.fit(X_train,y_train)
    
    if new_predict == None:
        new_predict = X_test
        
    y_1 = reger_1.predict(new_predict)
    return y_1,reger_1

def Logistic(df):
    
    X = df.iloc[:,[1]].values
    y = df.iloc[:,-1] .values
    
    
    print ("Logistic")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)
    
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    
    print ("confusion metrix")
    cm = confusion_matrix(y_test,y_pred)
    print (cm)
    
    reg_coef      = classifier.coef_
    reg_intercept = classifier.intercept_
    
    #print("Intercept (Constant) is: %.2f" % reg_intercept)
    #print("Coeficient vector is:", reg_coef)
    
    return  (reg_intercept,reg_coef)


def get_formola(coef,intercept_,num = 4):
    
    lenNum     = len(coef[0])
    print (lenNum)
    coef_list = coef[0]
    makadam   = 'x^'
    range_me  = list(range(lenNum))
    
    
    list_finish = []
    for i in range(lenNum):
        if coef_list[i] > 0:
            sign = '+'
        else:
            sign = '-'
        list_finish.append(sign+str(abs(float(round(coef_list[i],4))))+ makadam + str(range_me[i]))
        
    list_finish = list_finish[1:]
    list_finish.insert(0,str(float(round(intercept_[0],num))))
    string = ''.join(i for i in list_finish)
    
    print  (string)
    return (string)

y_1,reger_1 =  DecisionTree(df_all)

draw_tree(reger_1, ['Temp','Humidity'], size=9, ratio=0.5, precision=3)


reg_intercept,reg_coef = Logistic(df_all)



get_formola(reg_coef,reg_intercept,num = 4)