import matplotlib.pyplot as mat
import pandas as pd
import numpy as np

diabetes = pd.read_csv("/media/ziad/Data/Python/start/ML_Project/Machine-Learning-Project/diabetes.csv")

Pregnancies_list = list(diabetes['Pregnancies'])
Glucose_list = list(diabetes['Glucose'])
BloodPressure_list = list(diabetes['BloodPressure'])
SkinThickness_list = list(diabetes['SkinThickness'])
Insulin_list = list(diabetes['Insulin'])
BMI_list = list(diabetes['BMI'])
DiabetesPedigreeFunction_list = list(diabetes['DiabetesPedigreeFunction'])
Age_list = list(diabetes['Age'])
Outcome_list = list(diabetes['Outcome'])

# mat.scatter(Outcome_list, Pregnancies_list)
# mat.scatter(Outcome_list, Glucose_list)
# mat.scatter(Outcome_list, BloodPressure_list)
# mat.scatter(Outcome_list, SkinThickness_list)
# mat.scatter(Outcome_list, Insulin_list)
# mat.scatter(Outcome_list, BMI_list)
# mat.scatter(Outcome_list, DiabetesPedigreeFunction_list)
# mat.scatter(Outcome_list, Outcome_list)

# mat.show()

outcome_1 = diabetes[diabetes['Outcome'] == 1]
print(outcome_1)
print("the mean for the glucose values for the diabitic ones: ",(np.array(outcome_1['SkinThickness'])).mean())
output_DF = outcome_1.to_csv("/media/ziad/Data/Python/start/ML_Project/Machine-Learning-Project/outcome_1.csv", index = False)