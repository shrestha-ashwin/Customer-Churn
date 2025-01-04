import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get("GROQ_API_KEY")
)

def load_model(fileName): 
  with open(fileName, 'rb') as file:
    return pickle.load(file)

decision_tree_model = load_model('dt_model.pkl')
random_forest_model = load_model('rf_model.pkl')
knn_model = load_model('knn_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
svm_model = load_model('svm_model.pkl')
xgboost_model = load_model('xgb_model.pkl')
xgboost_featureEngineered_model = load_model('xgboost_enhanced.pkl')
xgboost_SMOTE_model = load_model('xgboost_SMOTE.pkl')
voting_classifier_model = load_model('voting_clf.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):

  input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_credit_card,
    "isActiveMember" : is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography_France": 1 if location == "France" else 0 ,
    "Geography_Germany": 1 if location == "Germany" else 0 ,
    "Geography_Spain": 1 if location == "Spain" else 0 ,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_female": 1 if gender == "Female" else 0,  
 }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict
  

def make_prediction(input_df, input_dict):
  probabilites = {
    "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
    "RandomForest": random_forest_model.predict_proba(input_df)[0][1],
    "K-NearestNeighbours": knn_model.predict_proba(input_df)[0][1],
  }

  avg_prob = np.mean(list(probabilites.values()))

  st.markdown("### Model Probabilites")

  for model, prob in probabilites.items():
    st.write(f"{model} - {prob}")
  st.write(f"Average Probability: {avg_prob}")

  return avg_prob

def explain_prediction(probability, input_dict, surname):

  prompt = f'''You are a data scientist expert at a bank, where you specialize in interpreting and explaining the predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100,1)} of churning, based on information provided below.

  Here is the customer information: {input_dict}

  Here are the machine learning model's top 10 most important features for predicting customer churn:

         Feature	  |    Importance
---------------------------------------
  	NumOfProducts   |	    0.323888
  	IsActiveMember	|     0.164146
  	           Age	|     0.109550    
 Geography_Germany	|     0.091373
  	       Balance	|     0.052786
  Geography_France  |    	0.046463
     Gender_Female	|     0.045283
   Geography_Spain	|     0.036855
  	CreditScore	    |     0.035005
   EstimatedSalary	|     0.032655
  	HasCrCard	      |     0.031940
  	Tenure	        |     0.030054
    Gender_Male	    |     0.000000

    {pd.set_option('display.max_columns', None)}

    Here is the summary statistic for churned customers:

    {df[df['Exited'] == 1].describe()}

    Here is the summary statistic for non-churned customers:

    {df[df['Exited'] == 0].describe()}

- If the user has over 40% risk of churning, generate a 3 sentence explanation of why they are at a risk of churning.
- If the user has less than 40% risk of churning, generate a 3 sentence explanation of why they might not be at a risk of churning.
- Choose only one type of explanation of 3 sentences, either they are at risk of churning or they are not at risk of churning. No calculations, no probability, no customer data needs to be explained. Just explain the prediction.
- Your explanation should be based on the customer's information, summary statistic of churned and non-churned customers, and the feature importances provided. 

Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.

    
  '''
  print("Prompt: ", prompt)
  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[{
      "role": "user",
      "content": prompt,
    }],
  )
  return raw_response.choices[0].message.content
 
def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentived with various offers
  
  You noticed a customer {surname} has a {round(probability * 100,1)}% probability of churning.

  Here is the customer information: 
  {input_dict}

  Here is the explanation of why the customer might be at a risk of churing: 
  {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at a risk of churning, or offering them incentives so they become more loyal to the bank.

  Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
  """
  raw_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{
      "role": "user",
      "content": prompt,
    }],
  )
  print("Email: \n", raw_response)
  return raw_response.choices[0].message.content


st.title("Churn Customer Prediction")

df = pd.read_csv("churn.csv.xls")

customers = [f"{row['CustomerId']} - {row['Surname']}"
            for _, row in df.iterrows()]

customer_selected_option = st.selectbox("Select a customer", customers)

if customer_selected_option:
  customer_id = int(customer_selected_option.split("-")[0])
  customer_surname = customer_selected_option.split("-")[1]


selected_customer = df.loc[df["CustomerId"] == customer_id].iloc[0]

print(selected_customer)

col1, col2 = st.columns(2)

with col1:

  credit_score = st.number_input("Credit Score",
                                min_value=300,
                                max_value=850,
        value=int(selected_customer['CreditScore']))

  
  location = st.selectbox(
    "Location", ["Spain","France","Germany"],index=["Spain","France","Germany"].index(selected_customer['Geography'])
  )

  gender = st.radio("Gender", ["Male", "Female"],
                   index = 0 if selected_customer['Gender'] == "Male" else 1)

  age = st.number_input("Age", min_value=10, max_value=100,
                     value=int(selected_customer['Age']))

  tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=int(selected_customer['Tenure']))

with col2:
  balance = st.number_input("Balance", min_value=0.0,value=float(selected_customer['Balance']))

  num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))

  has_credit_card = st.checkbox("HasCrCard",
                            value=bool(selected_customer['HasCrCard']))

  is_active_member = st.checkbox("IsActiveMember",
value=bool(selected_customer['IsActiveMember']))

  estimated_salary = st.number_input("Estimated Salary",
                                   min_value=0.0, value=float(selected_customer['EstimatedSalary']))           
  input_df,input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

avg_prob = make_prediction(input_df, input_dict)

explanation = explain_prediction(avg_prob, input_dict, selected_customer['Surname'])

st.markdown("--")
st.subheader("Explanation of Prediction")
st.markdown(explanation)

email = generate_email(avg_prob, input_dict, explanation, selected_customer['Surname'])

st.markdown("--")
st.subheader("Email to Customer")
st.markdown(email)
                            