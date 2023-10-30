from ast import literal_eval
#import os
import pickle
import pandas as pd
import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
shap.initjs()

import lightgbm as lgb
import lightgbm

# palette
jaune ='#FFCA18'
rouge ='#88001B'
bleu ='#000064'
vert ='#055D00'
pink = '#FC6C85'
choco = '#4b2312'
cyan = '#00FFFF'
orange = '#FF6103'
magenta = '#FF00FF'
pastel = ['#FFE06F','#FF9594','#78CF80','#FFB178','#A7B9FF','#FDE8D8']
pastel2 = [rouge, jaune, bleu, pink, vert, choco, cyan, orange, magenta]

df_test = pd.read_csv('./ressources/test_sample.csv', index_col=0)
df_test = df_test.set_index('SK_ID_CURR')
df_train = pd.read_csv('./ressources/train_sample.csv', index_col=0)
df_train = df_train.set_index('SK_ID_CURR')
df_train = df_train.drop(columns=['TARGET'])
cm_met1 = pd.read_csv('./ressources/cm_met1_sample.csv', index_col=0)
cm_met1 = cm_met1.set_index('SK_ID_CURR')

# load the model + explainer from disk
model_met1 = pickle.load(open('./ressources/model_met1.sav', 'rb'))

#host:

#HOST = 'http://127.0.0.1:8000'     # developement on local server
HOST = 'http://40.115.33.188' #Azure 


## Streamlit ##########
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

##add images
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
with col1:
    st.image('Image1.jpg', width=150)
with col2:
    st.write(" ")
with col3:
    st.write(" ")
with col4:
    st.write(" ")
with col5:
    st.write(" ")
with col6:
    st.write(" ")
with col7:
    st.write(" ")
with col8:
    st.write(" ")
with col9:
    st.image('Image2.png', width=140)



st.markdown("<h1 style='text-align: center;'>Projet 7 : 'Implémentez un modèle de scoring'</h1>", unsafe_allow_html=True)

# Selectbox
ID = st.selectbox('Sélection du client par son numéro ID', df_test.index)


#get prediction
def get_prediction(id_client: int):
    """Gets the probability of default of a client on the API server.
    Args : 
    - id_client (int).
    Returns :
    - probability of default (float).
    """
    json_client = df_test.loc[int(id_client)].to_json()
    response = requests.get(HOST+'/prediction/', data=json_client, timeout=80)
    proba_default = eval(response.content)["probability"]
    print(response.content)
    #proba_default = response["probability"]
    result = round(proba_default*100, 1)
    print(result)
    return result 
#f"La probabilité du client {str(var)} d'appartenir à la classe 1 (non solvable) est de {str(result)}."


## Gaugeplot
def gauge(var, var2):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 100-var2,
    domain = {'x': [0, 1], 'y': [0, 1]},
    number = {"prefix": "Score : ", "suffix": " %"},
    title = {'text': f"Probabilité du client {str(var)} d'être solvable (classe 0))"},
    gauge = {'axis': {'range': [None, 100]},
            'threshold' : {'line': {'color': choco, 'width': 4}, 'thickness': 0.75, 'value': 65},
            'bar': {'color': vert if (100-var2) > 65 else rouge}}))
            #'bar': {'color': vert if (100-var2) > 61 else rouge}}))
         #   'steps' : [
          #      {'range': [0, 40], 'color': rouge},
           #     {'range': [40, 60], 'color': 'orange'},
            #    {'range': [60, 80], 'color': jaune},
             #   {'range': [80, 100], 'color': vert}]})),"""
    fig.update_layout(font = {'color': "darkblue", 'family': "Arial"}) #paper_bgcolor = "lavender", 
    #fig.show()
    return fig

#col1, col2 = st.columns(2)

st.plotly_chart(gauge(ID, get_prediction(ID)), use_container_width=True)

# Selectbox
feature = st.selectbox('Sélection de la feature', df_train.columns)


def kde_fig(id: int, Feature: str = 'EXT_SOURCE_1'):
    df = df_test.loc[id]
    #print(df.shape)
    x = df[Feature]
    #print(x)
    index_x1 = list(cm_met1.index[cm_met1['TARGET']==0])
    index_x2 = list(cm_met1.index[cm_met1['TARGET']==1])
    x1 = np.array(df_train[Feature].drop(index_x2))
    x2 = np.array(df_train[Feature].drop(index_x1))
    
    fig = go.Figure()
    fig.add_trace(go.Violin(x=x1, line_color=rouge, name='Classe 0 (solvable)', y0=0, opacity=0.4))
    fig.add_trace(go.Violin(x=x2, line_color=bleu, name= 'Classe 1 (insolvable)', y0=0, opacity=0.4))

    fig.update_traces(orientation='h', side='positive', meanline_visible=True)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)   
    # Add title
    fig.update_layout(title_text=f'Distribution des classes 0 et 1 pour la feature "{Feature}"')
    fig.add_vline(x=x, line_width=3, line_dash="dash", line_color=magenta)
    fig.add_annotation(x=x, y=0,
            text=f"Value {ID} : {round(x, 2)}",
            showarrow=True,
            font=dict(family="sans serif", size=14, color=magenta),
            arrowhead=2)
    #fig.show()
    return fig

st.plotly_chart(kde_fig(ID, feature), use_container_width=True)

##shap
def shap_fig(id: int):
    df = df_test.loc[[id]]
    explainer = shap.Explainer(model_met1, df_test)
    shap_values = explainer(df, check_additivity=False)
    st.title(f'SHAP values du client n°{id}')
    st_shap(shap.plots.waterfall(shap_values[0]))
    #plt.title(f'SHAP values du client n°{id}')
    return

shap_fig(ID)