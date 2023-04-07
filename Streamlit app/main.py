import string
from unittest import result
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pickle
import requests
import time
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from PIL import Image


st.set_page_config(
    page_title="IOT FOOD MONITOR",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)



def timetaken_DAL(quality):
        if (quality>=9.15):
            time = (quality - 9.05 ) / 0.003979646611374403
            myvar1 = "Dal is fresh. Consume it under "
            myvar2 = round(time)
            myvar3 = " mins. Enjoy your meal"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is fresh. Consume it under") + round(time) + str("mins"))
        if (quality>=8.7 and quality < 9.15):
            time = (quality - 8.6 ) / 0.001877660059602652
            myvar1 = "Dal is cold and needs to be reheated. Heat and eat before "
            myvar2 = round(time)
            myvar3 = " mins"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is cold and needs to be reheated. Heat and eat before") + round(time)+ str(" mins"))
        if (quality>=7.8 and quality < 8.7) :
            time = (quality - 7.7 ) /  0.0024480416168478265
            myvar1 = "It needs to be stored in the refrigerator before the given time. Time left to spoil is "
            myvar2 = round(time)
            myvar3 = " mins"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Time left to spoil is") + round(time) + str("mins. Store in the refrigerator before") + round(time) + str("mins"))
        elif (quality< 7.8):
            s = ("Dal is spoiled, discard it.")

        return s

def timetaken_RICE(quality):
        if (quality>=9.1):
            time = (quality - 9.0 ) / 0.0005752517860538821
            myvar1 = "Rice is fresh. And it can be consumed upto "
            myvar2 = round(time)
            myvar3 = " mins. Enjoy your meal."
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is fresh. Consume it under") + round(time) + str("mins"))
        if (quality>=6.5 and quality < 9.1):
            time = (quality - 6.4 ) / 0.010911497036900369
            myvar1 = "Rice is cold and needs to be reheated. Heat and eat before "
            myvar2 = round(time)
            myvar3 = " mins"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is cold and needs to be reheated. Heat and eat before") + round(time)+ str(" mins"))
        if (quality>=5.0 and quality < 6.5) :
            time = (quality - 4.9 ) /  0.0036010291037181993
            myvar1 = "It needs to be stored in the refrigerator before the given time. Time left to spoil is "
            myvar2 = round(time)
            myvar3 = " mins"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Time left to spoil is") + round(time) + str("mins. Store in the refrigerator before") + round(time) + str("mins"))
        elif (quality< 5.0):
            s = ("Rice is spoiled, discard it.")

        return s

def timetaken_MILK(quality):
        if (quality>=8.2):
            time = (quality - 8.1 ) / 0.004739068958333333
            myvar1 = "Milk is fresh. Consume it under "
            myvar2 = round(time)
            myvar3 = " mins. Enjoy your drink."
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is fresh. Consume it under") + round(time) + str("mins"))
        if (quality>=7.5 and quality < 8.2):
            time = (quality - 7.4 ) / 0.00810364128571428
            myvar1 = "Milk is cold and needs to be reheated. Bacteria may grow soon. Heat and consume before "
            myvar2 = round(time)
            myvar3 = " mins"
            s = ("{}{}{}".format(myvar1, myvar2, myvar3))
            #s = (str("Food is cold and needs to be reheated. Heat and eat before") + round(time)+ str(" mins"))
        elif (quality< 7.5):
            s = ("Milk is spoiled, discard it.")

        return s


header = st.container()
dataset = st.container()
features = st.container()
data = st.container()
model_training = st.container()


with header:


    st.title("PROJECT - FOOD QUALITY MONITORING SYSTEM USING IOT")
    st.text("")
    st.text("")
    with st.expander("We can predict how long a food item can last"):
        st.write("Here we have used machine learning algorithms to train our model to predict how long our food items can be stored for before they are deemed unfit for out consumption. Check it our for yourself !!")
    st.text("")
    st.text("")
    
with dataset:
    st.subheader('We have created dataset of 3 most commonly served food items- Dal, Rice and Milk')
    st.text('We collected this dataset using arduino')
    st.text("")

with features:
    st.header('The features I created')

    col1, col2, col3, col4 = st.columns(4)
    col1.text("Temperature")
    image = Image.open('temp.png')
    col1.image(image, width= 150)
    col2.text("Humidity")
    image = Image.open('humidity.png')
    col2.image(image, width= 100)
    col3.text("Ethanol")
    image = Image.open('ethanol.png')
    col3.image(image, width= 100)
    col4.text("Methane")
    image = Image.open('methane.png')
    col4.image(image, width= 100)
    st.text("")
    st.text("")

df1 = pd.read_csv('DAL_Final.csv')
df2 = pd.read_csv('MILK_FINAL.csv')
df3 = pd.read_csv('RICE_FINAL.csv')


with data:
    food = st.selectbox(
        "Pick a food item",
        ('Dal', 'Milk', 'Rice'))
    st.text("")
    st.text("")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)        
    col1.write("You have selected :")
    col2.write(food)
    if food=="Dal" :
        df = df1   
    elif food == "Milk" :
        df = df2
    else:
        df=df3
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    col8.download_button(
        label="Download data",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )


    st.text("")
    st.text("")



def CheckFood():
    while True:
    
        url1 = "https://api.thingspeak.com/channels/1785779/feeds/last.json?api_key=64LX24ENHBDW3JZV"   #methane
        url2 = "https://api.thingspeak.com/channels/1780792/feeds/last.json?api_key=6PKCOVRLO9U2HFJ7"   #ethanol
        url3 = "https://api.thingspeak.com/channels/1744711/feeds/last.json?api_key=LQYCBT76OZKU05HV"   #humidity
        url4 = "https://api.thingspeak.com/channels/1744701/feeds/last.json?api_key=2HY016JWRK0O7IFI"   #temp


        response1 = requests.get(url1)
        data_disc1 = json.loads(response1.text)
        input1 =  data_disc1['field1']                                      

        response2 = requests.get(url2)
        data_disc2 = json.loads(response2.text)
        input2 =  data_disc2['field1']

        response3 = requests.get(url3)
        data_disc3 = json.loads(response3.text)
        input3 = data_disc3['field1']

        response4 = requests.get(url4)
        data_disc4 = json.loads(response4.text)
        input4 = data_disc4['field1']


        if food=="Dal" :
            loaded_model = pickle.load(open('trained_model_dal.sav', 'rb'))
        elif food=="Milk":
            loaded_model = pickle.load(open('trained_model_milk.sav', 'rb'))
        elif food=="Rice" :
            loaded_model = pickle.load(open('trained_model_rice.sav', 'rb'))

        
        #input_data = (input1 ,input2 ,input3, input4)
        input_data = (140.6 ,120.7 , 90, 29.5)
        input_data = list(np.float_(input_data))                                #methane, ethanol, humidity, temp 
        print(input_data)

        input_data_as_numpy = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        q = prediction[0]
        if food=="Dal" :
            k = timetaken_DAL(q)
        elif food=="Milk":
            k = timetaken_MILK(q) 
        elif food=="Rice" :
            k = timetaken_RICE(q) 

        st.success(k)
        methane = input_data[0]
        ethanol = input_data[1]
        humidity = input_data[2]
        temp = input_data[3]


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", temp,)
        col2.metric("Humidity", humidity)
        col3.metric("Ethanol", ethanol)
        col4.metric("Methane", methane)

        if food=="Dal" :
            image = Image.open('dal.jpg')
            st.image(image, width= 250)
        elif food=="Milk":
            image = Image.open('milk.jpg')
            st.image(image, width= 250) 
        elif food=="Rice" :
            k = timetaken_RICE(q)  
            image = Image.open('rice.jpg')
            st.image(image, width= 400)
  
        break


result = st.button("Click here to find current food qualiy")
if result:
    CheckFood()
    result = False    

