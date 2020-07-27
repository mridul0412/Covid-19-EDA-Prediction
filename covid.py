

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_context("notebook")
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot


def main():

    st.title("Covid 19 India ")
    st.sidebar.title("Covid 19 India")
    def home():
        st.image('coronavirus.jpg',use_column_width=True)
        st.markdown('Coronavirus is a family of viruses that can cause illness, which can vary from common cold and cough to sometimes more severe disease. Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV) were such severe cases with the world already has faced.\
                SARS-CoV-2 (n-coronavirus) is the new virus of the coronavirus family, which first discovered in 2019, which has not been identified in humans before. It is a contiguous virus which started from Wuhan in December 2019. Which later declared as Pandemic by WHO due to high rate spreads throughout the world.\
                Pandemic is spreading all over the world; it becomes more important to understand about this spread. This WebApp is an effort to analyze the cumulative data of confirmed, deaths, and recovered cases over time. In this Web Application, the main focus is to analyze the spread trend of this virus all India.')

        st.header('The following two curves shows why we need to flattern the curve and follow the social distancing measures:')
        st.image('flattening_curve_1.jpg',use_column_width=True)
        st.image('flattening_curve_2.jpg',use_column_width=True)

    if st.sidebar.button('Home',key=0):
        home()

    st.sidebar.title('Analysis')
    st.sidebar.text("      ")

    @st.cache(allow_output_mutation=True)
    def load_data():
        ageGroup = pd.read_csv('AgeGroupDetails.csv')
        covid19India = pd.read_csv('covid_19_india.csv')
        hospitalBeds = pd.read_csv('HospitalBedsIndia.csv')
        icmrTestLabs = pd.read_csv('ICMRTestingLabs.csv')
        indiDetails = pd.read_csv('IndividualDetails.csv')
        indiaCencus = pd.read_csv('population_india_census2011.csv')
        stateDetails = pd.read_csv('StatewiseTestingDetails.csv')
        df=pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

        return ageGroup,covid19India,hospitalBeds,icmrTestLabs,indiDetails,indiaCencus,stateDetails,df

    ageGroup,covid19India,hospitalBeds,icmrTestLabs,indiDetails,indiaCencus,stateDetails,df=load_data()

    df2=df

    def age_group():
        plt.figure(figsize=(14,8))
        sns.barplot(data=ageGroup,x='AgeGroup',y='TotalCases',color=sns.color_palette('Set3')[0])
        plt.title('Age Group Distribution')
        plt.xlabel('Age Group')
        plt.ylabel('Total Cases')
        for i in range(ageGroup.shape[0]):
            count = ageGroup.iloc[i]['TotalCases']
            plt.text(i,count+1,ageGroup.iloc[i]['Percentage'],ha='center')

        from IPython.display import display, Markdown
        display(Markdown("Most Number of cases have occured in the age group **20-50**"))
        st.pyplot()

        if st.checkbox('Show Data',False,key=1):
            st.write(ageGroup)

    if st.sidebar.checkbox('Age Group Analysis',False):
        st.subheader('Age Group Analysis')
        st.subheader("Most Number of cases have occured in the age group 20-50")
        age_group()


    indicopy=indiDetails.copy()



    def gender():
        plt.figure(figsize=(14,8))
        sns.countplot(data=indiDetails,x='gender',order=indiDetails['gender'].value_counts().index,color=sns.color_palette('Set3')[2])
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Total Cases')
        order2 = indiDetails['gender'].value_counts()

        for i in range(order2.shape[0]):
            count = order2[i]
            strt='{:0.1f}%'.format(100*count / indiDetails.gender.dropna().count() )
            plt.text(i,count+2,strt,ha='center')
        st.pyplot()

        if st.checkbox('Show Data',False,key=2):
            st.write(indiDetails)

    if st.sidebar.checkbox('Gender based Analysis',False):
        st.subheader('Gender Based Distriution')
        gender()

    def gender_missing():
        indicopy.gender.fillna('Missing',inplace = True)
        plt.figure(figsize=(14,8))
        sns.countplot(data=indicopy,x='gender',order=indicopy['gender'].value_counts().index,color=sns.color_palette('Set3')[1])
        plt.title('Gender Distribution (Considering Missing Values)')
        plt.xlabel('Gender')
        plt.ylabel('Total Cases')
        order2 = indicopy['gender'].value_counts()

        for i in range(order2.shape[0]):
            count = order2[i]
            strt='{:0.1f}%'.format(100*count / indicopy.shape[0])
            plt.text(i,count+2,strt,ha='center')
        st.pyplot()

        if st.checkbox('Show Data',False,key=3):
            st.write(indicopy)


    if st.sidebar.checkbox('Gender based Analysis(Missing Data)',False):
        st.subheader('Gender Distribution (Considering Missing Values)')
        gender_missing()





    covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
    df1=covid19India.groupby('Date').sum()
    df1.reset_index(inplace=True)

    def cov_analysis():
        plt.figure(figsize= (14,8))
        plt.xticks(rotation = 90 ,fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.xlabel("Dates",fontsize = 20)
        plt.ylabel('Total cases',fontsize = 20)
        plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)

        ax1 = plt.plot_date(data=df1,y= 'Confirmed',x= 'Date',label = 'Confirmed',linestyle ='-',color = 'b')
        ax2 = plt.plot_date(data=df1,y= 'Cured',x= 'Date',label = 'Cured',linestyle ='-',color = 'g')
        ax3 = plt.plot_date(data=df1,y= 'Deaths',x= 'Date',label = 'Death',linestyle ='-',color = 'r')
        plt.legend();
        st.pyplot()

        if st.checkbox('Show Data',False,key=4):
            st.write(df1)

    if st.sidebar.checkbox('Covid-19 Spread Analysis',False):
        st.subheader('Cases in India')
        cov_analysis()

    def bar_analysis():
        df2=df1.tail(25)
        df2['Date'] = df2['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        plt.figure(figsize=(14,8))
        sns.barplot(data=df2,x='Date',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')
        sns.barplot(data=df2,x='Date',y='Cured',color=sns.color_palette('Set3')[4],label='Cured')
        sns.barplot(data=df2,x='Date',y='Deaths',color=sns.color_palette('Set3')[5],label='Deaths')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.xticks(rotation = 90)
        plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)
        plt.legend(frameon=True,fontsize=12);
        st.pyplot()

        if st.checkbox('Show Data',False,key=5):
            st.write(df2)

    if st.sidebar.checkbox('Bar Chart Case Analysis',False):
        st.subheader('Bar Chart Analysis')
        bar_analysis()




    def state_analysis():
        state_cases=covid19India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
        state_cases['Active'] = state_cases['Confirmed'] - abs((state_cases['Deaths']- state_cases['Cured']))
        state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
        state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
        state_cases.sort_values('Confirmed', ascending= False).fillna(0).style.background_gradient(cmap='Reds',subset=["Confirmed"])\
                                .background_gradient(cmap='Blues',subset=["Deaths"])\
                                .background_gradient(cmap='Greens',subset=["Cured"])\
                                .background_gradient(cmap='Purples',subset=["Active"])\
                                .background_gradient(cmap='Greys',subset=["Death Rate (per 100)"])\
                                .background_gradient(cmap='Oranges',subset=["Cure Rate (per 100)"])


        state_cases=state_cases.sort_values('Confirmed', ascending= False).fillna(0)
        state_cases=state_cases.head(15)
        plt.figure(figsize=(14,8))
        sns.barplot(data=state_cases,x='State/UnionTerritory',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')
        sns.barplot(data=state_cases,x='State/UnionTerritory',y='Active',color=sns.color_palette('Set3')[7],label='Active')
        sns.barplot(data=state_cases,x='State/UnionTerritory',y='Cured',color=sns.color_palette('Set3')[8],label='Cured')
        sns.barplot(data=state_cases,x='State/UnionTerritory',y='Deaths',color=sns.color_palette('Set3')[9],label='Deaths')
        plt.xticks(rotation=90)
        plt.legend();
        st.pyplot()

        if st.checkbox('Show Data',False,key=6):
            st.write(state_cases)

    if st.sidebar.checkbox('State wise Analysis',False):
        st.subheader('State Wise Analysis')
        state_analysis()

    def district_barh():
        df3=indiDetails.groupby(['detected_state','detected_district']).count()
        df3.reset_index(inplace=True)
        states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Telangana','Andhra Pradesh',
                    'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
        plt.figure(figsize=(20,60))
        for i,state in enumerate(states_list):
            plt.subplot(8,2,i+1)
            df4=df3[df3['detected_state']==state].sort_values('id',ascending=False)
            df4=df4.head(10)
            sns.barplot(data=df4,x='id',y='detected_district')
            plt.xlabel('Number of Cases')
            plt.ylabel('')
            plt.title(state)
        plt.tight_layout()
        plt.show()
        st.pyplot()

        if st.checkbox('Show Data',False,key=7):
            st.write(df3)


    if st.sidebar.checkbox('State and District Analysis',False):
        st.subheader('State and District Analysis')
        district_barh()

    def district_bar():
        states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Andhra Pradesh',
                    'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
        df5=covid19India[covid19India['Date']>'2020-04-07']
        df5=df5.groupby(['Date','State/UnionTerritory']).sum()
        df5.reset_index(inplace=True)
        df5['Date'] = df5['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        plt.figure(figsize=(20,60))

        for i,state in enumerate(states_list):
            plt.subplot(7,2,i+1)
            df4=df5[df5['State/UnionTerritory']==state]
            plt.bar(df4.Date,df4.Confirmed,label='Confirmed')
            plt.bar(df4.Date,df4.Cured,label='Cured')
            plt.bar(df4.Date,df4.Deaths,label='Death')
            plt.xticks(rotation=60)
            plt.title(state)
            plt.ylabel('Total Cases')
            plt.xlabel('Date')
            plt.legend()
        plt.tight_layout()
        plt.show()
        st.pyplot()

        if st.checkbox('Show Data',False,key=8):
            st.write(df5)


    if st.sidebar.checkbox('State Wise Bar Analysis',False):
        st.subheader('State Wise Bar Analysis')
        district_bar()





    covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
    data=covid19India.groupby(['Date','State/UnionTerritory'])['Confirmed','Cured','Deaths'].sum()
    data.reset_index(inplace=True)
    data['Date']=data['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))

    def icmr():
        state=list(icmrTestLabs['state'].value_counts().index)
        count=list(icmrTestLabs['state'].value_counts())
        plt.figure(figsize=(14,8))
        sns.barplot(x=count,y=state,color=sns.color_palette('Set3')[10])
        plt.xlabel('Counts')
        plt.ylabel('States')
        plt.title('ICMR Test labs per States')
        plt.tight_layout()
        st.pyplot()

        if st.checkbox('Show Data',False,key=9):
            st.write(icmrTestLabs)

    if st.sidebar.checkbox('ICMR Test Labs',False):
        st.subheader('ICMR Test Labs per State')
        icmr()



    def hospital_beds(hospitalBeds):



        plt.figure(figsize=(20,60))
        plt.subplot(4,1,1)
        hospitalBeds=hospitalBeds.sort_values('NumUrbanHospitals_NHP18', ascending= False)
        sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('Pastel2')[0])
        plt.title('Urban Hospitals per states')
        plt.xlabel('Count')
        plt.ylabel('States')
        for i in range(hospitalBeds.shape[0]):
            count = hospitalBeds.iloc[i]['NumUrbanHospitals_NHP18']
            plt.text(count+10,i,count,ha='center',va='center')

        plt.subplot(4,1,2)
        hospitalBeds=hospitalBeds.sort_values('NumRuralHospitals_NHP18', ascending= False)
        sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('Pastel2')[1])
        plt.title('Rural Hospitals per states')
        plt.xlabel('Count')
        plt.ylabel('States')
        for i in range(hospitalBeds.shape[0]):
            count = hospitalBeds.iloc[i]['NumRuralHospitals_NHP18']
            plt.text(count+100,i,count,ha='center',va='center')

        plt.subplot(4,1,3)
        hospitalBeds=hospitalBeds.sort_values('NumUrbanBeds_NHP18', ascending= False)
        sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('Pastel2')[6])
        plt.title('Urban Beds per states')
        plt.xlabel('Count')
        plt.ylabel('States')
        for i in range(hospitalBeds.shape[0]):
            count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']
            plt.text(count+1500,i,count,ha='center',va='center')

        plt.subplot(4,1,4)
        hospitalBeds=hospitalBeds.sort_values('NumRuralBeds_NHP18', ascending= False)
        sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('Pastel2')[7])
        plt.title('Rural Beds per states')
        plt.xlabel('Count')
        plt.ylabel('States')
        for i in range(hospitalBeds.shape[0]):
            count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']
            plt.text(count+1500,i,count,ha='center',va='center')

        plt.show()
        plt.tight_layout()
        st.pyplot()

        if st.checkbox('Show Data',False,key=10):
            st.write(hospitalBeds)

    if st.sidebar.checkbox('Hospitals & Beds Analysis',False):
        st.subheader('Hospital & Beds(Rural & Urban) per State')
        hospital_beds(hospitalBeds)

    def predict_cases(df,days):


        df=df[df['location']=='India']
        df=df.groupby('date')[['new_cases','new_deaths']].sum()
        df=df.reset_index()
        pred_cnfrm = df[["date","new_cases"]]
        pr_data = pred_cnfrm
        pr_data.columns = ['ds','y']
        m=Prophet(daily_seasonality=True)
        m.fit(pr_data)
        future=m.make_future_dataframe(periods=days)
        forecast=m.predict(future)

        if days==1:
            st.header('Prediction of Daily Cases for the next '+str(days)+' day')
        else:
            st.header('Prediction of Daily Cases for the next '+str(days)+' days')

        fig = plot_plotly(m, forecast)

        st.plotly_chart(fig,use_container_width=True)


        fig2 = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
        st.pyplot()

        if st.checkbox('Show Raw Data',False,key=11):
            st.write(df)






    st.sidebar.title("Covid 19 Forecasting using Facebook Prophet Model")
    st.sidebar.text("      ")

    if st.sidebar.checkbox("Daily Cases Forecasting", False):
        st.header("Prediction of Daily Cases")
        days=st.slider('Select the number of days for prediction',min_value=1,max_value=30,key=13)
        predict_cases(df,days)

    def predict_deaths(df,days):

        df=df[df['location']=='India']
        df=df.groupby('date')[['new_cases','new_deaths']].sum()
        df=df.reset_index()
        pred_cnfrm = df[["date","new_deaths"]]
        pr_data = pred_cnfrm
        pr_data.columns = ['ds','y']
        m=Prophet(daily_seasonality=True)
        m.fit(pr_data)
        future=m.make_future_dataframe(periods=days)
        forecast=m.predict(future)

        if days==1:
            st.header('Prediction of Daily Deaths for the next '+str(days)+' day')
        else:
            st.header('Prediction of Daily Deaths for the next '+str(days)+' days')

        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig,use_container_width=True)


        fig2 = m.plot(forecast,xlabel='Date',ylabel='Death Count')
        st.pyplot()

        if st.checkbox('Show Raw Data',False,key=14):
            st.write(df)



    if st.sidebar.checkbox("Daily Deaths Forecasting", False):
        st.header("Prediction of Daily Deaths")
        days=st.slider('Select the number of days for prediction',min_value=1,max_value=30,key=16)
        predict_deaths(df2,days)



    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")
    st.sidebar.text("      ")

    st.sidebar.markdown('Developed by - Mridul Aggarwal')
    st.sidebar.markdown('Email - mridulagarwal04@gmail.com')

if __name__=='__main__':
    main()
