#Main libraries
import geopandas
import numpy as np
import pandas as pd
import seaborn as sb
import streamlit as sl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sl.title('Global Trends in Mental Health Disorders')
data_df = pd.read_csv('Mental health Depression disorder Data.csv')
url='https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder'
sl.write('Data source: [click link](' + url + ')')

def convert_df(df):
    return data_df.to_csv()#.encode('utf-8')
csv = convert_df(data_df)
sl.download_button('Download raw data', csv)

###DATA CLEANING###
#Table division
data1=data_df[:6468] 
data2=data_df[6468:54276]
data3=data_df[54276:102084]
data4=data_df[102084:]

#Second table
new_header = data2.iloc[0] 
data2.columns = new_header
data2=data2[1:]
data2=data2[['Entity','Code','Year','Prevalence in males (%)','Prevalence in females (%)','Population']]

#Third table
new_header = data3.iloc[0] 
data3.columns = new_header
data3=data3[1:]
data3=data3[['Entity','Code','Year','Suicide rate (deaths per 100,000 individuals)','Depressive disorder rates (number suffering per 100,000)','Population']]

#Forth table
new_header = data4.iloc[0] #grab the first row for the header
data4.columns = new_header
data4=data4[1:]
data4=data4[['Entity','Code','Year','Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number) (people suffering from depression)']]

#Null-values handling, change of data type and column name change
countries=pd.read_csv('countries_codes_and_coordinates.csv')
countries=countries['Alpha-3 code']
countries=countries.map(lambda x:x.replace('"',''))
countries=countries.map(lambda x:x.replace(' ',''))

data1=data1.loc[data1['Code'].isin(countries)] #First table

data1['Schizophrenia (%)'] = pd.to_numeric(data1['Schizophrenia (%)'])
data1['Bipolar disorder (%)'] = pd.to_numeric(data1['Bipolar disorder (%)'])
data1['Eating disorders (%)'] = pd.to_numeric(data1['Eating disorders (%)'])
data1['Year'] = pd.to_numeric(data1['Year'])

data1=data1.drop('index', axis=1)
data1.index = range(len(data1))
data1.rename(columns={"Entity": "Country"}, inplace=True)

countries2=data1['Country']

data2=data2.loc[data2['Entity'].isin(countries2)] #Second table
range_years=list(data2[data2['Prevalence in males (%)'].notnull()]['Year'].unique())
data2=data2[data2.Year.str.contains('|'.join(range_years))]

data2['Prevalence in males (%)'] = pd.to_numeric(data2['Prevalence in males (%)'])
data2['Prevalence in females (%)'] = pd.to_numeric(data2['Prevalence in females (%)'])
data2['Population'] = pd.to_numeric(data2['Population'])
data2['Year'] = pd.to_numeric(data2['Year'])

data2.index = range(len(data2))
data2.rename(columns={"Entity": "Country"}, inplace=True)

data3=data3.loc[data3['Entity'].isin(countries2)] #Third table
data3=data3[data3.Year.str.contains('|'.join(range_years))]

data3['Suicide rate (deaths per 100,000 individuals)'] = pd.to_numeric(data3['Suicide rate (deaths per 100,000 individuals)'])
data3['Depressive disorder rates (number suffering per 100,000)'] = pd.to_numeric(data3['Depressive disorder rates (number suffering per 100,000)'])
data3['Population'] = pd.to_numeric(data3['Population'])
data3['Year'] = pd.to_numeric(data3['Year'])

data3.index = range(len(data3))
data3.rename(columns={"Entity": "Country"}, inplace=True)

data4=data4.loc[data4['Entity'].isin(countries2)] #Forth table
data4['Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number) (people suffering from depression)'] = pd.to_numeric(data4['Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number) (people suffering from depression)'])
data4['Year'] = pd.to_numeric(data4['Year'])

data4.index = range(len(data4))
data4.rename(columns={"Entity": "Country"}, inplace=True)
data4.rename(columns={"Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number) (people suffering from depression)": "People suffering from depression"}, inplace=True)

#Merge of the tables
data12=pd.merge(data1, data2)
data34=pd.merge(data3, data4)
data=pd.merge(data12, data34)

### DATA VISUALIZATION ###
if sl.checkbox('Show raw data'):
    sl.write('Raw data:')
    sl.write(data_df)
sl.write('Cleaned data:')
sl.write(data)
sl.caption('The data are collected between 1990 and 2017.')

if sl.checkbox('Show column descriptions'):    
    sl.write("- **Country**: country to which the rates are associated")
    sl.write("- **Code**: country/region code to which the rates are associated")
    sl.write("- **Year**: year in which the rates are calculated")
    sl.write("- **Schizophrenia (%)**: percentage of people in a specified country/region and year suffering from schizophrenia. \n*Schizophrenia is a mental disorder characterized by continuous or relapsing episodes of psychosis.*") 
    sl.write("- **Bipolar disorder (%)**: percentage of people in a specified country/region and year suffering from bipolar disorder. \n*Bipolar disorder is a mental disorder characterized by periods of depression and periods of abnormally elevated mood that last from days to weeks each.*") 
    sl.write("- **Eating disorders (%)**: percentage of people in a specified country/region and year suffering from eating disorders. \n*Eating disorders are a cluster of mental disorders defined by abnormal eating behaviors that negatively affect a person's physical or mental health.*") 
    sl.write("- **Anxiety disorders (%)**: percentage of people in a specified country/region and year suffering from anxiety disorders. \n*Anxiety disorders are a cluster of mental disorders characterized by significant and uncontrollable feelings of anxiety and fear such that a person's social, occupational, and personal function are significantly impaired.*")
    sl.write("- **Drug use disorders (%)**: percentage of people in a specified country/region and year suffering from drug use disorders. \n*Drug use disorder is the persistent use of drugs despite substantial harm and adverse consequences as a result of their use, characterized by an array of mental/emotional, physical, and behavioral problems.*") 
    sl.write("- **Depression (%)**: percentage of people in a specified country/region and year suffering from depression. \n*Depression is a mental and behavioral disorder characterized by low mood and aversion to activity.*") 
    sl.write("- **Alcohol use disorders (%)**: percentage of people in a specified country/region and year suffering from alcohol use disorders (%). \n*Alcohol use disorder, or alcoholism, is any drinking of alcohol that results in significant mental or physical health problems. It's similar to drug use disorder.*") 
    sl.write("- **Prevalence in males (%)**: percentage of males in a specified country and year suffering from mental disorders compared to the total of the population")
    sl.write("- **Prevalence in females (%)**: percentage of females in a specified country and year suffering from mental disorders compared to the total of the population")
    sl.write("- **Population**: total of the population in a specified country and year")
    sl.write("- **Suicide rate (deaths per 100,000 individuals)**: number of people in a specified country and year who committed suicide on a total of 100000 individuals")
    sl.write("- **Depressive disorder rates (number suffering per 100,000)**: number of people in a specified country and year who suffer from depression on a total of 100000 individuals")
    sl.write("- **People suffering from depression**: number of people in a specified country and year who suffer from depression")

#Data description
sl.subheader('Some statistics')
sl.write(data.loc[:,data.columns!='Year'].describe().T)

equal_to_max_mask = data['Anxiety disorders (%)'] == data['Anxiety disorders (%)'].max()
max_anxiety=data[equal_to_max_mask] 

equal_to_max_mask = data['Depression (%)'] == data['Depression (%)'].max()
max_depression=data[equal_to_max_mask] 
col1, col2 = sl.columns(2)
col1.write('Maximum value for anxiety disorders')
col1.metric(max_anxiety['Country'].item(), max_anxiety['Anxiety disorders (%)'])
col2.write('Maximum value for depression')
col2.metric(max_depression['Country'].item(),max_depression['Depression (%)'])
data_for_graphs=data1[['Schizophrenia (%)','Bipolar disorder (%)','Eating disorders (%)','Anxiety disorders (%)','Drug use disorders (%)','Depression (%)','Alcohol use disorders (%)']]

#Correlation (matrix and heatmap)
sl.subheader('Correlation between mental disorders')
sl.write(data1.loc[:,data1.columns!='Year'].corr())
sl.caption('Correlation matrix')
fig=plt.figure(figsize=(8,6))
sb.heatmap(data_for_graphs.corr(), cmap=sb.color_palette("crest", as_cmap=True),annot=True)
sl.write(fig)
sl.caption('Heatmap')

#Bar plot for average prevalences
data_mean = data1.groupby('Country').mean()
sl.subheader('Prevalence of mental health disorders (%) in space and time')
data_for_bar=data_for_graphs.mean().sort_values(ascending=False).reset_index() 
fig=plt.figure(figsize=(20, 5))
plt.bar(data_for_bar['index'], data_for_bar[0], color=sb.color_palette("mako",7))
plt.title('Average percentage of each mental disorder')
plt.ylabel('(%)')
plt.show()
sl.write('Average percentage of each mental disorder')
sl.write(fig)

#Distribution for prevalences
sl.write('Distribution of mental disorder rates')
option=sl.selectbox('Pick one mental disorder:', ['Alcohol use disorders','Anxiety disorders','Bipolar disorder','Depression','Drug use disorders','Eating disorders','Schizophrenia'])
fig=plt.figure(figsize=(30,10))
plt.title(f"{option} (%)")
if (option=='Alcohol use disorders'):
    sb.distplot(data_for_graphs['Alcohol use disorders (%)'], color=sb.color_palette('mako')[1])
elif (option=='Anxiety disorders'):
    sb.distplot(data_for_graphs['Anxiety disorders (%)'], color=sb.color_palette('mako')[1])
elif (option=='Bipolar disorder'):
    sb.distplot(data_for_graphs['Bipolar disorder (%)'], color=sb.color_palette('mako')[1])
elif (option=='Depression'):
    sb.distplot(data_for_graphs['Depression (%)'], color=sb.color_palette('mako')[1])
elif (option=='Drug use disorders'):
    sb.distplot(data_for_graphs['Drug use disorders (%)'], color=sb.color_palette('mako')[1])
elif (option=='Eating disorders'):
    sb.distplot(data_for_graphs['Eating disorders (%)'], color=sb.color_palette('mako')[1])
elif (option=='Schizophrenia'):
    sb.distplot(data_for_graphs['Schizophrenia (%)'], color=sb.color_palette('mako')[1])
plt.show()
sl.write(fig)

#Relationship between variables 
data_for_scatter=data.query('Year == 2017')
data_for_scatter=data_for_scatter[['Schizophrenia (%)','Bipolar disorder (%)','Eating disorders (%)','Anxiety disorders (%)','Drug use disorders (%)','Depression (%)','Alcohol use disorders (%)']]
fig=sb.pairplot(data_for_scatter, height=2, aspect=2)
plt.show()
sl.write('Relationship between mental disorder prevalences (for 2017)')
sl.pyplot(fig.figure) 

#Differences between countries
italy=data[data['Country']=='Italy']
NZ=data[data['Country']=='New Zealand']
Norway=data[data['Country']=='Norway']
Greenland=data[data['Country']=='Greenland']
UK=data[data['Country']=='United Kingdom']
USA=data[data['Country']=='United States']
Albania=data[data['Country']=='Albania']
morocco=data[data['Country']=='Morocco']
japan=data[data['Country']=='Japan']
Lesotho=data[data['Country']=='Lesotho']

sl.write('Trends of anxiety and depression prevalences in different countries') 
option=sl.selectbox('Pick one country (or ALL for comparison of all countries together):', ['ALL', 'Albania', 'Greenland', 'Italy', 'Japan', 'Lesotho', 'Morocco', 'New Zealand', 'Norway', 'UK', 'USA'])
col1, col2=sl.columns(2)
fig=plt.figure(figsize=(30, 10)) 
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name
if (option=='Albania'):
    plt.plot(Albania['Year'], Albania['Anxiety disorders (%)'], label='Albania')
elif (option=='Italy'):
    plt.plot(italy['Year'], italy['Anxiety disorders (%)'], label='Italy')
elif (option=='UK'):
    plt.plot(UK['Year'], UK['Anxiety disorders (%)'], label='UK')
elif (option=='NZ'):
    plt.plot(NZ['Year'], NZ['Anxiety disorders (%)'], label='New Zeleand')
elif (option=='Norway'):
    plt.plot(Norway['Year'], Norway['Anxiety disorders (%)'], label='Norway')
elif (option=='Greenland'):
    plt.plot(Greenland['Year'], Greenland['Anxiety disorders (%)'], label='Greenland')
elif (option=='USA'):
    plt.plot(USA['Year'], USA['Anxiety disorders (%)'], label='USA')
elif (option=='Albania'):
    plt.plot(Albania['Year'], Albania['Anxiety disorders (%)'], label='Albania')
elif (option=='Morocco'):
    plt.plot(morocco['Year'], morocco['Anxiety disorders (%)'], label='Morocco')
elif (option=='Japan'):
    plt.plot(japan['Year'], japan['Anxiety disorders (%)'], label='Japan')
elif (option=='Lesotho'):
    plt.plot(Lesotho['Year'], Lesotho['Anxiety disorders (%)'], label='Lesotho')
elif (option=='ALL'):
    plt.plot(italy['Year'], italy['Depression (%)'], label='Italy')
    plt.plot(UK['Year'], UK['Depression (%)'], label='UK')
    plt.plot(NZ['Year'], NZ['Depression (%)'], label='New Zeleand')
    plt.plot(Norway['Year'], Norway['Depression (%)'], label='Norway')
    plt.plot(Greenland['Year'], Greenland['Depression (%)'], label='Greenland')
    plt.plot(USA['Year'], USA['Depression (%)'], label='USA')
    plt.plot(Albania['Year'], Albania['Depression (%)'], label='Albania')
    plt.plot(morocco['Year'], morocco['Depression (%)'], label='Morocco')
    plt.plot(japan['Year'], japan['Depression (%)'], label='Japan')
    plt.plot(Lesotho['Year'], Lesotho['Depression (%)'], label='Lesotho')
plt.xlabel('Years')
plt.ylabel('Anxiety disorders rate (%)')
plt.legend()
plt.show()
col1.write('Anxiety:')
col1.write(fig)

fig=plt.figure(figsize=(30, 10)) 
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name
if (option=='Albania'):
    plt.plot(Albania['Year'], Albania['Depression (%)'], label='Albania')
elif (option=='Italy'):
    plt.plot(italy['Year'], italy['Depression (%)'], label='Italy')
elif (option=='UK'):
    plt.plot(UK['Year'], UK['Depression (%)'], label='UK')
elif (option=='NZ'):
    plt.plot(NZ['Year'], NZ['Depression (%)'], label='New Zeleand')
elif (option=='Norway'):
    plt.plot(Norway['Year'], Norway['Depression (%)'], label='Norway')
elif (option=='Greenland'):
    plt.plot(Greenland['Year'], Greenland['Depression (%)'], label='Greenland')
elif (option=='USA'):
    plt.plot(USA['Year'], USA['Depression (%)'], label='USA')
elif (option=='Albania'):
    plt.plot(Albania['Year'], Albania['Depression (%)'], label='Albania')
elif (option=='Morocco'):
    plt.plot(morocco['Year'], morocco['Depression (%)'], label='Morocco')
elif (option=='Japan'):
    plt.plot(japan['Year'], japan['Depression (%)'], label='Japan')
elif (option=='Lesotho'):
    plt.plot(Lesotho['Year'], Lesotho['Depression (%)'], label='Lesotho')
elif (option=='ALL'):
    plt.plot(italy['Year'], italy['Depression (%)'], label='Italy')
    plt.plot(UK['Year'], UK['Depression (%)'], label='UK')
    plt.plot(NZ['Year'], NZ['Depression (%)'], label='New Zeleand')
    plt.plot(Norway['Year'], Norway['Depression (%)'], label='Norway')
    plt.plot(Greenland['Year'], Greenland['Depression (%)'], label='Greenland')
    plt.plot(USA['Year'], USA['Depression (%)'], label='USA')
    plt.plot(Albania['Year'], Albania['Depression (%)'], label='Albania')
    plt.plot(morocco['Year'], morocco['Depression (%)'], label='Morocco')
    plt.plot(japan['Year'], japan['Depression (%)'], label='Japan')
    plt.plot(Lesotho['Year'], Lesotho['Depression (%)'], label='Lesotho')
plt.xlabel('Years')
plt.ylabel('Depression (%)')
plt.legend()
plt.show()
col2.write('Depression:')
col2.write(fig)

#Geographic maps
sl.write('Average values for prevalence of anxiety and depression in the world')
countries=pd.read_csv('countries_codes_and_coordinates.csv')
countries=countries[['Alpha-3 code','Latitude (average)','Longitude (average)']]
for i in range(len(countries.iloc[0])):
    countries.iloc[:,i]=countries.iloc[:,i].map(lambda x:x.replace('"',''))
    countries.iloc[:,i]=countries.iloc[:,i].map(lambda x:x.replace(' ',''))
data_mean_with_code = data1.groupby('Code').mean()
merged_data=pd.merge(data_mean_with_code, countries,  right_on='Alpha-3 code', left_on='Code').sort_values(by='Anxiety disorders (%)',ascending=False).reset_index()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.columns=['pop_est', 'continent', 'name', 'Alpha-3 code', 'gdp_md_est', 'geometry']
merged_data2=pd.merge(world,merged_data,on='Alpha-3 code')
plt.title('Anxiety disorders (%)',fontsize=25)
fig=merged_data2.plot(column='Anxiety disorders (%)', #Anxiety
            figsize=(30,10),
            legend=True, cmap='Blues')
plt.title('Anxiety disorders (%)', fontsize=25)
plt.show()
sl.pyplot(fig.figure) 
sl.caption('Anxiety')

merged_data=pd.merge(data_mean_with_code, countries,  right_on='Alpha-3 code', left_on='Code').sort_values(by='Depression (%)',ascending=False).reset_index()
merged_data2=pd.merge(world,merged_data,on='Alpha-3 code')
fig=merged_data2.plot(column='Depression (%)', #Depression
            figsize=(30,10),
            legend=True, 
            cmap='Blues')
plt.title('Depression (%)', fontsize=25)
plt.show()
sl.pyplot(fig.figure) 
sl.caption('Depression')


#Prevalences of males and females
sl.subheader('Comparison of prevalence of mental disorders between males and females')
data_for_bar=data[['Prevalence in males (%)','Prevalence in females (%)']].mean().sort_values(ascending=False).reset_index() 
fig=plt.figure(figsize=(10, 10))
plt.bar(['Females','Males'], data_for_bar[0], color=['pink', 'blue'])
plt.title('Prevalences of males and females suffering from mental disorders')
plt.ylabel('(%)')
plt.show()
sl.write('Average values of prevalence of mental disorders for males and females')
sl.write(fig)

data_groupby=data.groupby('Year').mean()
data_gap_MF=data_groupby[['Prevalence in males (%)','Prevalence in females (%)']]
data_gap_MF['Gap']=data_gap_MF['Prevalence in females (%)']-data_gap_MF['Prevalence in males (%)']
option=sl.slider('Pick a year:', 1990, 2017)
sl.write('Gap between males and females in a selected year')
sl.metric('(Average) Female prevalence (%) - (average) male prevalence (%)', '{:.3f}'.format(data_gap_MF['Gap'][option]))

#Trends of prevalence in some countries for males and females
Italy=data[data['Country']=='Italy']
NZ=data[data['Country']=='New Zealand']
Norway=data[data['Country']=='Norway']
Greenland=data[data['Country']=='Greenland']
UK=data[data['Country']=='United Kingdom']
USA=data[data['Country']=='United States']
Albania=data[data['Country']=='Albania']
Morocco=data[data['Country']=='Morocco']
Japan=data[data['Country']=='Japan']
Lesotho=data[data['Country']=='Lesotho']

list_of_countries=[Italy, NZ, Norway, Greenland, UK, USA, Albania, Morocco, Japan, Lesotho]
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

fig_1=plt.figure(figsize=(30,30))
plt.title('Comparison of difference between prevalences of males and females in some countries')
for i in range(len(list_of_countries)):
    ax=f"ax_{'i'}"
    ax=fig_1.add_subplot(5,2,i+1)
    ax.plot(np.array(list_of_countries[i]['Year']), np.array(list_of_countries[i]['Prevalence in males (%)']), label='Males', color='blue')
    ax.plot(np.array(list_of_countries[i]['Year']), np.array(list_of_countries[i]['Prevalence in females (%)']), label='Females', color='pink')
    ax.legend()
    plt.title(f"{get_df_name(list_of_countries[i])}")
sl.write('Trend of prevalence of mental disorders between males and females in some countries')
sl.write(fig_1)

#Suicide rate vs. depression rate
sl.subheader('Relationship between suicide and depression rates')
data_suicide_depr=data[['Suicide rate (deaths per 100,000 individuals)','Depressive disorder rates (number suffering per 100,000)']]

sl.caption('Association between suicide and depression rate')
fig=plt.figure(figsize=(30,10)) #General
plt.title('Association between suicide rate (%) and depression rate (%)')
plt.scatter(data_suicide_depr['Suicide rate (deaths per 100,000 individuals)'], data_suicide_depr['Depressive disorder rates (number suffering per 100,000)'], cmap='Blues')
plt.xlabel('Suicide rate')
plt.ylabel('Depression rate')
plt.show()
sl.write(fig)
sl.caption('General')

fig_1=plt.figure(figsize=(30,30)) #In different countries
plt.title('Association between suicide rates (%) and depressive rates (%) in different countries')
for i in range(len(list_of_countries)):
    ax=f"ax_{'i'}"
    ax=fig_1.add_subplot(5,2,i+1)
    ax.scatter(np.array(list_of_countries[i]['Suicide rate (deaths per 100,000 individuals)']), np.array(list_of_countries[i]['Depressive disorder rates (number suffering per 100,000)']))
    plt.xlabel('Suicide')
    plt.ylabel('Depression')
    plt.title(f"{get_df_name(list_of_countries[i])}")
sl.write(fig_1)
sl.caption('In different countries')

### DATA MODELING ###
sl.subheader('Data modeling')
model = LinearRegression()

#Suicide vs. Depression rate
for i in range(len(list_of_countries)):
    x_train=np.array(list_of_countries[i]['Suicide rate (deaths per 100,000 individuals)'])
    y_train=np.array(list_of_countries[i]['Depressive disorder rates (number suffering per 100,000)'])
    x_train = x_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    model=model.fit(x_train, y_train)

fig_1=plt.figure(figsize=(30,30)) #Linear
plt.title('Association between suicide rates (%) and depressive rates (%) in different countries')
for i in range(len(list_of_countries)):
    x_train=np.array(list_of_countries[i]['Suicide rate (deaths per 100,000 individuals)'])
    y_train=np.array(list_of_countries[i]['Depressive disorder rates (number suffering per 100,000)'])
    x_train = x_train.reshape(-1, 1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    ax=f"ax_{'i'}"
    ax=fig_1.add_subplot(5,2,i+1)
    ax.scatter(x_train, y_train, c='blue')
    ax.plot(x_train, y_pred, c='grey')
    plt.xlabel('Suicide')
    plt.ylabel('Depression')
    plt.title(f"{get_df_name(list_of_countries[i])}")
sl.write('Regression model: Association between suicide and depression rates in different countries') 
sl.write(fig_1)

#Prediction of prevalence
fig_1=plt.figure(figsize=(30,10))
plt.title('Prediction of anxiety disorders (%) after 2017')
model = LinearRegression()
y_train=np.array(UK[UK['Year']<2017]['Anxiety disorders (%)'])
y_test=np.array(UK[UK['Year']==2017]['Anxiety disorders (%)'])
x_train=np.array(UK[UK['Year']<2017]['Year'])
x_test=np.array([2017,2018,2019,2020,2021,2022])
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test.reshape(-1, 1))
plt.scatter(x_train, y_train, c='blue')
plt.scatter(x_test, y_pred, c='red')
plt.plot(x_test, y_pred, c='grey')
plt.xlabel('Year')
plt.ylabel('Anxiety (%)')
sl.write('Prediction of anxiety disorders (%) after 2017 in UK')
sl.write(fig_1)
sl.caption('Linear graph')
sl.write(str(y_pred))
sl.caption('Values')

#Clustering
from sklearn.cluster import KMeans
square_distances = []
x = data[['Anxiety disorders (%)', 'Depression (%)']]
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x)
    square_distances.append(km.inertia_)

n_clstr=4
km = KMeans(n_clusters=n_clstr, random_state=42)#.fit(x)
y_pred = km.fit(x)
cluster_map = data
cluster_map['data_index'] = x.index.values
cluster_map['cluster'] = km.labels_

fig=plt.figure(figsize=(25,15))
for i in range(n_clstr):
    plt.scatter(cluster_map[cluster_map['cluster']==i]['Anxiety disorders (%)'], cluster_map[cluster_map['cluster']==i]['Depression (%)'], label=i)
plt.xlabel('Anxiety disorders (%)')
plt.ylabel('Depression (%)')
plt.legend()
plt.show()
sl.write('Groups of countries')
sl.write(fig)

col1, col2, col3, col4= sl.columns(4)
with col1:
    sl.write('Cluster 1')
    sl.write(cluster_map[cluster_map['cluster']==0]['Country'].unique())
with col2:
    sl.write('Cluster 2')
    sl.write(cluster_map[cluster_map['cluster']==1]['Country'].unique())
with col3:
    sl.write('Cluster 3')
    sl.write(cluster_map[cluster_map['cluster']==2]['Country'].unique())
with col4:
    sl.write('Cluster 4')
    sl.write(cluster_map[cluster_map['cluster']==3]['Country'].unique())
sl.caption('Countries in each cluster')



