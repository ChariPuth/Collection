#!/usr/bin/env python
# coding: utf-8

# #                    COVID-19 Data Visualization using Python

# The first wave of covid-19 impacted the global economy as the world was never ready for the pandemic. It resulted in a rise in cases, deaths, unemployment , poverty, resulting in an economic slowdown. Here,is an analysis of the spread of Covid-19 cases and the impacts of covid-19 on the economy.

# The dataset used to analyze contains data about:
# 
# > List of all the countries          
# > Confirmed Cases                 
# > Deaths                   
# > Recovered                
# > Active                   
# > New cases                 
# > New deaths              
# > New recovered            
# > Deaths / 100 Cases     
# > Recovered / 100 Cases   
# > Deaths / 100 Recovered  
# > Confirmed cases last week     
# > 1 week change            
# > 1 week % increase 
# > WHO Region

# In[1]:


#Load the dataset
import pandas as pd
data1=pd.read_csv('C:/Users/chari/Downloads/country_wise_latest.csv')
data1.head()


# #### Preprocessing the data

# In[3]:


data1.info()


# The dataset does not have any null values or duplicates.

# In[4]:


# We rename the column Country/Region  as Country
data1.rename(columns={'Country/Region': "Country"}, inplace=True)


# In[5]:


data1.sample(6)


# In[6]:


#Select only necessary columns for the analysis
data=data1.groupby("Country")['Confirmed','Active','Recovered','Deaths'].sum().reset_index()


# In[7]:


data.info()


# ## Exploratory Data Analysis and Data Visualization
# 
# This process is quite long as it is the main part of data analysis. So it can be divided into three steps for our ease:
# 
# a. Ranking countries based on COVID-19 aspects
# 
# b. Time Series on COVID-19 Cases
# 
# c. Classification and Distribution of the cases

# ####  Countries with maximum cases

# In[8]:


top_cases = pd.DataFrame(data.groupby('Country')['Confirmed'].sum().nlargest(10).sort_values(ascending = False))
top_cases


# In[9]:


import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_cases['Confirmed'],
    y=top_cases.index,  # Use the DataFrame index as the y-axis
    orientation='h',
    marker=dict(color=['deepskyblue', 'lightpink']*5),
))

fig.update_layout(
    title='Top 10 Countries with Highest Confirmed COVID-19 Cases',
    xaxis_title='Confirmed Cases',
    yaxis_title='Country',
    yaxis=dict(autorange="reversed"),  # To display the countries from top to bottom
)

# Add the death count values beside the bars
for i, value in enumerate(top_cases['Confirmed']):
    fig.add_annotation(
        x=value,
        y=top_cases.index[i],
        text=f'{value:,.0f}',
        showarrow=True,
        arrowhead=1,
        yshift=10,
    )

fig.show()


# By this we can observe that:
# 
# >US has the highest number of confirmed cases with 4290259 cases
# 
# >Brazil has the second higest number of confirmed cases with 2442375 cases
# 
# >India has the third highest number of confirmed cases with 1480073 cases
# 
# >Russia has the fourth highest number of confirmed cases with 816680 cases
# 
# >South Africa has the fifth highest number of confirmed cases with 452529 cases
# 
# >Mexico has the sixth highest number of confirmed cases with 395489 cases
# 
# >Peru has the seventh highest number of confirmed cases with 389717 cases
# 
# >Chile has the eight highest number of confirmed cases with 347923 cases
# 
# >United Kingdom has the ninth highest number of confirmed cases with 301708 cases
# 
# >Iran has the tenth highest number of confirmed cases with 293606 cases

# ### Interpretation
# Magnitude of the Pandemic: The numbers illustrate the scale of the pandemic's spread across the world. The fact that several countries are reporting millions of confirmed cases underscores the pervasive nature of the virus and its ability to rapidly infect populations.
# 
# Global Health Inequalities: The variation in case counts reveals the inequalities in healthcare systems, resources, and capacity among different nations. Countries with higher confirmed case counts might be struggling with healthcare infrastructure, resources, or challenges in containing the virus.
# 
# Testing and Surveillance: The data highlights the importance of testing and surveillance efforts in understanding the true extent of the pandemic. Countries with extensive testing programs are likely to identify and report a larger number of cases.
# 
# Public Health Measures: The case counts also reflect the effectiveness of public health measures implemented by different countries. Stringent measures such as lockdowns, travel restrictions, and social distancing may have impacted the rate of transmission.
# 
# Behavioral Factors: The response of the public to guidelines, such as mask-wearing, hygiene practices, and social distancing, can impact the spread of the virus and, consequently, the number of confirmed cases.
# 
# Population Density: Countries with higher population densities might experience faster transmission of the virus, leading to higher case counts.
# 
# Timeliness of Response: The timeliness and effectiveness of government responses, including testing, contact tracing, and quarantine measures, can influence the growth rate of cases.
# 
# Vaccination Impact: The case counts might also reflect the status of vaccination campaigns. Countries with successful vaccination efforts may see a decrease in new cases over time.

# ### Countries with maximum deaths

# In[10]:


top_deaths = pd.DataFrame(data.groupby('Country')['Deaths'].sum().nlargest(10).sort_values(ascending = False))
top_deaths


# In[11]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_deaths['Deaths'],
    y=top_deaths.index,  # Use the DataFrame index as the y-axis
    orientation='h',
    marker=dict(color=['lightblue', 'pink']*5),
))

fig.update_layout(
    title='Top 10 Countries with Highest Deaths due to COVID-19',
    xaxis_title='Deaths',
    yaxis_title='Country',
    yaxis=dict(autorange="reversed"),  # To display the countries from top to bottom
)

# Add the death count values beside the bars
for i, value in enumerate(top_deaths['Deaths']):
    fig.add_annotation(
        x=value,
        y=top_deaths.index[i],
        text=f'{value:,.0f}',
        showarrow=True,
        arrowhead=1,
        yshift=10,
    )

fig.show()


# By this we can say that:
# 
# >US has the highest number of death cases recorded with 148011
# 
# >Brazil has the second higest number of death cases recorded with 87618
# 
# >United Kingdom has the third highest number of death cases with 45844
# >Mexico has the fourth highest number of death cases with 44022
# 
# >Italy has the fifth highest number of death cases with 35112
# 
# >India has the sixth highest number of death cases with 33408
# 
# >France has the seventh highest number of death cases with 30212
# 
# >Spain has the eight highest number of death cases with 28432
# 
# >Peru has the ninth highest number of death cases with 18418
# 
# >Iran has the tenth highest number of death cases with 15912

# ### Interpretation
# 
# Severity of Impact: The number of recorded death cases reflects the severity of the pandemic in each country. Countries with higher numbers of deaths have experienced a more significant impact on public health and healthcare systems.
# 
# Healthcare Infrastructure: The differences in the number of death cases can be attributed, in part, to the capacity and quality of healthcare systems. Countries with robust healthcare infrastructure may have been better equipped to provide critical care to those affected by severe COVID-19 cases.
# 
# Government Responses: Government policies and actions play a crucial role in managing the spread of the virus and mitigating its impact. Countries with successful strategies for containing the virus and protecting vulnerable populations may have lower death rates.
# 
# Socioeconomic Factors: Socioeconomic factors, such as population density, access to healthcare, living conditions, and economic resources, can contribute to the variation in death cases between countries.
# 
# Demographic Factors: The age distribution of the population and the prevalence of underlying health conditions can influence the severity of COVID-19 outcomes. Older populations and individuals with pre-existing health conditions are often at higher risk.
# 
# Testing and Reporting: Variations in testing availability and reporting practices can affect the accuracy of death case counts. Differences in testing rates can impact the identification and reporting of COVID-19-related deaths.
# 
# Global Solidarity: The presence of multiple countries on the list underscores the global nature of the pandemic and the interconnectedness of nations in the face of a health crisis.
# 
# Impact on Society: Beyond the numbers, these results represent the collective impact on society, healthcare workers, families, and communities. Each loss has reverberations that extend beyond the immediate health realm.
# 
# Need for Continued Vigilance: The ongoing presence of COVID-19 and the potential for new variants emphasize the need for continued vigilance, public health measures, and vaccination efforts to prevent further loss of life.

# ### Chloropleth map
# 
# A Choropleth map is a type of thematic map used to visualize data by shading or coloring geographic regions
# Using this map we can easily visualize the countries with maximum deaths due to Covid19 in the world.The country with highest deaths is colored darkest in the map.

# In[12]:


import geopandas as gpd
import matplotlib.pyplot as plt


# In[13]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[14]:


merged_data = world.merge(top_deaths, left_on='name', right_index=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

merged_data.plot(column='Deaths', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
for x, y, label in zip(merged_data.geometry.centroid.x, merged_data.geometry.centroid.y, merged_data['name']):
    ax.text(x, y, label, fontsize=8, ha='center', va='center')
plt.title('Max Deaths due to Covid')

ax.set_xticks([]);
ax.set_yticks([]);
ax.set_xticklabels([]);
ax.set_yticklabels([]);

plt.show();


# ### Top 10 Recovered Cases Countries

# In[15]:


recovered_case = pd.DataFrame(data.groupby('Country')['Recovered'].sum().nlargest(10))
recovered_case


#  By this we can say that:
#  
#  >Brazil has the highest no. of recovered cases with 1846641
# 
# >US has the second higest no. of recovered cases with 1325804
# 
# >India has the third highest no. of recovered cases with 951166
# 
# >Russia has the fourth highest no. of recovered cases with 602249
# 
# >Chile has the fifth highest no. of recovered cases with 319954
# 
# >Mexico has the sixth highest no. of recovered cases with 303810
# 
# >South Africa has the seventh highest no. of recovered cases with 274925
# 
# >Peru has the eight highest no. of recovered cases with 272547
# 
# >Iran has the ninth highest no. of recovered cases with 255144
# 
# >Pakistan has the tenth highest no. of recovered cases with 241026  

# ### Interpretation
# 
# Brazil has the highest number of recovered cases: Brazil has reported the highest number of recovered COVID-19 cases among the listed countries, indicating that a significant proportion of individuals who were previously infected with COVID-19 have successfully recovered.
# 
# Variability in recovery: There is variability in the number of recovered cases among the countries listed. This could be due to a variety of factors, including differences in healthcare infrastructure, testing and reporting practices, population size, government policies, and the severity of the pandemic within each country.
# 
# Global impact of the pandemic: The presence of multiple countries on this list indicates that the COVID-19 pandemic has had a significant global impact. Many countries have experienced substantial numbers of individuals recovering from the virus, which is a positive sign in terms of the overall response to the pandemic.
# 
# Differences in healthcare systems: The differences in the number of recovered cases could reflect variations in the effectiveness of healthcare systems in each country. Countries with robust healthcare systems might be better equipped to provide treatment and care to COVID-19 patients, leading to higher recovery rates.
# 
# Continued monitoring: While the numbers presented here provide insight into the progress of recovery, it's important to note that the situation can change rapidly. New cases and recoveries are reported daily, and ongoing monitoring and analysis are necessary to understand the evolving dynamics of the pandemic.
# 
# Influence of public health measures: The recovery numbers could also reflect the impact of public health measures such as lockdowns, social distancing, mask mandates, and vaccination efforts. Countries with effective measures might see higher recovery rates due to reduced transmission.
# 

# ### Choropleth map

# In[17]:


merged_data = world.merge(recovered_case, left_on='name', right_index=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

merged_data.plot(column='Recovered', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
for x, y, label in zip(merged_data.geometry.centroid.x, merged_data.geometry.centroid.y, merged_data['name']):
    ax.text(x, y, label, fontsize=8, ha='center', va='center')
plt.title('Maximum cases of recovery from Covid')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.show()


# ### Time Series on COVID-19 Cases
# 
# To do analysis on Covid data we perform two types of time series analysis:
# >COVID-19 cases Worldwide
# 
# >Most affected countries over time

# In[16]:


import pandas as pd

time_series = pd.read_csv('C:/Users/chari/Downloads/WHO.csv', encoding='ISO-8859-1')
time_series


# In[17]:


# Replace the problematic column name with 'Date_reported'
time_series.rename(columns=lambda x: x.strip("ï»¿"), inplace=True)
time_series.rename(columns={' Date_reported': 'Date_reported'}, inplace=True)
time_series.head()


# In[18]:


time_series['Date_reported'] = pd.to_datetime(time_series['Date_reported'])


# ## EDA for COVID-19 cases worldwide:

# In[19]:


dates = time_series.groupby('Date_reported').sum()
dates.info()


# The above columns are:
# >New_cases: This column represents the number of newly confirmed cases of a specific disease or infection reported in a given time period, usually on a daily basis.
# >Cumulative_cases: This column shows the total number of confirmed cases of the disease since the beginning of the outbreak or the reporting period. 
# >New_deaths: This column indicates the number of newly reported deaths caused by the disease within a specific time period, typically on a daily basis.
# >Cumulative_deaths: This column represents the total number of deaths caused by the disease since the beginning of the outbreak or the reporting period.

# ####  Cumulative cases worldwide:
# 
# The following code produces a time series of cumulative cases worldwide right from the beginning of the outbreak.

# In[20]:


import pandas as pd

# Finding cumulative cases worldwide
cumulative_cases_worldwide = dates['Cumulative_cases'].sum()

print("Cumulative Cases Worldwide:", cumulative_cases_worldwide)


# According to the data  there are 432888466607 cumulative cases of the specific disease worldwide, representing the total number of confirmed cases from the beginning of the outbreak up to the current date or the end of the reporting period.

# In[21]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = dates.index, y = dates['Cumulative_cases'], fill = 'tonexty', line_color = 'blue'))
fig.update_layout(title = 'Cumulative Cases Worldwide')
fig.show()


# With this graph, we can observe the following insights:
# 
# >Overall Case Growth: The slope of the blue line indicates the rate at which new cases are being added to the cumulative count. Steeper slopes suggest rapid case growth, while flatter slopes indicate a slower increase.
# 
# >Disease Spread: If the line has a steep upward trend, it indicates the disease is spreading rapidly, whereas a flatter trend suggests a slower spread.
# 
# >Outbreak Peaks: Sharp peaks in the graph represent periods with a significant increase in new cases, potentially indicating outbreaks or clusters of infections.
# 
# >Disease Dynamics: The shape of the curve and any patterns in the data can provide insights into the dynamics of the disease, such as waves of infection or variations in transmission rates.

# ####  Cumulative death cases worldwide: 
# 
# The following code produces a time series of cumulative deaths worldwide right from the beginning of the outbreak

# In[22]:


import pandas as pd

# Finding cumulative cases worldwide
cumulative_deaths_worldwide = dates['Cumulative_deaths'].sum()

print("Cumulative Deaths Worldwide:",cumulative_deaths_worldwide)


# According to the data  there are 5445797841 cumulative deaths of the specific disease worldwide, representing the total number of confirmed cases from the beginning of the outbreak up to the current date or the end of the reporting period.

# In[23]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = dates.index, y = dates['Cumulative_deaths'], fill = 'tonexty', line_color = 'blue'))
fig.update_layout(title = 'Cumulative Deaths Worldwide')
fig.show()


# The graph showing the trend of cumulative deaths worldwide over time provides several insights:
#     
# >Disease Fatality: The graph shows the cumulative number of deaths due to the disease, indicating the total mortality burden. By observing the slope of the line, one can assess the overall fatality rate of the disease, which is the proportion of confirmed cases that result in death.
# 
# >Disease Impact Over Time: The upward trend of the blue line reflects the continuous increase in the number of deaths over time. Steeper slopes indicate periods with higher death tolls, potentially pointing to more severe phases of the disease outbreak.
# 
# >Outbreak Peaks and Lulls: Sharp peaks in the graph represent periods with a significant increase in the number of deaths, suggesting potential outbreaks or surges in fatalities. On the other hand, flatter sections indicate periods with fewer reported deaths, possibly indicating successful interventions or containment measures.
#     
# >Disease Progression: The shape of the curve may reveal patterns in the disease progression, such as waves of infections leading to corresponding waves of deaths. This can help identify periods of increased transmission and its subsequent impact on mortality.
# 
# >Public Health Response: Changes in the slope or shape of the curve can indicate the effectiveness of public health measures, such as vaccination campaigns, social distancing, or lockdowns. If the curve starts to flatten after implementing interventions, it may indicate successful control measures.
# 
# >Regional Disparities: If available, the graph can highlight differences in mortality rates between countries or regions. It may provide insights into variations in healthcare capacity, access to medical resources, or varying response strategies.
# 
# >Long-Term Impact: The overall upward trajectory demonstrates the long-term impact of the disease on a global scale. It highlights the cumulative toll on human lives and the need for ongoing vigilance and public health measures to combat the disease.

# ## EDA for Most affected countries over time:
# 
# Extracting data of countries USA, Brazil, India, Russia, and Peru respectively as they are highly affected by COVID-19 in the world.

# In[24]:


# USA 
time_series_us = time_series['Country'] == ('United States of America')
time_series_us = time_series[time_series_us]

# Brazil
time_series_brazil = time_series['Country'] == ('Brazil')
time_series_brazil = time_series[time_series_brazil]

# India
time_series_india = time_series['Country'] == ('India')
time_series_india = time_series[time_series_india]

# Russia
time_series_russia = time_series['Country'] == ('Russia')
time_series_russia = time_series[time_series_russia]

# Peru
time_series_peru = time_series['Country'] == ('Peru')
time_series_peru = time_series[time_series_peru]


# #### Most affected Countries’ Cumulative cases over time

# In[27]:


fig = go.Figure()

fig.add_trace(go.Line(x = time_series_us['Date_reported'], y = time_series_us['Cumulative_cases'], name = 'USA'))
fig.add_trace(go.Line(x = time_series_brazil['Date_reported'], y = time_series_brazil['Cumulative_cases'], name = 'Brazil'))
fig.add_trace(go.Line(x = time_series_india['Date_reported'], y = time_series_india['Cumulative_cases'], name = 'India'))
fig.add_trace(go.Line(x = time_series_russia['Date_reported'], y = time_series_russia['Cumulative_cases'], name = 'Russia'))
fig.add_trace(go.Line(x = time_series_peru['Date_reported'], y = time_series_peru['Cumulative_cases'], name = 'Peru'))

fig.update_layout(title = 'Time Series of Most Affected countries, Cumulative Cases',height=700,width=800)


fig.show()


# #### Most affected Countries ,cumulative death cases over time

# In[29]:


fig = go.Figure()

fig.add_trace(go.Line(x = time_series_us['Date_reported'], y = time_series_us['Cumulative_deaths'], name = 'USA'))
fig.add_trace(go.Line(x = time_series_brazil['Date_reported'], y = time_series_brazil['Cumulative_deaths'], name = 'Brazil'))
fig.add_trace(go.Line(x = time_series_india['Date_reported'], y = time_series_india['Cumulative_deaths'], name = 'India'))
fig.add_trace(go.Line(x = time_series_russia['Date_reported'], y = time_series_russia['Cumulative_deaths'], name = 'Russia'))
fig.add_trace(go.Line(x = time_series_peru['Date_reported'], y = time_series_peru['Cumulative_deaths'], name = 'Peru'))

fig.update_layout(title = 'Time Series of Most Affected countries,Cumulative Death Cases',height=700,width=800)

fig.show()


# In[ ]:





# ## Decision tree classifier
# 
# Determines the epidemiological trends i.e analysis of the confirmed cases, deaths, and recovery rates over time to understand the spread and impact of the virus in different countries.
# 
# >This model will predict whether a country is at "high risk" or "low risk" of COVID-19 spread based on the number of new cases reported in that country.

# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df=pd.read_csv('C:/Users/chari/Downloads/country_wise_latest.csv')
print(df.columns)


# In[31]:


# Drop unnecessary columns for modeling
df.drop(['New deaths', 'New recovered', 'Deaths / 100 Cases','Recovered / 100 Cases', 'Deaths / 100 Recovered','Confirmed last week', '1 week change', '1 week % increase', 'WHO Region'], axis=1, inplace=True)


# In[32]:


# Define a binary variable 'high_risk' based on new cases threshold (we assume it to be 1000)
threshold = 1000
df['high_risk'] = df['New cases'] >= threshold


# In[34]:


# Encode 'high_risk' as 1 and 'low_risk' as 0
df['high_risk'] = df['high_risk'].astype(int)


# In[35]:


# Split data into features (X) and target (y)
X = df[['New cases']]
y = df['high_risk']


# In[36]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Create and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[38]:


# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and show the classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')


# An accuracy of 0.97 (97%) indicates that the Decision Tree Classifier is performing well in predicting the risk category (high-risk or low-risk) based on the number of new COVID-19 cases.

# In[39]:


print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')


# The confusion matrix shows the following:
# 
# >True Positives(TP): 5 'high_risk' countries were correctly predicted as 'high_risk'.
# 
# >True Negatives(TN): 32 'low_risk' countries were correctly predicted as 'low_risk'.
# 
# >False Positives(FP): 1 'low_risk' country was incorrectly predicted as 'high_risk'.
# 
# >False Negatives(FN): 0 'high_risk' countries were incorrectly predicted as 'low_risk'.

# In[42]:


# Plot the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the Decision Tree
plt.figure(figsize=(8, 8))
plot_tree(clf, filled=True, feature_names=['New_cases'], class_names=['low_risk', 'high_risk'])
plt.show()


# >Gini Impurity (gini): Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly classified. Gini impurity is used to determine how well a particular split separates the classes. Lower values are preferred.
# 
# >Samples: Indicates the number of samples (data points) in that node.
# 
# >Value: This shows the number of samples from each class in that node. 
# For example [27, 1] means there are 27 samples of class 0 ('low_risk') and 1 sample of class 1 ('high_risk').
# 
# >Class: The "class" in each leaf node indicates the class prediction for that node. For example, [27, 1] indicates that the class prediction is 'low_risk' because there are more samples of that class.

# ## Hypothesis testing 
# 
# Reasons to do hypothesis testing with this dataset:
# 
# >Comparing Groups: You might want to compare different groups of countries or regions to determine if there are statistically significant differences in COVID-19 metrics (e.g., confirmed cases, deaths, recovered cases) between them. This can provide insights into how the pandemic has affected different parts of the world.
# 
# >Assessing Interventions: Hypothesis testing can help assess the effectiveness of interventions or public health measures. For instance, you could test whether the implementation of certain measures has led to a statistically significant decrease in the number of new cases.
# 
# >Understanding Trends: You could analyze trends over time, such as changes in confirmed cases or death rates between different weeks. Hypothesis testing can help determine if these changes are statistically significant.
# 
# >Comparing Regions: If you have data from different regions within a country, you could test whether there are significant differences in COVID-19 metrics among those regions. This can aid in understanding localized impacts.
# 
# >Supporting Decision-Making: Hypothesis testing provides a formal way to support decisions with data-driven evidence. For example, it can help determine if the differences observed in the dataset are likely due to chance or if they represent real differences.
# 
# >Drawing Generalizations: By conducting hypothesis tests on a sample of data, you can make inferences about the broader population. This can be useful for drawing generalizations about how COVID-19 has affected different regions or groups.

# In[44]:


import pandas as pd
from scipy import stats

# Load the COVID-19 dataset into a pandas DataFrame
# Assume the data is stored in a CSV file named 'covid_data.csv'
data = pd.read_csv('C:/Users/chari/Downloads/country_wise_latest.csv')

# Filter data for "Americas" and "Europe" regions
americas_data = data[data['WHO Region'] == 'Americas']['Confirmed']
europe_data = data[data['WHO Region'] == 'Europe']['Confirmed']

# Set up hypotheses
# H0: Mean number of confirmed cases is the same for both regions
# Ha: Mean number of confirmed cases is different for the two regions
alpha = 0.05

# Perform two-sample t-test (assuming equal variances)
t_statistic, p_value = stats.ttest_ind(americas_data, europe_data, equal_var=True)

# Compare p-value with alpha
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")

print("p-value:", p_value)


# ### Interpretation
# 
# >Fail to Reject the Null Hypothesis:
# This statement suggests that based on the analysis you performed, there isn't enough statistical evidence to conclude that there is a significant difference in the mean number of confirmed COVID-19 cases between the "Americas" and "Europe" regions. In other words, the data you have collected doesn't strongly support the idea that the mean case counts in these two regions are meaningfully different from each other.
# 
# >No Significant Difference:
# The analysis suggests that any observed differences in the mean case counts between the regions could reasonably be attributed to random chance or sampling variability. This means that the data doesn't provide strong evidence to support the claim that there is a genuine, non-random difference in case counts between the "Americas" and "Europe."
# 
# >P-value: 0.0843566487727856:
# The p-value is a measure of the strength of evidence against the null hypothesis. In this case, the p-value is 0.0843566487727856. This value is larger than the common significance level of 0.05. The p-value indicates the probability of observing data as extreme as what you have collected if the null hypothesis (no significant difference) is true. A higher p-value suggests that the data is more consistent with the null hypothesis.
# 
# > The data does not provide enough evidence to confidently conclude that there is a significant difference in the mean number of confirmed COVID-19 cases between the "Americas" and "Europe" regions. This could be due to various factors such as data variability, sample size, or other unaccounted factors
