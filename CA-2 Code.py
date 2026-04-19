import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


# Data Preprocessing:
    
data = pd.read_csv(r"D:\B.tech\Sem 4\Python Liberary\CA2-Project\Final Sheet.csv")
print(data.info())
data["Traffic Control Presence"] = data["Traffic Control Presence"].fillna('No Control')
data["Driver License Status"] = data["Driver License Status"].fillna('No License')
print(data.info())



# (1.)Foundational Insights: Visual Data Exploration & Relationships
# Focus - Summary Statistics, Correlation, and Matplotlib.

print(data[['Number of Casualties', 'Number of Fatalities']].describe())

# Correlation Heatmap
cm = data[['Number of Casualties','Number of Fatalities','Speed Limit (km/h)','Driver Age']].corr()

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation Heatmap: Road safety Variables")
plt.show()

#Alcohol Pie Chart
alcohol_counts = data['Alcohol Involvement'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(alcohol_counts, labels=alcohol_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue','lightpink'], explode=(0.05, 0))
plt.title('Percentage of Total Accidents Involving Alcohol')
plt.show()

# Top 10 States
plt.figure(figsize=(12,6))
data['State Name'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title('Top 10 states by Number of Road Accidents')
plt.xlabel('State Name')
plt.ylabel('Accident Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# (2.)Data Profiling: Probability Distributions & Outlier Analysis
# Focus: Shapiro-Wilk Test, Normal Distribution, and Anomaly Detection.

#Shapiro-Wilk Test
stat, p_value = stats.shapiro(data['Driver Age'].head(500))
print(f"Shapiro-Wilk Test (Driver Age): Statistics={stat:.3f},p-value={p_value:.3f}")

if p_value > 0.05:
    print("Data looks Normally Distributed (Fail to reject Null Hypothesis)")
else:
    print("Data does not look normally distributed (Reject Null Hypothesis)")
    
# Outlier Detection:
Q1 = data['Number of Casualties'].quantile(0.25)
Q3 = data['Number of Casualties'].quantile(0.75)
IQR = Q3-Q1
outliers = data[(data['Number of Casualties'] < (Q1-1.5*IQR)) | (data['Number of Casualties']>(Q3+1.5*IQR))]
print(f"Number of Outliers detected in Casualties: {len(outliers)}")

# Histogram
plt.figure(figsize=(10,6))
sns.histplot(data['Driver Age'],kde=True,color='skyblue')
plt.title('Distribution of Driver Age with KDE')
plt.xlabel('Driver Age (years)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Casualties Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=data['Number of Casualties'], color='salmon', flierprops={"marker": "x", "markerfacecolor": "red"})
plt.title('Statistical Distribution & Outlier Identification in Casualties', fontsize=14)
plt.xlabel('Number of Casualties', fontsize=12)
#plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




# (3.)Advanced Visuals: Statistical Storytelling with Seaborn
# Focus: Advanced Seaborn plots and Customization.

# Weather vs Severity Countplot
#sns.set_theme(style="ticks")
plt.figure(figsize=(12,6))
sns.countplot(data=data,x='Weather Conditions',hue='Accident Severity', palette="viridis")
plt.title('Accident count by weather condition and Severity')
plt.xlabel('Weather Condition')
plt.ylabel('Total number of Accidents')
plt.legend(title='Severity', loc='upper right')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
avg_cas = data.groupby('Road Type')['Number of Casualties'].mean().sort_values()
ax = avg_cas.plot(kind='barh', color='teal')
plt.title('Infrastructure Risk: Avg Casualties by Road Type')
plt.xlabel('Average Number of Casualties')
for i, v in enumerate(avg_cas):
    ax.text(v+0.1,i,f'{v:.2f}',va='center',fontsize=10)
plt.tight_layout()
plt.show()

# Weather vs Road Condition Heatmap
pivot_risk = data.pivot_table(index='Weather Conditions', columns='Road Condition', values='Number of Casualties', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_risk, annot=True, cmap='YlOrRd')
plt.title('Synergistic Risk: Weather vs Road Condition (Avg Casualties)')
plt.tight_layout()
plt.show()




# (4.)Proving the Facts: Significance Testing & Safety Hypotheses
# Focus: T-Tests, Z-Tests, and Chi-Square Tests.

# Chi-Square Test
contingency_table = pd.crosstab(data['Alcohol Involvement'], data['Accident Severity'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Test (Alcohol vs Severity): p-value = {p_chi:.4f}")
if p_chi < 0.05:
    print("Alcohol has a statistically significant impact on severity.")
else:
    print("Alcohol does not has a significant impact on severity.")

# T-Test
fatal_speed = data[data['Accident Severity'] == 'Fatal']['Speed Limit (km/h)']
minor_speed = data[data['Accident Severity'] == 'Minor']['Speed Limit (km/h)']
t_stat, p_ttest = stats.ttest_ind(fatal_speed.dropna(), minor_speed.dropna())
print(f"T-Test (Speed of Fatal vs Minor): p-value = {p_ttest:.4f}")
if p_ttest < 0.05:
    print("Speed significantly differs between Fatal and Minor accidents.")
else:
    print("Speed does not differs between Fatal and Minor accidents.")
   
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Alcohol Involvement', y='Number of Casualties',hue='Alcohol Involvement', palette='Set2',legend=False)
plt.title('Statistical Variance: Casualties by Alcohol Involvement')
plt.xlabel('Alcohol Involvement')
plt.ylabel('Number of Casualties')
plt.tight_layout()
plt.show()



# (5.)Smart Diagnostics: Feature Quality & Experimental Design
# Focus: Variance Inflation Factor (VIF) and A/B Testing.

nh_day = data[(data['Road Type'] == 'National Highway') & 
              (data['Lighting Conditions'] == 'Daylight')]['Number of Casualties'].mean()

nh_dark = data[(data['Road Type'] == 'National Highway') & 
               (data['Lighting Conditions'] == 'Dark')]['Number of Casualties'].mean()

vr_day = data[(data['Road Type'] == 'Village Road') & 
              (data['Lighting Conditions'] == 'Daylight')]['Number of Casualties'].mean()

vr_dark = data[(data['Road Type'] == 'Village Road') & 
               (data['Lighting Conditions'] == 'Dark')]['Number of Casualties'].mean()

print(f"National Highway in Daylight : {nh_day:.2f}")
print(f"National Highway in Dark     : {nh_dark:.2f}")
print(f"Village Road in Daylight     : {vr_day:.2f}")
print(f"Village Road in Dark         : {vr_dark:.2f}")

# Create DataFrame for the grouped bar plot
comparison_df = pd.DataFrame({
    'Category': ['National Highway\nDaylight', 'National Highway\nDark', 
                 'Village Road\nDaylight', 'Village Road\nDark'],
    'Avg_Casualties': [nh_day, nh_dark, vr_day, vr_dark],
    'Group': ['National Highway', 'National Highway', 'Village Road', 'Village Road']
})

# Bar Plot - All four combinations in one graph
plt.figure(figsize=(12, 6))
sns.barplot(x='Category', 
            y='Avg_Casualties', 
            hue='Group', 
            data=comparison_df, 
            palette=['navy', 'darkblue', 'darkorange', 'chocolate'])

plt.title('Average Casualties: National Highway vs Village Road in Daylight vs Dark')
plt.xlabel('Road Type & Lighting Condition')
plt.ylabel('Average Number of Casualties')
plt.legend(title='Road Type')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()




# Data Preparation for Modelling:
scaler = MinMaxScaler()
data[['Speed Limit (km/h)','Driver Age']] = scaler.fit_transform(data[['Speed Limit (km/h)','Driver Age']])
print(data[['Speed Limit (km/h)','Driver Age']].describe())


# (6.)Future Predictions: Machine Learning using the CRISP-DM Process
# Focus: Supervised Learning (Linear Regression) and CRISP-DM.

loc_map = {
    'Straight Road':1,
    'Curve':2,
    'Intersection':3,
    'Bridge':4
}
data['Location_Encoded'] = data['Accident Location Details'].map(loc_map)
X = data[['Speed Limit (km/h)','Location_Encoded']]
Y = data['Number of Casualties']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)

pred = model.predict(X_test)
mse = mean_squared_error(Y_test, pred)
r2 = r2_score(Y_test, pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Model Accuracy (R-squared): {r2:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Location Coefficient: {model.coef_[1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(20,6))

importance = pd.DataFrame({'Variable':['Speed Limit','Location Encoded'],
                           'Coefficient': model.coef_})

sns.barplot(ax=axes[0], x='Coefficient', y='Variable', data = importance,hue='Variable', palette='plasma',legend=False)
axes[0].set_title('Feature Impact (Coefficients)')

# Actual vs Predicted
axes[1].scatter(Y_test, pred, alpha=0.6, color='royalblue')
axes[1].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],'r--',lw=2)
axes[1].set_title(f'Actual vs Predicted (R^2 = {r2:.3f})')
axes[1].set_xlabel('Actual Casualties')
axes[1].set_ylabel('Predicted Casualties')

residuals = Y_test - pred
axes[2].scatter(pred, residuals, alpha=0.6, color='purple')
axes[2].axhline(y=0, color='black', linestyle='--')
axes[2].set_title('Residual Analysis')
axes[2].set_xlabel('Predicted Values')
axes[2].set_ylabel('Residuals (Errors)')
plt.tight_layout()
plt.show()

print(f"Speed Coefficient: {model.coef_[0]:.4f}")

