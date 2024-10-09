#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


calls = pd.read_csv('calls.csv')
customers = pd.read_csv('customers.csv')
reasons = pd.read_csv('reason.csv')
sentiment_stats = pd.read_csv('sentiment_statistics.csv')
test = pd.read_csv('test.csv')


# In[4]:


print(calls.head())
print(customers.head())
print(reasons.head())
print(sentiment_stats.head()) 


# In[5]:


# Merge datasets
calls_full = pd.merge(calls, customers, on='customer_id', how='left')
calls_full = pd.merge(calls_full, sentiment_stats, on='call_id', how='left')
calls_full = pd.merge(calls_full, reasons, on='call_id', how='left')

# Ensure date columns are properly formatted
calls_full['call_start_datetime'] = pd.to_datetime(calls_full['call_start_datetime'])
calls_full['call_end_datetime'] = pd.to_datetime(calls_full['call_end_datetime'])
calls_full['agent_assigned_datetime'] = pd.to_datetime(calls_full['agent_assigned_datetime'])

# Preview the merged dataset
print(calls_full.info())


# In[6]:


# Calculate the handle time (in seconds)
calls_full['handle_time'] = (calls_full['call_end_datetime'] - calls_full['agent_assigned_datetime']).dt.total_seconds()

# Calculate AHT (Average Handle Time)
aht = calls_full['handle_time'].mean()
print(f"Average Handle Time (AHT): {aht} seconds")


# In[8]:


print(calls_full.columns)


# In[9]:


# Group by the correct column name and calculate the mean handle time (AHT) for each agent
aht_by_agent = calls_full.groupby('agent_id_x')['handle_time'].mean().reset_index()

# Rename the columns for better readability
aht_by_agent.columns = ['agent_id_x', 'average_handle_time']

# Print or inspect the results
print(aht_by_agent)


# In[10]:


# Group by the correct column name and calculate the mean handle time (AHT) for each agent
aht_by_agent = calls_full.groupby('agent_id_y')['handle_time'].mean().reset_index()

# Rename the columns for better readability
aht_by_agent.columns = ['agent_id_y', 'average_handle_time']

# Print or inspect the results
print(aht_by_agent)


# In[11]:


# Analyze AHT by call reason
aht_by_reason = calls_full.groupby('primary_call_reason')['handle_time'].mean().sort_values(ascending=False)
print(aht_by_reason)

# Analyze AHT by agent
aht_by_agent = calls_full.groupby('agent_id_x')['handle_time'].mean().sort_values(ascending=False)
print(aht_by_agent)


# In[13]:


# Calculate the waiting time (in seconds)
calls_full['waiting_time'] = (calls_full['agent_assigned_datetime'] - calls_full['call_start_datetime']).dt.total_seconds()

# Calculate AST (Average Speed to Answer)
ast = calls_full['waiting_time'].mean()
print(f"Average Speed to Answer (AST): {ast} seconds")

# Group by call reason or agent
ast_by_reason = calls_full.groupby('primary_call_reason')['waiting_time'].mean().sort_values(ascending=False)
print(ast_by_reason)


# # Sentiment Analysis & Call Handling

# In[14]:


# Analyze correlation between sentiment and AHT/AST
sns.heatmap(calls_full[['handle_time', 'waiting_time', 'average_sentiment', 'silence_percent_average']].corr(), annot=True)

# Visualize relationship between sentiment and handle time
sns.boxplot(x='agent_tone', y='handle_time', data=calls_full)
plt.title('Agent Tone vs Handle Time')

sns.boxplot(x='customer_tone', y='waiting_time', data=calls_full)
plt.title('Customer Tone vs Waiting Time')


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

# Extract top phrases from call transcripts
vectorizer = CountVectorizer(stop_words='english', max_features=100)
transcript_matrix = vectorizer.fit_transform(calls_full['call_transcript'])

# Display common phrases
common_phrases = vectorizer.get_feature_names_out()
print("Common phrases in call transcripts:", common_phrases)


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of AHT by primary call reason
plt.figure(figsize=(12, 8))
sns.barplot(x=aht_by_reason.index, y=aht_by_reason.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Average Handle Time by Primary Call Reason')
plt.ylabel('Average Handle Time (seconds)')
plt.xlabel('Primary Call Reason')
plt.tight_layout()
plt.show()


# In[29]:


# Calculate Average Speed to Answer (AST)
calls_full['waiting_time'] = (calls_full['agent_assigned_datetime'] - calls_full['call_start_datetime']).dt.total_seconds()
ast_by_agent = calls_full.groupby('agent_id_x')['waiting_time'].mean().sort_values(ascending=False)

# Plot AST by agent
plt.figure(figsize=(12, 6))
sns.barplot(x=ast_by_agent.index, y=ast_by_agent.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Average Speed to Answer by Agent')
plt.ylabel('Average Waiting Time (seconds)')
plt.xlabel('Agent ID')
plt.tight_layout()
plt.show()


# In[30]:


# Calculate Average Speed to Answer (AST)
calls_full['waiting_time'] = (calls_full['agent_assigned_datetime'] - calls_full['call_start_datetime']).dt.total_seconds()
ast_by_agent = calls_full.groupby('agent_id_y')['waiting_time'].mean().sort_values(ascending=False)

# Plot AST by agent
plt.figure(figsize=(12, 6))
sns.barplot(x=ast_by_agent.index, y=ast_by_agent.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Average Speed to Answer by Agent')
plt.ylabel('Average Waiting Time (seconds)')
plt.xlabel('Agent ID')
plt.tight_layout()
plt.show()


# In[31]:


# Plot sentiment distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='primary_call_reason', y='average_sentiment', data=calls_full, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Sentiment Distribution by Primary Call Reason')
plt.ylabel('Sentiment Score')
plt.xlabel('Primary Call Reason')
plt.tight_layout()
plt.show()


# In[32]:


# Plot correlation heatmap
plt.figure(figsize=(10, 6))
corr_matrix = calls_full[['handle_time', 'waiting_time', 'average_sentiment', 'silence_percent_average']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[33]:


# Convert call_start_datetime to datetime if not already done
calls_full['call_start_datetime'] = pd.to_datetime(calls_full['call_start_datetime'])

# Plot call volume over time (e.g., by hour)
calls_full['hour'] = calls_full['call_start_datetime'].dt.hour
plt.figure(figsize=(10, 6))
sns.countplot(x='hour', data=calls_full, palette='coolwarm')
plt.title('Call Volume by Hour of Day')
plt.ylabel('Number of Calls')
plt.xlabel('Hour of Day')
plt.tight_layout()
plt.show()


# In[43]:


# Plot AHT distribution per agent
plt.figure(figsize=(20, 6))
sns.boxplot(x='agent_id_x', y='handle_time', data=calls_full)
plt.xticks(rotation=90)
plt.title('Handle Time Distribution by Agent')
plt.ylabel('Handle Time (seconds)')
plt.xlabel('Agent ID')
plt.tight_layout()
plt.show()


