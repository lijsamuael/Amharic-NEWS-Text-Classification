#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
data = pandas.read_csv('data/AmharicNewsDataset.csv')
data.head(20)


# In[2]:


data.tail(10)


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.groupby("category").sum()


# In[ ]:





# In[6]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
data.groupby('category').headline.count().plot.bar(ylim=0)

plt.show()


# In[ ]:





# In[ ]:





# In[7]:


#cheking for null values
data.isna().sum()


# In[8]:


#removing null values form the data
data = data.dropna(subset=['headline','category'] )
data.shape


# In[9]:


#numrical discription of the data
data.describe()


# In[ ]:





# In[10]:


#cheking again for null values after removal
data.isna().sum()


# In[ ]:





# In[ ]:





# In[11]:


#getting the number of words in the article and headline
data['article_len'] = data['article'].str.split().str.len()
data['headline_len'] = data['headline'].str.split().str.len()
data.head()


# In[12]:


#avoiding numbers from the article
data.article = data.article.str.replace('\d+', '')
data.headline = data.headline.str.replace('\d+', '')

#visualizing the data after removing numbers
data['article_len'] = data['article'].str.split().str.len()
data['headline_len'] = data['headline'].str.split().str.len()
data.head(5)


# In[13]:


#removing english letters from the dataset
import string
#removing all lowercase letters
for i in string.ascii_lowercase:
    data.headline = data.headline.str.replace(i, "")
    data.article = data.article.str.replace(i, "")
#removing all uppercase letters
for i in string.ascii_uppercase:
    data.headline = data.headline.str.replace(i, "")
    data.article = data.article.str.replace(i, "")
#removing latine words
latinwords = ["ãýùçãëğéèêïôöæčćć"]
for i in latinwords:
    data = data.replace(i, ' ')
data.head()


# In[14]:


#Defining stop words
import numpy as np
sw =np.array(['የ','በ', 'ለ','እኔ', 'የኔ', 'የእኔ', 'የኛ', 'የእኛ', 'አንተ', 'የአንተ', 'አንች', 'የአንች', 'አንተን', 'አንችን', "እሱ", "እሱን", 'እሷን', 'የሱ', 'የሷ', 'የእሱ', 'የእሷ', 'የእኛ', 'የኛ', 'እነሱ', 'የእነርሱ', "የእነሱ", 'ምን', 'ምንድን', 'የት', 'የቱ', "የቱን", 'ማን', 'ማንን', 'የማን', 'የእነማን', 'የነማን', 'ይህ', 'ይህኛው', 'እሄኛው', 'ያ', 'ያኛው', 'ያኛውን', 'እነዛ', 'እነዚህ', "ነኝ", 'ናት', 'ነው', 'ነን', 'ናቸው', 'ነበር', 'ነበሩ', 'ነበርን', 'ነበረች', 'ነበር', 'ትላንት', 'ትናንት', 'ነገ', 'ዛሬ', 'ከነገ', 'ከትናንት', 'ወዲያ', 'በስትያ', 'በስተያ', 'ሆነ', 'ሆነች', 'ሆኑ', 'ሆንን', 'አለኝ', 'አላት', 'አለን', 'አላቸው', 'መኖር', 'አደረገ', 'አደረገች', 'አደረጉ', 'አደረግን', 'ና', 'እና', 'ነገር', 'ግን', 'ነገርግን', 'ወይም', 'ወይንም', 'ምክንያቱም', 'ምክንያት', 'እስከ', 'እስከዛ', 'ከዛ', 'ወደዛ', 'ወደዚህ', 'እያለ', 'እያለች', 'እያሉ', 'እያልን', 'እየተባለ', 'ከላይ', 'በላይ', 'የበላይ', 'ከታች', 'የበታች', 'ታች', 'ላይ', 'ስለ', 'ጋር', 'ጋራ', 'መተባበር', 'በመተባበር', 'መካከል', 'በመካከል', 'ውስጥ', 'ከውስጥ', 'በውስጥ', 'ድጋሜ', 'በድጋሜ', 'በተደጋጋሚ', 'በመደጋገም', 'በስፋት', 'አንዴ', 'ሁለቴ', 'ሶስቴ', 'አንደኛ', 'ሁለተኛ', 'ሶስተኛ', 'አንድ', 'ሁለት', 'ሶስት', 'ለምን', 'ስለምን', 'ምክያት', 'ምክኛቱም', 'ምን', 'ምንን', 'ቢሆን', 'ሲሆን', 'ሁለቱም', "ሶስቱም", 'ሁሉም', "ጥቂት", 'በጥቂቱ', 'ቲኒሽ', 'በእቲኒሹ', 'ሌላ', 'ሌሎች', 'ሌላኛው', 'ተጨማሪ', 'በተጨማሪ', 'ብቻ', 'ብቻውን', "ብቻዋን", 'ተመሳሳይ', "መልኩ", 'መልክ', "ስለ", 'ስለዚህ', "ስለሆነ", 'ስለሆነም', "ይበልጣል", 'ትበልጣለች', "ይበልጣሉ", 'በጣም', "እጅግ", 'አለበት', "አለባት", 'አለብን', 'አለባቸው', "አለብን", 'አሁን', "ዛሬ", 'ነገ', "መቼ", 'እና', "በፊት", 'በኋላ', "ቀጥሎ", 'በመቀጠል', "ከዛ", 'ወደ', "ስለ", 'በእርግጥ', "በመሆኑም", 'በመሆኑ', "ስለሆነ", 'ስለሆነም', 'በግልፅ', 'በዝርዝር', 'ግልፅ', 'ዝርዝር', 'ይቻላል', 'ይችላል', 'ተቻለ', "የተለየ", 'የተለያዩ', "እያንዳንዱ", 'እያንዳንዱን', "መጀመሪያ", 'በመጀመሪያ', "መጨረሻ", 'በመጨረበ', "አስታወቀ", 'አስተዋዋቀ', "ይጠበቃል", 'በለዋል', "ብሏል", 'ብላለች', "ብለናል", 'ለ', 'ወደፊት', "ፊት", 'ከፊት', "ወደኋላ", 'ኋላ', "ሙሉ", 'በመሉ', "ዛሬም", "ተፈጠረ", 'ተፈጠሩ', "ወዲ", 'ወዲህ', "ወዲያ", 'በፍጥነት', "አካሄደ", 'ተካሄደ', "ሄደ", 'ሄደች', "ውስጥ", 'በውስጥ', 'አመለከተ', "ያመለክታል", 'በስፋት', "ያስደስታል", 'ቀጥታ', "በቀጥታ", 'ጠበቀ', "ጠበቁ", 'ጥብቀዋል', "በመጠበቅ", 'ላይ', "የታወቀ", 'በመሆኑ', "በመሆኑም", 'ቲኒሽ', "ትልቅ", 'ጥቂት', "ማለት", "ማለትም", 'ቀደም', 'የመጀመሪያ', 'ብዙ', 'ጠቅላላ', 'ሆኖም', " የ", " ከ", ' ለ', ' በ', 'ስለ'])
sea = set(['እኔ', 'የኔ', 'የእኔ', 'የኛ', 'የእኛ', 'አንተ', 'የአንተ', 'አንች', 'የአንች', 'አንተን', 'አንችን', "እሱ", "እሱን", 'እሷን', 'የሱ', 'የሷ', 'የእሱ', 'የእሷ', 'የእኛ', 'የኛ', 'እነሱ', 'የእነርሱ', "የእነሱ", 'ምን', 'ምንድን', 'የት', 'የቱ', "የቱን", 'ማን', 'ማንን', 'የማን', 'የእነማን', 'የነማን', 'ይህ', 'ይህኛው', 'እሄኛው', 'ያ', 'ያኛው', 'ያኛውን', 'እነዛ', 'እነዚህ', "ነኝ", 'ናት', 'ነው', 'ነን', 'ናቸው', 'ነበር', 'ነበሩ', 'ነበርን', 'ነበረች', 'ነበር', 'ትላንት', 'ትናንት', 'ነገ', 'ዛሬ', 'ከነገ', 'ከትናንት', 'ወዲያ', 'በስትያ', 'በስተያ', 'ሆነ', 'ሆነች', 'ሆኑ', 'ሆንን', 'አለኝ', 'አላት', 'አለን', 'አላቸው', 'መኖር', 'አደረገ', 'አደረገች', 'አደረጉ', 'አደረግን', 'ና', 'እና', 'ነገር', 'ግን', 'ነገርግን', 'ወይም', 'ወይንም', 'ምክንያቱም', 'ምክንያት', 'እስከ', 'እስከዛ', 'ከዛ', 'ወደዛ', 'ወደዚህ', 'እያለ', 'እያለች', 'እያሉ', 'እያልን', 'እየተባለ', 'ከላይ', 'በላይ', 'የበላይ', 'ከታች', 'የበታች', 'ታች', 'ላይ', 'ስለ', 'ጋር', 'ጋራ', 'መተባበር', 'በመተባበር', 'መካከል', 'በመካከል', 'ውስጥ', 'ከውስጥ', 'በውስጥ', 'ድጋሜ', 'በድጋሜ', 'በተደጋጋሚ', 'በመደጋገም', 'በስፋት', 'አንዴ', 'ሁለቴ', 'ሶስቴ', 'አንደኛ', 'ሁለተኛ', 'ሶስተኛ', 'አንድ', 'ሁለት', 'ሶስት', 'ለምን', 'ስለምን', 'ምክያት', 'ምክኛቱም', 'ምን', 'ምንን', 'ቢሆን', 'ሲሆን', 'ሁለቱም', "ሶስቱም", 'ሁሉም', "ጥቂት", 'በጥቂቱ', 'ቲኒሽ', 'በእቲኒሹ', 'ሌላ', 'ሌሎች', 'ሌላኛው', 'ተጨማሪ', 'በተጨማሪ', 'ብቻ', 'ብቻውን', "ብቻዋን", 'ተመሳሳይ', "መልኩ", 'መልክ', "ስለ", 'ስለዚህ', "ስለሆነ", 'ስለሆነም', "ይበልጣል", 'ትበልጣለች', "ይበልጣሉ", 'በጣም', "እጅግ", 'አለበት', "አለባት", 'አለብን', 'አለባቸው', "አለብን", 'አሁን', "ዛሬ", 'ነገ', "መቼ", 'እና', "በፊት", 'በኋላ', "ቀጥሎ", 'በመቀጠል', "ከዛ", 'ወደ', "ስለ", 'በእርግጥ', "በመሆኑም", 'በመሆኑ', "ስለሆነ", 'ስለሆነም', 'በግልፅ', 'በዝርዝር', 'ግልፅ', 'ዝርዝር', 'ይቻላል', 'ይችላል', 'ተቻለ', "የተለየ", 'የተለያዩ', "እያንዳንዱ", 'እያንዳንዱን', "መጀመሪያ", 'በመጀመሪያ', "መጨረሻ", 'በመጨረበ', "አስታወቀ", 'አስተዋዋቀ', "ይጠበቃል", 'በለዋል', "ብሏል", 'ብላለች', "ብለናል", 'ለ', 'ወደፊት', "ፊት", 'ከፊት', "ወደኋላ", 'ኋላ', "ሙሉ", 'በመሉ', "ተፈጠረ", 'ተፈጠሩ', "ወዲ", 'ወዲህ', "ወዲያ", 'በፍጥነት', "አካሄደ", 'ተካሄደ', "ሄደ", 'ሄደች', "ውስጥ", 'በውስጥ', 'አመለከተ', "ያመለክታል", 'በስፋት', "ያስደስታል", 'ቀጥታ', "በቀጥታ", 'ጠበቀ', "ጠበቁ", 'ጥብቀዋል', "በመጠበቅ", 'ላይ', "የታወቀ", 'በመሆኑ', "በመሆኑም", 'ቲኒሽ', "ትልቅ", 'ጥቂት', "ማለት", "ማለትም"])
stopwords = np.array(sea)
print(stopwords)


# In[15]:


#Removing stop words
import nltk
data['article'] = data['article'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))


#visuslizing the dataset after removing stop words
data['article_len'] = data['article'].str.split().str.len()
data['headline_len'] = data['headline'].str.split().str.len()
data.head(20)


# In[16]:


#Removing punctuation marks

punctuations = ['፣', '፤', '፥', '።', '.', '/', "'\'" "'", '"', '፡', '-', '››', '!', '.', '%', '$', '(', ')', '@', '&','#','$','+', '=','*', '>', '‹‹', 'ʻ', 'ʼ', '_'] 

for punc in punctuations:
        data.article = data.article.str.replace(punc, ' ')
        data.headline = data.headline.str.replace(punc, ' ')
data.head(12)
        


# In[17]:


#removing words that are used most in NEWS
zena_words = ['ተናግረዋል','አስተላልፈዋል', 'ያሳያል','ተደርጓል','ይጠቀሳል','አድርጓል','አስተላልፏል', 'ይሆናል','አስረድተዋል','ነበር', 'እንዲሁም','እንደሚያደርገው','እንችላለን', 'የለም','ይፈቅዳል','ይገባቸዋል','ሳይሆን','ገብተዋል', 'አቅርበዋል','ጠቅሰው', 'መኖራቸውን','ጉዳይ','እናመሰግናለን','ሁኔታ', 'አይደለም','ይህም', 'ይገኛሉ','ይታያል','እንደሆነ',  'መሆናቸውን','ይሆናሉ', 'እንደመሆኑ','ይችላሉ', 'ረገድ'   ]
for word in zena_words:
        data.headline = data.headline.str.replace(word, '')
        data.article = data.article.str.replace(word, '')
        
#visuslizing the dataset after frequently used NEWS words
data['article_len'] = data['article'].str.split().str.len()
data['headline_len'] = data['headline'].str.split().str.len()
data.head(20)


# # Steaming
# 

# In[18]:


#Removing post fixes that are added to not sab ab
import nltk
abzi_kitya = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን', '']
        
for punc in abzi_kitya:
        data.headline = data.headline.str.replace(punc, '')
        data.article = data.article.str.replace(punc, '')
        
data.head(20)


# In[ ]:


#removing prefix የ

#removing the prefix when it appears first in the sentence
for i in data.headline:
    data.headline[i] = i.lstrip("የ")
for i in data.article:
    data.headline[i] = i.lstrip("የ")
data.head(20)


# In[ ]:


abzii_kitya = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ha = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_le = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_hemeru = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_me = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_se = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_re = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_sse = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_she = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_qe = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_be = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_te = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_che = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ne = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_gne = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_a = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ke = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_he = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_we = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_aa = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ze = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_zze = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ye = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_de = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_je = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ge = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_tte = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_cce = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_ppe = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_tse = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_tsse = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_fe = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']
abzi_pe = ['ዎቻቸው', 'ዎቻቸውን', 'ዎቻችን', 'ዎቻችንን', 'ዎችም', 'ዎችና', 'ዎችን', 'ዎቿ', 'ዎቿን', 'ዎቹ', 'ዎቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ዎችንም', 'ዎችሽች', 'ዎቼ', 'ዎቼን', 'ዎቹንም', 'ዎቿንም', 'ዎቾቼን']

count = 0
for i in abzii_kitya:
    abzi_ha[count] = i.replace('ዎ', 'ሆ')
    abzi_le[count] = i.replace('ዎ', 'ሎ')
    abzi_hemeru[count] = i.replace('ዎ', 'ሖ')
    abzi_me[count] = i.replace('ዎ', 'ሞ')
    abzi_re[count] = i.replace('ዎ', 'ሮ')
    abzi_se[count] = i.replace('ዎ', 'ሶ')
    abzi_she[count] = i.replace('ዎ', 'ሾ')
    abzi_qe[count] = i.replace('ዎ', 'ቆ')
    abzi_be[count] = i.replace('ዎ', 'ቦ')
    abzi_te[count] = i.replace('ዎ', 'ቶ')
    abzi_che[count] = i.replace('ዎ', 'ቾ')
    abzi_gne[count] = i.replace('ዎ', 'ኞ')
    abzi_ne[count] = i.replace('ዎ', 'ኖ')
    abzi_a[count] = i.replace('ዎ', 'ኦ')
    abzi_ke[count] = i.replace('ዎ', 'ኮ')
    abzi_he[count] = i.replace('ዎ', 'ኆ')
    abzi_we[count] = i.replace('ዎ', 'ዎ')
    abzi_aa[count] = i.replace('ዎ', 'ዖ')
    abzi_ze[count] = i.replace('ዎ', 'ዞ')
    abzi_zze[count] = i.replace('ዎ', 'ዦ')
    abzi_ye[count] = i.replace('ዎ', 'ዮ')
    abzi_de[count] = i.replace('ዎ', 'ዶ')
    abzi_je[count] = i.replace('ዎ', 'ጆ')
    abzi_ge[count] = i.replace('ዎ', 'ጎ')
    abzi_tte[count] = i.replace('ዎ', 'ጦ')
    abzi_cce[count] = i.replace('ዎ', 'ጮ')
    abzi_ppe[count] = i.replace('ዎ', 'ጶ')
    abzi_tse[count] = i.replace('ዎ', 'ጾ')
    abzi_tsse[count] = i.replace('ዎ', 'ፆ')
    abzi_fe[count] = i.replace('ዎ', 'ፎ')
    abzi_pe[count] = i.replace('ዎ', 'ፖ')
    abzi_sse[count] = i.replace('ዎ', 'ሦ')
    count += 1
    
print(abzi_ha)
print(abzi_le)
print(abzi_hemeru)
print(abzi_me)
print(abzi_se)
print(abzi_re)
print(abzi_sse)
print(abzi_she)
print(abzi_qe)
print(abzi_be)
print(abzi_te)
print(abzi_che)
print(abzi_gne)
print(abzi_ne)
print(abzi_a)
print(abzi_ke)
print(abzi_he)
print(abzi_we)
print(abzi_aa)
print(abzi_ze)
print(abzi_zze)
print(abzi_ye)
print(abzi_de)
print(abzi_je)
print(abzi_ge)
print(abzi_tte)
print(abzi_cce)
print(abzi_ppe)
print(abzi_tse)
print(abzi_tsse)
print(abzi_fe)
print(abzi_pe)


# In[ ]:


#replacing the multi sufixs of the words that ends with sixth letters of amharic words
for punc in abzi_ha:
        data.headline = data.headline.str.replace(punc, 'ህ')
        data.article = data.article.str.replace(punc, 'ህ')
        
for kitya in abzi_le:
            data.headline = data.headline.str.replace(kitya, 'ል')
            data.article = data.article.str.replace(kitya, 'ል')
                    
for kitya in abzi_hemeru:
            data.headline = data.headline.str.replace(kitya, 'ሕ')
            data.article = data.article.str.replace(kitya, 'ሕ')
                    
for kitya in abzi_me:
            data.headline = data.headline.str.replace(kitya, 'ም')
            data.article = data.article.str.replace(kitya, 'ም')
                    
for kitya in abzi_se:
            data.headline = data.headline.str.replace(kitya, 'ስ')
            data.article = data.article.str.replace(kitya, 'ስ')
        
for kitya in abzi_re:
            data.headline = data.headline.str.replace(kitya, 'ር')
            data.article = data.article.str.replace(kitya, 'ር')
                    
for kitya in abzi_sse:
            data.headline = data.headline.str.replace(kitya, 'ሥ')
            data.article = data.article.str.replace(kitya, 'ሥ')
                    
for kitya in abzi_she:
            data.headline = data.headline.str.replace(kitya, 'ሽ')
            data.article = data.article.str.replace(kitya, 'ሽ')
                    
for kitya in abzi_qe:
            data.headline = data.headline.str.replace(kitya, 'ቅ')
            data.article = data.article.str.replace(kitya, 'ቅ')
                    
for kitya in abzi_be:
            data.headline = data.headline.str.replace(kitya, 'ብ')
            data.article = data.article.str.replace(kitya, 'ብ')
                    
for kitya in abzi_te:
            data.headline = data.headline.str.replace(kitya, 'ት')
            data.article = data.article.str.replace(kitya, 'ት')
                    
for kitya in abzi_che:
            data.headline = data.headline.str.replace(kitya, 'ች')
            data.article = data.article.str.replace(kitya, 'ች')
                    
for kitya in abzi_gne:
            data.headline = data.headline.str.replace(kitya, 'ኝ')
            data.article = data.article.str.replace(kitya, 'ኝ')
                    
for kitya in abzi_ne:
            data.headline = data.headline.str.replace(kitya, 'ን')
            data.article = data.article.str.replace(kitya, 'ን')
                    
for kitya in abzi_a:
            data.headline = data.headline.str.replace(kitya, 'እ')
            data.article = data.article.str.replace(kitya, 'እ')
                    
for kitya in abzi_ke:
            data.headline = data.headline.str.replace(kitya, 'ክ')
            data.article = data.article.str.replace(kitya, 'ክ')
                    
for kitya in abzi_he:
            data.headline = data.headline.str.replace(kitya, 'ኅ')
            data.article = data.article.str.replace(kitya, 'ኅ')
                    
for kitya in abzi_we:
            data.headline = data.headline.str.replace(kitya, 'ው')
            data.article = data.article.str.replace(kitya, 'ው')
                    
for kitya in abzi_aa:
            data.headline = data.headline.str.replace(kitya, 'ዕ')
            data.article = data.article.str.replace(kitya, 'ዕ')
                    
for kitya in abzi_ze:
            data.headline = data.headline.str.replace(kitya, 'ዝ')
            data.article = data.article.str.replace(kitya, 'ዝ')
                    
for kitya in abzi_zze:
            data.headline = data.headline.str.replace(kitya, 'ዥ')
            data.article = data.article.str.replace(kitya, 'ዥ')
                    
for kitya in abzi_ye:
            data.headline = data.headline.str.replace(kitya, 'ይ')
            data.article = data.article.str.replace(kitya, 'ይ')
                    
for kitya in abzi_de:
            data.headline = data.headline.str.replace(kitya, 'ድ')
            data.article = data.article.str.replace(kitya, 'ድ')
                    
for kitya in abzi_je:
            data.headline = data.headline.str.replace(kitya, 'ጅ')
            data.article = data.article.str.replace(kitya, 'ጅ')
                    
for kitya in abzi_ge:
            data.headline = data.headline.str.replace(kitya, 'ግ')
            data.article = data.article.str.replace(kitya, 'ግ')
                    
for kitya in abzi_tte:
            data.headline = data.headline.str.replace(kitya, 'ጥ')
            data.article = data.article.str.replace(kitya, 'ጥ')
                    
for kitya in abzi_cce:
            data.headline = data.headline.str.replace(kitya, 'ጭ')
            data.article = data.article.str.replace(kitya, 'ጭ')
                                
for kitya in abzi_ppe:
            data.headline = data.headline.str.replace(kitya, 'ጵ')
            data.article = data.article.str.replace(kitya, 'ጵ')
                                
for kitya in abzi_tse:
            data.headline = data.headline.str.replace(kitya, 'ጽ')
            data.article = data.article.str.replace(kitya, 'ጽ')
                                
for kitya in abzi_tsse:
            data.headline = data.headline.str.replace(kitya, 'ፅ')
            data.article = data.article.str.replace(kitya, 'ፅ')
                                
for kitya in abzi_fe:
            data.headline = data.headline.str.replace(kitya, 'ፍ')
            data.article = data.article.str.replace(kitya, 'ፍ')
                                
for kitya in abzi_pe:
            data.headline = data.headline.str.replace(kitya, 'ፕ')
            data.article = data.article.str.replace(kitya, 'ፕ')
        
data.head(20)


# In[ ]:


#removing possession indicator suffixes from the texts
balebetA_kitya = ['ቱን', 'ነት', 'ቻቸው', 'ቻቸውን', 'ቻችን', 'ቻችንን', 'ችም', 'ችና', 'ችን', 'ቿ', 'ቿን', 'ቹ', 'ቹም', 'ወች', 'ዋች', 'ዎች', 'ዎችም', 'ዎችን', 'ንም', 'ች', 'ቹን']

for kitya in balebetA_kitya:
    data.headline = data.headline.str.replace(kitya, '')
    data.article = data.article.str.replace(kitya, '')
data.head(10)


# In[ ]:


#Changing words to one if that have similar sounds.
ha_sound = ['ሃ','ኅ','ኃ','ሐ','ሓ','ኻ']
hu_sound = ['ሑ','ኁ','ኹ']
hi_sound = ['ሒ','ኂ','ኺ']
he_sound = ['ሔ','ኄ','ኼ']
h_sound = ['ሕ','ኅ','ኽ']
ho_sound = ['ሑ','ኁ','ዅ']
a_sound = ['ዓ', 'ኣ', 'ዐ']

for fidel in ha_sound:
            data.headline = data.headline.str.replace(fidel, 'ሀ')
            data.article = data.article.str.replace(fidel, 'ሀ')
for fidel in hu_sound:
            data.headline = data.headline.str.replace(fidel, 'ሁ')
            data.article = data.article.str.replace(fidel, 'ሁ')
for fidel in hi_sound:
            data.headline = data.headline.str.replace(fidel, 'ሂ')
            data.article = data.article.str.replace(fidel, 'ሂ')
for fidel in he_sound:
            data.headline = data.headline.str.replace(fidel, 'ሄ')
            data.article = data.article.str.replace(fidel, 'ሄ')
for fidel in h_sound:
            data.headline = data.headline.str.replace(fidel, 'ህ')
            data.article = data.article.str.replace(fidel, 'ህ')
for fidel in ho_sound:
            data.headline = data.headline.str.replace(fidel, 'ሆ')
            data.article = data.article.str.replace(fidel, 'ሆ')
for fidel in a_sound:
            data.headline = data.headline.str.replace(fidel, 'አ')
            data.article = data.article.str.replace(fidel, 'አ')
            
data.headline = data.headline.str.replace('ዑ', 'ኡ')
data.article = data.article.str.replace('ዑ', 'ኡ')

data.headline = data.headline.str.replace('ዒ', 'ኢ')
data.article = data.article.str.replace('ዒ', 'ኢ')

data.headline = data.headline.str.replace('ዔ', 'ኤ')
data.article = data.article.str.replace('ዔ', 'ኤ')

data.headline = data.headline.str.replace('ዕ', 'እ')
data.article = data.article.str.replace('ዕ', 'እ')

data.headline = data.headline.str.replace('ዖ', 'ኦ')
data.article = data.article.str.replace('ዖ', 'ኦ')

            
data.headline = data.headline.str.replace('ሠ', 'ሰ')
data.article = data.article.str.replace('ሠ', 'ሰ')

data.headline = data.headline.str.replace('ሡ', 'ሱ')
data.article = data.article.str.replace('ሡ', 'ሱ')

data.headline = data.headline.str.replace('ሢ', 'ሲ')
data.article = data.article.str.replace('ሢ', 'ሲ')

data.headline = data.headline.str.replace('ሣ', 'ሳ')
data.article = data.article.str.replace('ሣ', 'ሳ')

data.headline = data.headline.str.replace('ሤ', 'ሴ')
data.article = data.article.str.replace('ሴ', 'ሴ')

data.headline = data.headline.str.replace('ሥ', 'ስ')
data.article = data.article.str.replace('ሦ', 'ሶ')

data.headline = data.headline.str.replace('ጸ', 'ፀ')
data.article = data.article.str.replace('ጸ', 'ፀ')

data.headline = data.headline.str.replace('ጹ', 'ፁ')
data.article = data.article.str.replace('ጹ', 'ፁ')

data.headline = data.headline.str.replace('ጺ', 'ፂ')
data.article = data.article.str.replace('ጺ', 'ፂ')

data.headline = data.headline.str.replace('ጻ', 'ፃ')
data.article = data.article.str.replace('ጻ', 'ፃ')

data.headline = data.headline.str.replace('ጼ', 'ፄ')
data.article = data.article.str.replace('ጼ', 'ፄ')

data.headline = data.headline.str.replace('ጽ', 'ፅ')
data.article = data.article.str.replace('ጽ', 'ፅ')

data.headline = data.headline.str.replace('ጾ', 'ፆ')
data.article = data.article.str.replace('ጾ', 'ፆ')

data.head(5)


# # Counting the number of occurance of each word in the whole document

# In[ ]:


#counting the number of 
import nltk
from nltk import word_tokenize
wordcount = {}
for row in data.article:
    words = nltk.word_tokenize(str(row))
    for word in words:
        if word not in wordcount.keys():
            wordcount[word] = 1
        else:
            wordcount[word] += 1
print(len(wordcount))
wordcount


# # Vectorizing using CountVectorizer
# 

# In[ ]:


#changing words to numbers using count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000, min_df=3, max_df=0.7)
article_vectorizer = vectorizer.fit_transform(data.article).toarray()
                                            
print(article_vectorizer)


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(article_vectorizer, data.category , train_size = 0.8 ,  random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # Naive Bayes

# In[ ]:


# modeling with Naive Bayes 
from sklearn.naive_bayes import GaussianNB
Model_Bayes = GaussianNB()
Model_Bayes.fit(X_train, Y_train)

# Predict the category based on article and headline
prediction = Model_Bayes.predict(X_test)




# In[ ]:





# In[ ]:


# Accuracy measurement
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, prediction)

print("Prediction  accurary using training data  : " , accuracy_score(Y_train, prediction))


# # TF-IDF with SVM

# In[108]:


#splitting again for tf-idf vectorizer
Train_X, Test_X, Train_Y, Test_Y = train_test_split(data.article,data.category,test_size=0.2)
print(Train_X.shape)
print(Test_X.shape)
print(Train_Y.shape)
print(Test_Y.shape)


# # TF-IDF

# In[111]:


from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['article'].values)

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Train_X_Tfidf)


# In[113]:


# Classifier - Algorithm - SVM
from sklearn import svm

# fit the training dataset to the model
Model_SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
Model_SVM.fit(Train_X_Tfidf,Train_Y)

# predict the category of the text
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

