# Sentiment-Analysis



1. Crawling the information from the website of stock news (http://www.aastocks.com/en/stocks/news/aafn-company-news) and created news repository by extracting the data from html content. The crawling data fields
are Headline, Releasing Time, Company Name, Stock Code, Abstract, and Polarity.

2. Performing sentiment analysis utilizing data extracted by using Machine Learning algorithms particularly focused on traditional neural network, and BERT by splitting datasets into 80:20 for training datasets and
validation datasets.

3. Implementation of various interpolation, optimization algorithms with error analysis, revealed with graphs utilizing
Pyplot. Started this project as I have an interest on creating intuitive graphs, and reports from different statistical formula and tools to enhance logical approach to analyzing data.








**Result** 


BERT has better performance compared to RNN with difference of approximately 0.07 in terms of Recall and 0.11 higher accuracy. However, BERT has much more complex as well as heavy model. Due to less of data overfitting occurs on BERT model, if much of crawling of the more data carefully, the performance for BERT would be much more promising. In terms of the complexity, BERT has Model Parameters of 109484547, RNN has Model Parameters of 2053443. From tqdm, we can deduce the time taken per epoch, BERT takes 1763 seconds, and RNN takes 53 seconds in total. The following recent trend is that BERT has significance that pretraining is feasible and large model. For the pos/neg/neu ratio, the positive ratio takes over 90 percent, and hard to define neutral. Hence, the labeled as 2 when pos > neg*2, 0 when neg > pos, 1 for neutral.

**RNN** 
![RNN1](https://user-images.githubusercontent.com/74304944/203905984-0de07018-47a3-40ca-a3f9-28d2b7d9b7b8.png)

**BERT** 
![BERT1](https://user-images.githubusercontent.com/74304944/203905996-80086906-a219-4ad5-908a-eeb3dff7d342.png)

**RNN Graph** 
![RNN](https://user-images.githubusercontent.com/74304944/203906031-92c32833-65d4-4689-b14d-ad7750d351af.png)

**BERT Graph** 
![BERT](https://user-images.githubusercontent.com/74304944/203906056-911ceb98-67db-4ccb-aaa4-9068ac0f8ca3.png)





