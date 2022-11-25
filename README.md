# Sentiment-Analysis



1. Crawling the information from the website of stock news (http://www.aastocks.com/en/stocks/news/aafn-company-news) and created news repository by extracting the data from html content. The crawling data fields
are Headline, Releasing Time, Company Name, Stock Code, Abstract, and Polarity.








Result: BERT has better performance compared to RNN with difference of approximately 0.07 in terms of Recall and 0.11 higher accuracy. However, BERT has much more complex as well as heavy model. Due to less of data overfitting occurs on BERT model, if much of crawling of the more data carefully, the performance for BERT would be much more promising. In terms of the complexity, BERT has Model Parameters of 109484547, RNN has Model Parameters of 2053443. From tqdm, we can deduce the time taken per epoch, BERT takes 1763 seconds, and RNN takes 53 seconds in total. The following recent trend is that BERT has significance that pretraining is feasible and large model. For the pos/neg/neu ratio, the positive ratio takes over 90 percent, and hard to define neutral. Hence, the labeled as 2 when pos > neg*2, 0 when neg > pos, 1 for neutral.

![alt text](http://url/to/img.png)
