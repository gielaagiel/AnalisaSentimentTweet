#install paket yang diperlukan
install.packages("ggplot2")
install.packages("NLP")
install.packages("tm")
install.packages("SparseM")
install.packages("RTextTools")
install.packages("e1071")
install.packages("dplyr")
install.packages("caret")
install.packages("doMC", repos="http://R-Forge.R-project.org")
install.packages("twitteR")
install.packages("ROAuth")

library(ggplot2)
library(NLP)
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
library(doMC)
library(SnowballC)


registerDoMC(cores=detectCores())  # Use all available cores
setwd("D:/bahasaR-naivebayes/")
#ambil data sentimen yang akan diuji
setup_twitter_oauth("gEaVLZS4OZquX4sINqjbwn4S3","tGYxdGAmWwxIHLATnZ5zDuGHSk8YajK9uqdUD0FKugEqC19DX6","995668719857680384-z1cDvNVvxYQZnnzONGoqyPjVGFyFlM0","wXjSpPHqnRO34tuEi03U49RHUK2bgZ7nF6QFuI6NjXeN9")
# harvest some tweets
some_tweets = searchTwitter("jokowi", n=20,lang="in")
# get the text
some_txt = sapply(some_tweets, function(x) x$getText())
# remove retweet entities
some_txt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", some_txt)
# remove at people
some_txt = gsub("@\\w+", "", some_txt)
# remove punctuation
some_txt = gsub("[[:punct:]]", "", some_txt)
# remove numbers
some_txt = gsub("[[:digit:]]", "", some_txt)
# remove html links
some_txt = gsub("http\\w+", "", some_txt)
# remove unnecessary spaces
some_txt = gsub("[ \t]{2,}", "", some_txt)
some_txt = gsub("^\\s+|\\s+$", "", some_txt)
# define “tolower error handling” function
try.error = function(x)
{
  # create missing value
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error=function(e) e)
  # if not an error
  if (!inherits(try_error, "error"))
    y = tolower(x)
  # result
  return(y)
}
some_tweets<- twListToDF(some_tweets)
 write.table(some_tweets,"D:/bahasaR-naivebayes/sentimenttweet.csv", append=T, row.names=F, col.names=T,  sep=",")

df<- read.csv("D:/bahasaR-naivebayes/sentimenttweet.csv", stringsAsFactors = FALSE)
glimpse(df)

df$class <- as.factor(df$class)
custom_stopwords <- read.csv("D:/bahasaR-naivebayes/stopword.csv", header = FALSE)
custom_stopwords <- as.character(custom_stopwords$V1)
custom_stopwords <- c(custom_stopwords, stopwords())

library(tm)
corp <- Corpus(VectorSource(some_txt[1]))
corp <- tm_map(corp, content_transformer(stringi::stri_trans_tolower))
content(corp[[1]])
corpus <- Corpus(VectorSource(some_txt))
corpus
inspect(corpus[1:5])




  summary(corpus) # get a summary
  corpus <- tm_map(corpus,content_transformer(removePunctuation))
  corpus <- tm_map(corpus,content_transformer(stripWhitespace))
  corpus <- tm_map(corpus,content_transformer(removePunctuation))
  corpus <- Corpus(VectorSource(corpus)) # change class 
  matrix_term <- DocumentTermMatrix(corpus)

dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm[1:5, 1:4])

df.train <- df[1:30,]
df.test <- df[31:40,]

dtm.train <- dtm[1:30,]
dtm.test <- dtm[31:40,]
dim(dtm.train)


corpus.clean.train <- corpus.clean[1:30]
corpus.clean.test <- corpus.clean[31:40]


dtm.train.nb <- DocumentTermMatrix(corpus.clean.train)
dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test)
dim(dtm.test.nb)



convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("0", "1"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
system.time( classifier <- naiveBayes(trainNB, df.train$class, laplace = 1) )

predict(classifier,newdata=testNB)
system.time( pred <- predict(classifier, newdata=testNB) )


table("Predictions"= pred,  "Actual" = df.test$class )



conf.mat <- confusionMatrix(pred, df.test$class)

conf.mat


library("wordcloud")
library("RColorBrewer")

dta <- TermDocumentMatrix(corpus.clean)
m <- as.matrix(dta)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 9)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(18, "Dark2"))
head(d, 9)
barplot(d[1:9,]$freq, las = 2, names.arg = d[1:9,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

dr<- read.csv("D:/bahasaR-naivebayes/sentiment.csv", stringsAsFactors = FALSE)
factor(dr$class)
w=table(dr$class,paste(dr$username,sep ="."))
w=table(dr$username, dr$class)
class(w)
t=as.data.frame(w)
names(t)[1] = 'username'
t

