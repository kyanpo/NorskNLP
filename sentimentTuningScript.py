from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import DistilBertTokenizer
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import argparse

#The name here isn't specificlly for IMDB.  This is just from huggingface
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def main(data):
	

	#Read in data
	df = pd.read_csv(data)


	#Data must be formatted into csv file with "review" and "score".  Even if the data isn't necessarily like that.
	X_train,test_texts, y_train, test_labels = train_test_split(df['Review'], df['Score'], test_size=0.2, random_state=1)

	train_texts, val_texts, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


	#Get the pre-trained model
	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


	#get our encodings
	train_encodings = tokenizer(train_texts, truncation=True, padding=True)
	val_encodings = tokenizer(val_texts, truncation=True, padding=True)
	test_encodings = tokenizer(test_texts, truncation=True, padding=True)



	#Py-torch versions of encodings essentially
	train_dataset = IMDbDataset(train_encodings, train_labels)
	val_dataset = IMDbDataset(val_encodings, val_labels)
	test_dataset = IMDbDataset(test_encodings, test_labels)



	#These are training parameters
	training_args = TrainingArguments(
	    output_dir='./results',          # output directory
	    num_train_epochs=3,              # total number of training epochs
	    per_device_train_batch_size=16,  # batch size per device during training
	    per_device_eval_batch_size=64,   # batch size for evaluation
	    warmup_steps=500,                # number of warmup steps for learning rate scheduler
	    weight_decay=0.01,               # strength of weight decay
	    logging_dir='./logs',            # directory for storing logs
	    logging_steps=10,
	)

	model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

	trainer = Trainer(
	    model=model,                         # the instantiated 🤗 Transformers model to be trained
	    args=training_args,                  # training arguments, defined above
	    train_dataset=train_dataset,         # training dataset
	    eval_dataset=val_dataset             # evaluation dataset
	)

	trainer.train()


#For commandline functionality
def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN to single Highlight.')
    parser.add_argument("dataPath", help = "Location of csv data.")


    return parser.parse_args()



if __name__ == '__main__':


    arguments = parse_arguments()

    #startTime = time.time()

    main(arguments.dataPath) 

    #endTime = time.time()

    #print(endTime - startTime)




#Below is for actually using it in a pipeline.

#from transformers import pipeline
#pipeline('sentiment-analysis')

#nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device = 0)

#sequence_0 = "The company HuggingFace is based in New York City"
#sequence_1 = "Apples are especially bad for your health"
#sequence_2 = "HuggingFace's headquarters are situated in Manhattan"


#nlp(sequence_0)


#print(nlp(sequence_1))

#print(nlp(sequence_2))

#>>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#>>> model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', return_dict=True)




#from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

#tokeni = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#mode = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#nlp = pipeline('sentiment-analysis', model=mode, tokenizer=tokeni, device = 0)


#print(nlp(sequence_1))

#print(nlp(sequence_2))
