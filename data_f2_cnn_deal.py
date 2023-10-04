import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

gc.collect()
# gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def data_deal():
    users_df = pd.read_csv('users.csv', encoding='utf-8', low_memory=False)
    users_df = users_df[["id", "statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "description", "test_set_2"]]
    users_df = users_df.rename(columns={'id': 'user_id'})


    tweets_df = pd.read_csv('tweets.csv', encoding='utf-8', low_memory=False)
    tweets_df = tweets_df[["id", "user_id", "source", "text", "retweet_count", "reply_count", "favorite_count"]]
    tweets_df = tweets_df.rename(columns={'id': 'tweet_id'})
    tweets_df = tweets_df.drop(tweets_df.index[-1])

    metadata_df = pd.merge(users_df, tweets_df, on='user_id')

    metadata_df.dropna(subset=['source', 'text'], inplace=True)

    metadata_df['description'].fillna(method='ffill', inplace=True)

    grouped_metadata_df = metadata_df.groupby('user_id')
    metadata_row = grouped_metadata_df.head(20)
    metadata_df = pd.DataFrame(metadata_row)

    test_set_2_0 = metadata_df[metadata_df['test_set_2'] == 0].head(30)
    test_set_2_1 = metadata_df[metadata_df['test_set_2'] == 1].head(30)

    new_users_df = pd.concat([test_set_2_0, test_set_2_1], ignore_index=True)
    metadata_df = new_users_df.sample(frac=1).reset_index(drop=True)


    tweet_numeric_columns = ['user_id', 'tweet_id', 'retweet_count', 'reply_count', 'favorite_count']
    metadata_df['tweet_id'] = metadata_df['tweet_id'].astype('float32')
    metadata_df['retweet_count'] = metadata_df['retweet_count'].astype('float32')
    metadata_df['reply_count'] = metadata_df['reply_count'].astype('float32')
    metadata_df['favorite_count'] = metadata_df['favorite_count'].astype('float32')
    tweet_metadata_numeric_tensor = torch.tensor(metadata_df[tweet_numeric_columns].values).to(device)

    vectorizer = CountVectorizer()

    tweet_text_transformed = vectorizer.fit_transform(metadata_df['text'])
    tweet_text_transformed_tensor = torch.tensor(tweet_text_transformed.toarray()).to(device)

    description_transformed = vectorizer.fit_transform(metadata_df['description'])
    description_transformed_tensor = torch.tensor(description_transformed.toarray()).to(device)

    source_transformed = vectorizer.fit_transform(metadata_df['source'])
    source_transformed_tensor = torch.tensor(source_transformed.toarray()).to(device)
    tweet_metadata_combined_tensor = torch.cat([tweet_metadata_numeric_tensor, source_transformed_tensor], dim=1)

    print(tweet_metadata_combined_tensor.shape)
    print(tweet_text_transformed_tensor.shape)
    print(description_transformed_tensor.shape)
    
    return tweet_metadata_combined_tensor, tweet_text_transformed_tensor, description_transformed_tensor, metadata_df


class TweetLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.01):
        super(TweetLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout


    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(F.dropout(embedded, p=self.dropout, training=self.training), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MetadataLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(MetadataLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        x = x.unsqueeze(1)
        h0 = h0.float()
        c0 = c0.float()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AccountDescCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes, out_channels):
        super(AccountDescCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, out_channels, kernel_size=k) for k in kernel_sizes])
        self.fc = nn.Linear(out_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(embedded)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x

class SocialBotDetector(nn.Module):
    def __init__(self, tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers, metadata_input_dim,
                 metadata_hidden_dim, metadata_num_layers, desc_vocab_size, desc_embedding_dim, desc_out_channels,
                 kernel_sizes, num_classes):
        super(SocialBotDetector, self).__init__()

        self.tweet_lstm = TweetLSTM(tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers, num_classes)
        self.metadata_lstm = MetadataLSTM(metadata_input_dim, metadata_hidden_dim, metadata_num_layers, num_classes)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes]
        self.desc_cnn = AccountDescCNN(desc_vocab_size, desc_embedding_dim, num_classes, list(kernel_sizes), desc_out_channels)
        self.fc = nn.Linear(num_classes * 3, num_classes)

    def forward(self, tweet_x, metadata_x, desc_x):
        tweet_out = self.tweet_lstm(tweet_x)
        metadata_out = self.metadata_lstm(metadata_x)
        desc_out = self.desc_cnn(desc_x)
        combined_out = torch.cat((tweet_out, metadata_out, desc_out), dim=1)
        out = self.fc(combined_out)
        return out



def train_and_eval():
    print("Loading data...")
    tweet_metadata_combined_tensor, tweet_text_transformed_tensor, description_transformed_tensor, metadata_df = data_deal()
    X_tweet_train, X_tweet_test, y_train, y_test = train_test_split(tweet_text_transformed_tensor.cpu().numpy(), metadata_df["test_set_2"].values, test_size=0.5, random_state=42)
    X_metadata_train, X_metadata_test, _, _ = train_test_split(tweet_metadata_combined_tensor.cpu().numpy(), metadata_df['test_set_2'].values, test_size=0.5, random_state=42)
    X_desc_train, X_desc_test, _, _ = train_test_split(description_transformed_tensor.cpu().numpy(), metadata_df['test_set_2'].values, test_size=0.5, random_state=42)

    tweet_vocab_size = 100000
    tweet_embedding_dim = 256
    tweet_hidden_dim = 128
    tweet_num_layers = 2
    metadata_input_dim = X_metadata_train.shape[1]
    metadata_hidden_dim = 64
    metadata_num_layers = 1
    desc_vocab_size = 100000
    desc_embedding_dim = 256
    desc_hidden_dim = 64
    desc_num_layers = 1
    num_classes = 1
    #学习率
    lr = 0.001
    num_epochs = 3
    batch_size = 1
    weight_decay = 0.002

    social_bot_detector = SocialBotDetector(tweet_vocab_size, tweet_embedding_dim, tweet_hidden_dim, tweet_num_layers,
                                             metadata_input_dim, metadata_hidden_dim, metadata_num_layers,
                                             desc_vocab_size, desc_embedding_dim, desc_hidden_dim, desc_num_layers,
                                             num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(social_bot_detector.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        social_bot_detector.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i in range(0, len(X_tweet_train), batch_size):
            end_idx = i + batch_size if i + batch_size < len(X_tweet_train) else len(X_tweet_train)
            tweet_x = torch.tensor(X_tweet_train[i:end_idx]).to(device)
            metadata_x = torch.tensor(X_metadata_train[i:end_idx]).to(device)
            desc_x = torch.tensor(X_desc_train[i:end_idx]).to(device)
            label = torch.tensor(y_train[i:end_idx], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()
            pred = social_bot_detector(tweet_x, metadata_x, desc_x)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end_idx - i)
            num_correct += ((pred > 0) == (label > 0)).sum().item()
            num_total += (end_idx - i)

        train_loss = total_loss / num_total
        train_acc = num_correct / num_total
        

        social_bot_detector.eval()
        with torch.no_grad():
            total_loss = 0
            num_correct = 0
            num_total = 0
            for i in tqdm(range(0, len(X_tweet_test), batch_size), desc="Testing"):
                end_idx = i + batch_size if i + batch_size < len(X_tweet_test) else len(X_tweet_test)
                tweet_x = torch.tensor(X_tweet_test[i:end_idx]).to(device)
                metadata_x = torch.tensor(X_metadata_test[i:end_idx]).to(device)
                desc_x = torch.tensor(X_desc_test[i:end_idx]).to(device)
                label = torch.tensor(y_test[i:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
                tweet_x = tweet_x.long()
                metadata_x = metadata_x.float()
                desc_x = desc_x.long()

                pred = social_bot_detector(tweet_x, metadata_x, desc_x)
                loss = criterion(pred, label)

                total_loss += loss.item() * (end_idx - i)
                num_correct += ((pred > 0) == (label > 0)).sum().item()
                num_total += (end_idx - i)

            val_loss = total_loss / num_total
            val_acc = num_correct / num_total

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    with torch.no_grad():
        social_bot_detector.eval()
        tweet_x = torch.tensor(X_tweet_test).to(device)
        metadata_x = torch.tensor(X_metadata_test).to(device)
        desc_x = torch.tensor(X_desc_test).to(device)
        label = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
        tweet_x = tweet_x.long()
        metadata_x = metadata_x.float()
        desc_x = desc_x.long()


        pred = social_bot_detector(tweet_x, metadata_x, desc_x)
        pred_label = (pred > 0).int().squeeze().cpu().numpy()
        print(y_test)
        print(pred_label)

        accuracy = accuracy_score(y_test, pred_label)
        precision = precision_score(y_test, pred_label)
        recall = recall_score(y_test, pred_label)
        f1 = f1_score(y_test, pred_label)

        print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(accuracy, precision, recall, f1))

        with torch.no_grad():
            social_bot_detector.eval()
            tweet_x = torch.tensor(X_tweet_train).to(device)
            metadata_x = torch.tensor(X_metadata_train).to(device)
            desc_x = torch.tensor(X_desc_train).to(device)
            label = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()

            pred_train = social_bot_detector(tweet_x, metadata_x, desc_x).cpu().numpy()
            auc_train = roc_auc_score(y_train, pred_train)

            tweet_x = torch.tensor(X_tweet_test).to(device)
            metadata_x = torch.tensor(X_metadata_test).to(device)
            desc_x = torch.tensor(X_desc_test).to(device)
            label = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            tweet_x = tweet_x.long()
            metadata_x = metadata_x.float()
            desc_x = desc_x.long()

            pred_test = social_bot_detector(tweet_x, metadata_x, desc_x).cpu().numpy()
            auc_test = roc_auc_score(y_test, pred_test)
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Train AUC: {:.4f}, Val AUC: {:.4f}'
                .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc, auc_train, auc_test))

train_and_eval()