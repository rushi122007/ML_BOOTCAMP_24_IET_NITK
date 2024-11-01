import pandas as pd
emails = pd.read_csv(r'/Users/hiren/Downloads/emails.csv')
emails = emails.dropna()
print(emails.head())
def process_email(text):
    text = text.lower()
    words = text.split()
    unique_words = list(set(words))
    return unique_words

emails['processed_text'] = emails['text'].apply(process_email)

print("Original text:", emails['text'].iloc[0])
print("Processed text:", emails['processed_text'].iloc[0])
num_emails = len(emails)
num_spam = sum(emails['spam'])
spam_probability = num_spam / num_emails

print(f"Number of emails: {num_emails}")
print(f"Number of spam emails: {num_spam}")
print(f"Probability of spam: {spam_probability:.4f}")
def train_naive_bayes(emails_data):
    model = {}
    for _, row in emails_data.iterrows():
        is_spam = row['spam']  
        words = row['processed_text']  
        for word in words:
            if word not in model:
                model[word] = {'spam': 1, 'ham': 1}
            if is_spam:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

    return model

model = train_naive_bayes(emails)

test_words = ['lottery', 'sale', 'meeting']
for word in test_words:
    if word in model:
        print(f"Word: {word}, Spam Count: {model[word]['spam']}, Ham Count: {model[word]['ham']}")
    else:
        print(f"Word: {word} not found in model")
def predict_naive_bayes(email_text, model, num_spam, num_ham):
    
    words = process_email(email_text)
    
    total_emails = num_spam + num_ham
    p_spam = num_spam / total_emails
    p_ham = num_ham / total_emails
    
    log_prob_spam = np.log(p_spam)
    log_prob_ham = np.log(p_ham)
    
    for word in words:
        if word in model:
            spam_count = model[word]['spam']
            ham_count = model[word]['ham']
        else:
            spam_count = 1
            ham_count = 1
        
        log_prob_spam += np.log(spam_count / (num_spam + 2))
        log_prob_ham += np.log(ham_count / (num_ham + 2))
    
    prob_spam = np.exp(log_prob_spam) / (np.exp(log_prob_spam) + np.exp(log_prob_ham))
    return prob_spam

test_emails = [
    "lottery winner claim prize money",
    "meeting tomorrow at 3pm",
    "buy cheap watches online"
]

for email in test_emails:
    spam_prob = predict_naive_bayes(email, model, num_spam, num_emails - num_spam)
    print(f"Email: '{email}'\nSpam Probability: {spam_prob:.4f}\n")
