import nltk
import gensim
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def clean_text(s, stops):
    # Lower case
    a = s.lower()
    # Remove punctuation
    a = ''.join(c for c in a if c not in string.punctuation)
    # Remove numbers
    a = ''.join(c for c in a if c not in '0123456789')
    # Remove stopwords and trim extra whitespace
    #  a = ' '.join(a.split())
    a = ' '.join(c for c in a.split() if c not in stops)
    return a


def clean_dataset(headlines, articles):
    stops = set(stopwords.words("english"))
    for k, v in articles.items():
        articles[k] = clean_text(v, stops)
    for s in headlines:
        s['Headline'] = clean_text(s['Headline'], stops)


def generate_tfidf_vect(dataset, tfidf, training=True, debug=True):
    is_headline = 1
    # 0 element: indicate whether it's an article(=0) or headline(=1),
    # 1: body id or key
    # 2: the string
    # 3: stance
    vect_tmp = []
    for k, v in dataset.articles.items():
        vect_tmp.append([0, k, v, None])
    for s in dataset.stances:
        vect_tmp.append([1, s['Body ID'], s['Headline'], s['Stance']])
    # TFIDF articles concat with headlines as 1 list
    v = [t[2] for t in vect_tmp]
    if training:
        sparse_mat = tfidf.fit_transform(v)
    else:
        sparse_mat = tfidf.transform(v)
    if debug:
        print("TFIDF features: " + str(tfidf.get_feature_names()))
    # form return values
    headlines_vect = []
    articles_vect = {}
    i = 0
    for row in sparse_mat:
        row = row.todense().tolist()[0]
        if vect_tmp[i][0] == is_headline:
            headlines_vect.append([row, fnc_label_to_vect(vect_tmp[i][3]), vect_tmp[i][3], vect_tmp[i][1], vect_tmp[i][2]])
        else:
            articles_vect[vect_tmp[i][1]] = row
        i += 1
    rv = []
    rv_mat = np.zeros((len(headlines_vect), 2*len(headlines_vect[0][0])))
    i = 0
    for h in headlines_vect:
        rv.append(
            [articles_vect[h[3]] + h[0], h[1], h[2], h[3], h[4]]
        )
        rv_mat[i, :] = np.array(h[0] + articles_vect[h[3]])
        i += 1
    label_vect_idx = 1
    input_idx = 0
    assert len(rv) != 0
    return rv, input_idx, label_vect_idx, rv_mat


def generate_fnc_features(word_embedding_w2v_file, headlines, articles, dim=300, max_token=10, verbose=True, glove=None,
                       lmtzr=None, up_sample=False, down_sample=False, strategy='concat'):
    def normalize_word(w, lm):
        return lm.lemmatize(w)
    if lmtzr is None:
        lmtzr = WordNetLemmatizer()
    if glove is None:
        glove = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_w2v_file, binary=False)
    if verbose:
        print('unrelated: ' + str(len([h for h in headlines if h['Stance'] == 'unrelated'])))
        print('agree: ' + str(len([h for h in headlines if h['Stance'] == 'agree'])))
        print('disagree: ' + str(len([h for h in headlines if h['Stance'] == 'disagree'])))
        print('discuss: ' + str(len([h for h in headlines if h['Stance'] == 'discuss'])))
    if up_sample and down_sample:
        up_sample = False
        down_sample = False
    if up_sample:
        print('up sampling')
        headlines = fnc_up_sample(headlines)
    if down_sample:
        print('down sampling')
        headlines = fnc_down_sample(headlines)
    rv = []
    rv_mat = None
    np.random.shuffle(headlines)
    print('generating vector representations, strategy=' + strategy)
    if strategy == 'concat':
        clean_dataset(headlines, articles)
        rv_mat = np.zeros((len(headlines), dim * max_token * 2))
        i = 0
        for h_line in headlines:
            label = h_line['Stance']
            body_id = h_line['Body ID']
            h_text = h_line['Headline']
            a_text = articles[body_id]
            curr_word_idx = 0
            h_embeddings = np.zeros((max_token, dim))
            a_embeddings = np.zeros((max_token, dim))
            for word in nltk.word_tokenize(h_text):
                if curr_word_idx >= max_token:
                    break
                if word in glove.vocab:
                    h_embeddings[curr_word_idx, :] = glove[word]
                else:
                    word = normalize_word(word, lmtzr)
                    if word in glove.vocab:
                        h_embeddings[curr_word_idx, :] = glove[word]
                curr_word_idx += 1
            h_embeddings = h_embeddings.flatten()
            curr_word_idx = 0
            for word in nltk.word_tokenize(a_text):
                if curr_word_idx >= max_token:
                    break
                if word in glove.vocab:
                    a_embeddings[curr_word_idx, :] = glove[word]
                else:
                    word = normalize_word(word, lmtzr)
                    if word in glove.vocab:
                        a_embeddings[curr_word_idx, :] = glove[word]
                curr_word_idx += 1
            a_embeddings = a_embeddings.flatten()
            x_vect = np.r_[a_embeddings, h_embeddings]
            assert x_vect.shape == (dim * max_token * 2,)
            rv_mat[i, :] = x_vect
            i += 1
            y_vect = fnc_label_to_vect(label)
            rv.append([x_vect.tolist(), y_vect, h_text, body_id, label])

    elif strategy == 'l2sum':
        clean_dataset(headlines, articles)
        rv_mat = np.zeros((len(headlines), dim * 2))
        i = 0
        for h_line in headlines:
            label = h_line['Stance']
            body_id = h_line['Body ID']
            h_text = h_line['Headline']
            a_text = articles[body_id]
            curr_word_idx = 0
            h_embeddings = np.zeros((dim,))
            a_embeddings = np.zeros((dim,))
            for word in nltk.word_tokenize(h_text):
                if curr_word_idx >= max_token:
                    break
                if word in glove.vocab:
                    h_embeddings += glove[word]
                else:
                    word = normalize_word(word, lmtzr)
                    if word in glove.vocab:
                        h_embeddings += glove[word]
                curr_word_idx += 1
            norm = np.linalg.norm(h_embeddings)
            if norm != 0:
                h_embeddings /= norm
            curr_word_idx = 0
            for word in nltk.word_tokenize(a_text):
                if curr_word_idx >= max_token:
                    break
                if word in glove.vocab:
                    a_embeddings += glove[word]
                else:
                    word = normalize_word(word, lmtzr)
                    if word in glove.vocab:
                        a_embeddings += glove[word]
                curr_word_idx += 1
            norm = np.linalg.norm(a_embeddings)
            if norm != 0:
                a_embeddings /= norm
            x_vect = np.r_[a_embeddings, h_embeddings]
            assert x_vect.shape == (dim * 2,)
            rv_mat[i, :] = x_vect
            i += 1
            y_vect = fnc_label_to_vect(label)
            rv.append([x_vect.tolist(), y_vect, h_text, body_id, label])

    elif strategy == 'paircosine':
        sid = SentimentIntensityAnalyzer()
        handcraft_features = 5
        rv_mat = np.zeros((len(headlines), max_token + handcraft_features))
        i = 0
        h_polars = []
        a_polars = {}
        for h_line in headlines:
            body_id = h_line['Body ID']
            h_text = h_line['Headline']
            a_text = articles[body_id]
            h_polarity = sid.polarity_scores(h_text)['compound']
            a_polarity = sid.polarity_scores(a_text)['compound']
            h_polars.append(h_polarity)
            a_polars[body_id] = a_polarity
        clean_dataset(headlines, articles)
        for h_line in headlines:
            x_vect = [0 for _ in range(max_token + handcraft_features)]
            label = h_line['Stance']
            body_id = h_line['Body ID']
            h_text = h_line['Headline']
            a_text = articles[body_id]
            x_vect[0] = h_polars[i]
            x_vect[1] = a_polars[body_id]
            curr_word_idx = 0
            strong_pos_count = 0
            strong_neg_count = 0
            eu_dist_count = 0
            h_word_polar_count = 0
            a_word_polar_count = 0
            h_word_list = nltk.word_tokenize(h_text)
            a_word_list = nltk.word_tokenize(a_text)
            for h_word in h_word_list:
                top_eu_dist = []
                top_cos_sim = []
                h_word_embedding = None
                if curr_word_idx >= max_token:
                    break
                if h_word in glove.vocab:
                    h_word_embedding = glove[h_word]
                else:
                    h_word = normalize_word(h_word, lmtzr)
                    if h_word in glove.vocab:
                        h_word_embedding = glove[h_word]
                if h_word_embedding is None:
                    continue
                x_vect[0] += sid.polarity_scores(h_word)['compound']
                h_word_polar_count += 1
                for a_word in a_word_list:
                    a_word_embedding = None
                    if a_word in glove.vocab:
                        a_word_embedding = glove[a_word]
                    else:
                        a_word = normalize_word(a_word, lmtzr)
                        if a_word in glove.vocab:
                            a_word_embedding = glove[a_word]
                    if a_word_embedding is None:
                        continue
                    x_vect[1] += sid.polarity_scores(h_word)['compound']
                    a_word_polar_count += 1
                    cos_sim = cosine_similarity(h_word_embedding.reshape(1, -1), a_word_embedding.reshape(1, -1))
                    eu_dist = euclidean_distances(h_word_embedding.reshape(1, -1), a_word_embedding.reshape(1, -1))
                    top_cos_sim.append(cos_sim[0, 0])
                    top_eu_dist.append(eu_dist[0, 0])
                    eu_dist_count += 1
                    if cos_sim[0, 0] > 0.8:
                        x_vect[2] += cos_sim[0, 0]
                        strong_pos_count += 1
                    elif cos_sim[0, 0] < -0.5:
                        x_vect[3] += cos_sim[0, 0]
                        strong_neg_count += 1
                top_cos_sim.sort(reverse=True)
                top_eu_dist.sort()
                top_cos_sim = top_cos_sim[:min(2, len(top_cos_sim))]
                top_eu_dist = top_eu_dist[:min(2, len(top_eu_dist))]
                x_vect[curr_word_idx + handcraft_features] = sum(top_cos_sim) / 2
                x_vect[handcraft_features - 1] += sum(top_eu_dist) / 2
                curr_word_idx += 1
            x_vect[0] /= max(h_word_polar_count, 1)
            x_vect[1] /= max(a_word_polar_count, 1)
            x_vect[handcraft_features - 1] /= max(len(h_word_list), 1)
            x_vect[2] /= max(strong_pos_count, 1)
            x_vect[3] /= max(strong_neg_count, 1)
            rv_mat[i, :] = np.array(x_vect)
            i += 1
            y_vect = fnc_label_to_vect(label)
            rv.append([x_vect, y_vect, h_text, body_id, label])
        max_dist = max([v[0][handcraft_features-1] for v in rv])
        for v in rv:
            v[0][handcraft_features-1] /= max_dist
    else:
        print('unknown strategy')
        exit(1)

    label_vect_idx = 1
    input_idx = 0
    assert len(rv) != 0
    return rv, input_idx, label_vect_idx, rv_mat


def fnc_one_hot_label_decode(pred_labels):
    pred_strs = []
    for a in range(len(pred_labels)):
        if pred_labels[a][0] == 1:
            pred_strs.append(fnc_vect_to_label(0))
        elif pred_labels[a][1] == 1:
            pred_strs.append(fnc_vect_to_label(1))
        elif pred_labels[a][2] == 1:
            pred_strs.append(fnc_vect_to_label(2))
        elif pred_labels[a][3] == 1:
            pred_strs.append(fnc_vect_to_label(3))
        else:
            pred_strs.append(fnc_vect_to_label(0))
    return pred_strs


def fnc_down_sample(data):
    unrelated_data = [d for d in data if d['Stance'] == 'unrelated']
    discuss_data = [d for d in data if d['Stance'] == 'discuss']
    agree_data = [d for d in data if d['Stance'] == 'agree']
    disagree_data = [d for d in data if d['Stance'] == 'disagree']
    min_len = min(len(unrelated_data),min(len(discuss_data),min(len(agree_data),len(disagree_data))))
    np.random.shuffle(unrelated_data)
    np.random.shuffle(discuss_data)
    np.random.shuffle(agree_data)
    np.random.shuffle(disagree_data)
    unrelated_data = unrelated_data[:min_len]
    discuss_data = discuss_data[:min_len]
    agree_data = agree_data[:min_len]
    disagree_data = disagree_data[:min_len]
    rv = unrelated_data + discuss_data + agree_data + disagree_data
    np.random.shuffle(rv)
    return rv


def fnc_up_sample(data):
    unrelated_data = [d for d in data if d['Stance'] == 'unrelated']
    discuss_data = [d for d in data if d['Stance'] == 'discuss']
    agree_data = [d for d in data if d['Stance'] == 'agree']
    disagree_data = [d for d in data if d['Stance'] == 'disagree']
    max_len = max(len(unrelated_data), max(len(discuss_data), max(len(agree_data), len(disagree_data))))
    rv = []
    tmp = []
    i = 0
    while len(tmp) < max_len:
        tmp.append(unrelated_data[i])
        i += 1
        i %= len(unrelated_data)
    rv += tmp
    tmp = []
    i = 0
    while len(tmp) < max_len:
        tmp.append(discuss_data[i])
        i += 1
        i %= len(discuss_data)
    rv += tmp
    tmp = []
    i = 0
    while len(tmp) < max_len:
        tmp.append(agree_data[i])
        i += 1
        i %= len(agree_data)
    rv += tmp
    tmp = []
    i = 0
    while len(tmp) < max_len:
        tmp.append(disagree_data[i])
        i += 1
        i %= len(disagree_data)
    rv += tmp
    np.random.shuffle(rv)
    return rv


def get_class_rep(data, label_index):
    return {
        'unrelated':next(obj for obj in data if obj[label_index] == 'unrelated'),
        'discuss':next(obj for obj in data if obj[label_index] == 'discuss'),
        'agree':next(obj for obj in data if obj[label_index] == 'agree'),
        'disagree':next(obj for obj in data if obj[label_index] == 'disagree'),
    }


def fnc_label_to_vect(label):
    y_vect = 0
    if label == 'unrelated':
        y_vect = 0
    elif label == 'discuss':
        y_vect = 1
    elif label == 'agree':
        y_vect = 2
    elif label == 'disagree':
        y_vect = 3
    return y_vect


def fnc_vect_to_label(vect):
    if vect == 0:
        return 'unrelated'
    elif vect == 1:
        return 'discuss'
    elif vect == 2:
        return 'agree'
    elif vect == 3:
        return 'disagree'
    else:
        raise RuntimeError('No matching label found for vector')


def split_by_ratio(train_percent, data, label_vect_idx):
    rv_train = []
    rv_test = []
    unrelated_data = [d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'unrelated']
    discuss_data = [d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'discuss']
    agree_data = [d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'agree']
    disagree_data = [d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'disagree']
    assert len(unrelated_data) != 0
    assert len(discuss_data) != 0
    assert len(agree_data) != 0
    assert len(disagree_data) != 0
    np.random.shuffle(unrelated_data)
    np.random.shuffle(discuss_data)
    np.random.shuffle(agree_data)
    np.random.shuffle(disagree_data)
    train_amount_unrelated = int(len(unrelated_data) * train_percent)
    train_amount_agree = int(len(agree_data) * train_percent)
    train_amount_disagree = int(len(disagree_data) * train_percent)
    train_amount_discuss = int(len(discuss_data) * train_percent)
    rv_train += unrelated_data[:train_amount_unrelated]
    rv_test += unrelated_data[train_amount_unrelated:]
    rv_train += agree_data[:train_amount_agree]
    rv_test += agree_data[train_amount_agree:]
    rv_train += disagree_data[:train_amount_disagree]
    rv_test += disagree_data[train_amount_disagree:]
    rv_train += discuss_data[:train_amount_discuss]
    rv_test += discuss_data[train_amount_discuss:]
    np.random.shuffle(rv_train)
    np.random.shuffle(rv_test)
    return rv_train, rv_test


def split_fnc_k_folds(data, k, label_vect_idx):
    rv = []
    unrelated_data = np.array([d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'unrelated'])
    discuss_data = np.array([d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'discuss'])
    agree_data = np.array([d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'agree'])
    disagree_data = np.array([d for d in data if fnc_vect_to_label(d[label_vect_idx]) == 'disagree'])
    fold_size = int(len(data) / k)
    assert fold_size != 0
    np.random.shuffle(unrelated_data)
    np.random.shuffle(discuss_data)
    np.random.shuffle(agree_data)
    np.random.shuffle(disagree_data)
    unrelated_data_split = np.split(unrelated_data, k)
    discuss_data_split = np.split(discuss_data, k)
    agree_data_split = np.split(agree_data, k)
    disagree_data_split = np.split(disagree_data, k)
    for i in range(k):
        fold_join = np.r_[unrelated_data_split[i], disagree_data_split[i], discuss_data_split[i], agree_data_split[i]]
        np.random.shuffle(fold_join)
        rv.append(fold_join.tolist())
    assert len(rv) == k
    return rv
