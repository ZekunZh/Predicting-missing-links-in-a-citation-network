import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv
import networkx as nx
import core_rank as cr

# nltk.download('punkt') # for tokenization
# nltk.download('stopwords')
keyword_num = 10
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)),random_predictions)

with open("random_predictions.csv","wb") as pred:
    csv_out = csv.writer(pred)
    for row in random_predictions:
        csv_out.writerow(row)
        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)


## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

# some nodes may not be connected to any other node
# hence the need to create the nodes of the graph from node_info.csv,
# not just from the edge list

nodes = IDs

# create empty directed graph
# g = igraph.Graph(directed=True)
g = nx.DiGraph()

# add vertices
# g.add_vertices(nodes)
g.add_nodes_from(nodes)

# add edges
# g.add_edges(edges)
g.add_edges_from(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
training_set_reduced = [training_set[i] for i in to_keep]

#############################################################
# Calculate simrank value based on existing graph
def simrank(G, r=0.8, max_iter=100, eps=1e-4):

    nodes = G.nodes()
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)

            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0: 
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:                    
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))


    return sim

# simRank_matrix = simrank(g)
##############################################################
# Calculate keywords for each node using coreRank 
keyword_corpus = []
for i, text in enumerate(corpus):
    all_terms = cr.clean_text_simple(text, remove_stopwords=True, pos_filtering=False, stemming=True)
    if (len(all_terms)<=keyword_num):
        keyword_corpus.append(all_terms[:keyword_num])
    else:
        g = cr.terms_to_graph(all_terms, w=5)
        sorted_cores_g = cr.core_dec(g, keyword_num=2*keyword_num, weighted=True)
        core_rank_scores = cr.sum_numbers_neighbors(g, sorted_cores_g)
        keyword_corpus.append([tup[1] for tup in core_rank_scores[:keyword_num]])
    print i

############################################################33
#------------------------------------------------------------------

# we will use three basic features:

# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []

# author citation history
cit_hist = []
auth_graph = igraph.Graph(directed=True)
#------------------------------------------------------------------
# document similarity for abstract
similarity = []
doc_similarity = features_TFIDF.dot(features_TFIDF.T)
#------------------------------------------------------------------
# inverse shortest-path (to avoid distance = infinity)
inverse_shortest_distances = []
#------------------------------------------------------------------
# keyword overlap
overlap_keyword = []

def inverse_shortest_dist(g, source, target):
    try:
        return 1. / (len(nx.shortest_path(g, source=source, target=target)) + 0.)
    except nx.NetworkXNoPath:
        return 0.
#------------------------------------------------------------------



counter = 0
for i in xrange(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    similarity.append(doc_similarity[index_source, index_target])
    
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    overlap_keyword.append(len(set(keyword_corpus[index_source]).intersection(set(keyword_corpus[index_target]))))
    
    inverse_shortest_distances.append(inverse_shortest_dist(g, source=source, target=target))

    counter += 1
    if counter % 1000 == True:
        print counter, "training examples processsed"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, comm_auth, similarity, inverse_shortest_distances,overlap_keyword]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
classifier = svm.LinearSVC()

# train
classifier.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
similarity_test = []
inverse_shortest_distances_test = []
overlap_keyword_test = []
counter = 0
for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    # source = training_set_reduced[i][0]
    # target = training_set_reduced[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    similarity_test.append(doc_similarity[index_source, index_target])

    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    overlap_keyword_test.append(len(set(keyword_corpus[index_source]).intersection(set(keyword_corpus[index_target]))))
    inverse_shortest_distances_test.append(inverse_shortest_dist(g, source=source, target=target))

   
    counter += 1
    if counter % 1000 == True:
        print counter, "testing examples processsed"
        
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test, similarity_test, inverse_shortest_distances_test,overlap_keyword_test]).T

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

with open("improved_predictions.csv","wb") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)