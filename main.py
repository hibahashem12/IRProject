
import ir_datasets
import itertools
import glob
import pickle
from dataProccessing import data_proccessing
from sklearn.metrics import silhouette_score


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import json
documents = {}
documentname = {}

#--------------------------procces data and save it in corpus1-------------------------------
# dataset = ir_datasets.load("highwire/trec-genomics-2006")
# index=1
# for doc in itertools.islice(dataset.docs_iter(),4000 ):
#     doc_id = doc.doc_id
#     doc_text = doc.abstract
#     with open(f'corpus2/{index}.txt', 'w', encoding='utf-8') as f:
#         tokens =data_proccessing(doc_text)
#         for token in tokens :
#           f.write(token + ' ')
#     f.close()
#     index=index+1
#-----------------------dataset2---------------------'
# dataset = ir_datasets.load("highwire/trec-genomics-2006")
# index=1
# for doc in itertools.islice(dataset.docs_iter(),4000):
#     doc_id = doc.doc_id
#     docs = doc.spans
#     text=''
#
#     print(len(docs))
#     for i in docs:
#         text+=i.text
#     with open(f'corpus2/{index}.txt', 'w', encoding='utf-8') as f:
#         tokens =data_proccessing(text)
#         for token in tokens :
#             f.write(token + ' ')
#         f.close()
#     index=index+1


#--------------------------build the documents-------------------------------

# dataset = ir_datasets.load("cord19/trec-covid")
# index=1
# for doc in itertools.islice(dataset.docs_iter(),4000 ):
#     doc_id = doc.doc_id
#     f = open('corpus1/'+str(index) + '.txt', "r", encoding='utf-8')
#     doc_text = f.read()
#
#     documents[doc_id]=doc_text
#     index=index+1
# print(documents)

#--------------------------save the documents that i build-------------------------------

# # Save the documents dictionary to a file using pickle
# with open("documents.pkl", "wb") as f:
#     pickle.dump(documents, f)
#--------------------------save the TF-IDF vectors-------------------------------
# with open("doc_vectors.pkl", "wb") as f:
#     pickle.dump(doc_vectors, f)
def searchquery(query,notprocced):
    #--------------------------Load the TF-IDF vectors-------------------------------
    with open("doc_vectors.pkl", "rb") as f:
        doc_vectors = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    # --------------------------Load document dictionary-------------------------------

    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    x=1
    for doc in documents:
        documentname[doc]=x
        x=x+1
    # # --------------------------create vectorizer for document dictionary-------------------------------
    #
    # vectorizer = TfidfVectorizer(stop_words='english')
    # doc_vectors=vectorizer.fit_transform(documents.values())
    #
    # # Save the vectorizer object and the document vectors using pickle
    # with open("vectorizer.pkl", "wb") as f:
    #     pickle.dump(vectorizer, f)
    #
    # with open("doc_vectors.pkl", "wb") as f:
    #     pickle.dump(doc_vectors, f)
    #----------------------------load vectorizer for documents --------------------------

    # Get the feature names (vocabulary) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Fit the vectorizer on the documents and transform the documents to TF-IDF representations
    # Print the TF-IDF vectors and feature names
    print(doc_vectors)
    print(feature_names)


    # -------------------------- Build an inverted index from the TF-IDF vectors-----------------------
    inverted_index = {}
    feature_names = vectorizer.get_feature_names_out()
    # for i, feature_name in enumerate(feature_names):
    #     inverted_index[feature_name] = doc_vectors.getcol(i).nonzero()[0]
    #-------------------------------save the inverted index----------------------
    # with open("invertedIndex.pkl", "wb") as f:
    #     pickle.dump(inverted_index, f)
    # #-------------------------------load the inverted index------------------------
    with open("invertedIndex.pkl", "rb") as f:
       inverted_index = pickle.load(f)
    #--------------------------------Perform KMeans clustering on the TF-IDF representations of the documents
    num_clusters =5  # Choose the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(doc_vectors)


    # --------------------------Transform the query into a TF-IDF vector------------------------


    query_vector = vectorizer.transform([query])
    #///////////////////////////////cluster_distances ///////////////////////
    cluster_distances = kmeans.transform(query_vector)
    print('cluster_distances',cluster_distances)
    doc_labels = kmeans.labels_
    # Sort the distances in ascending order and select the top num_relevant_clusters clusters
    relevant_clusters = cluster_distances.argsort()[0, :num_clusters]
    print('relevant_clusters',relevant_clusters)
    releventDocuments=[]
    for cluster_num in relevant_clusters[:3]:
        doc_indices = [i for i, label in enumerate(doc_labels) if label == cluster_num]
        releventDocuments+=doc_indices
        print(releventDocuments)
    print(len(releventDocuments))
    #////////////////////////////////////////////////////////////////////////
    cos_sim = np.zeros(doc_vectors.shape[0])
    query_words = query.split()
    for query_word in query_words:
        if query_word in inverted_index:
            # doc_indices = inverted_index[query_word]
            print('doc_indices',releventDocuments)
            doc_vectors_relevant = doc_vectors[releventDocuments]
            cos_sim_relevant = np.dot(doc_vectors_relevant, query_vector.T).toarray().flatten()
            cos_sim[releventDocuments] += cos_sim_relevant
    # for cluster_label in relevant_clusters:
    #     print("cluster_label", cluster_label)
    #     print(documents[0])
    #     cluster_documents = [documents[i] for i in range(len(documents)) if kmeans.labels_[i] == cluster_label]
    #     print('cluster_documents', cluster_documents)
    #     cluster_tfidf_matrix = doc_vectors[kmeans.labels_ == cluster_label]
    #     print('cluster_tfidf_matrix', cluster_tfidf_matrix)
    #     cluster_document_similarities = cosine_similarity(cluster_tfidf_matrix, query_vector)
    #     print('cluster_document_similarities', cluster_document_similarities)


    # --------------------------Rank the documents by their cosine similarity to the query----------------
    doc_ids = np.argsort(cos_sim)[::-1]
    #-----------------------to add retrieved_documents--------------------------------
    retrieved_documents=[]
    retrieved_documentsNames=[]

    # --------------------------Print the ranked documents and their cosine similarity scores-------------------
    for i, doc_id in enumerate(doc_ids):
        print(f"{i+1}. Document '{list(documents.keys())[doc_id]}' has cosine similarity score of {cos_sim[doc_id]:.4f}")
        if cos_sim[doc_id]>0.0:
         retrieved_documents.append(list(documents.keys())[doc_id])
         docid=list(documents.keys())[doc_id]
         retrieved_documentsNames.append(documentname[docid])

    print(retrieved_documentsNames)
    #///////////////////////////////////get qrel //////////////////////////////
    qrels = {}
    # qrels_file_paths = "C:\\Users\\Hiba\\PycharmProjects\\pythonProject2\\qrel"
    # qrels_file_paths = glob.glob(qrels_file_paths + "/*.txt")
    #
    #
    # for qrels_file_path in qrels_file_paths:
    #     with open(qrels_file_path, 'r') as qrels_file:
    #         for line in qrels_file:
    #             line_parts = line.strip().split(' ')
    #             if len(line_parts) != 3:
    #                 # print(f"Skipping line with invalid format: {line}")
    #                 continue
    #             # query_id, doc_id, relevance = line_parts
    #             query_id = int(line_parts[0][1:])  # Extract the numeric part of query_id and convert to int
    #             doc_id = line_parts[1]
    #             relevance = line_parts[2]
    #             if query_id in qrels:
    #                 qrels[query_id][doc_id] = relevance
    #             else:
    #                 qrels[query_id] = {doc_id: relevance}
    # -----------------------Save the qrels map--------------------------
    # with open("qrel.pkl", "wb") as f:
    #     pickle.dump(qrels, f)
    # -----------------------load the qrels map--------------------------
    with open("qrel.pkl", "rb") as f:
        qrels = pickle.load(f)
    #////////////////////////////get queries///////////////////////
    queries = {}
    #
    # queries_file_path = "C:\\Users\\Hiba\\PycharmProjects\\pythonProject2\\queries"
    # queries_file_paths = glob.glob(queries_file_path + "/*.txt")
    # queries = {}
    # for queries_file_path in queries_file_paths:
    #     with open(queries_file_path, 'r') as queries_file:
    #         for line in queries_file:
    #             line_parts = line.strip().split(None, 1)
    #             if len(line_parts) < 1:
    #                 continue  # Skip lines that don't have the expected format
    #             query_id = line_parts[0]
    #             query = line_parts[1]
    #             queries[query_id] = query
    # print(queries)
  # # #-----------------------------Save the queries map--------------------------
  #   with open("queries.pkl", "wb") as f:
  #       pickle.dump(queries, f)

    # -----------------------load the queries map--------------------------
    #
    with open("queries.pkl", "rb") as f:
        queries = pickle.load(f)
    # #---------------------find query index in qrel-------------------
    relevented_document=[]
    queryindex=0
    for q in queries:
        if notprocced==  queries[q]:

            queryindex=int(q[1])
    for doc in qrels[queryindex]:

        # for d in documents:
        #     if doc==d:
                relevented_document.append(doc)
    print("**************")
    print(relevented_document)
    print(retrieved_documents)
    #------------------------------------------ Calculate precision ---------------------
    #
    true_positives = len(set(relevented_document) & set(retrieved_documents))
    print(set(relevented_document) & set(retrieved_documents))
    false_positives = len(set(retrieved_documents) - set(relevented_document))
    false_negatives = len(set(relevented_document) - set(retrieved_documents))
    precision = true_positives / (true_positives + false_positives)
    print('Precision:', precision)
    #------------------------------------------ Calculate recall----------------------------------------------------------
    recall =true_positives / (true_positives+ false_negatives)
    print('recall',recall)
    #------------------------------------------ Calculate AP----------------------------------------------------------

    APprecision = 0
    num_relevant_docs_seen = 0
    sum_precision_at_k = 0
    k = 0
    for doc in retrieved_documents:
        k += 1
        if doc in relevented_document:
            num_relevant_docs_seen += 1
            APprecision = num_relevant_docs_seen / k
            sum_precision_at_k += APprecision
        if num_relevant_docs_seen == len(relevented_document):
            break

    AP = sum_precision_at_k / len(relevented_document)
    print('AP:', AP)

    #------------------------------------------ Calculate MRR----------------------------------------------------------

    reciprocal_rank = 0.0
    for rank, doc in enumerate(retrieved_documents):
        if doc in relevented_document:
            reciprocal_rank = 1.0 / (rank + 1)
            break

    MRR = reciprocal_rank

    print('MRR:', MRR)

    #
    # str = {'Precision':'',
    #        'document name':retrieved_documentsNames,
    #        'document index': retrieved_documents
    #
    #        }
    # res = json.dumps(str)
    # return res

query = 'do individuals who recover from COVID-19 show sufficient immune response, including antibody levels and T-cell mediated immunity, to prevent re-infection?'

proccedquery = data_proccessing(query)
searchquery( ' '.join(proccedquery),query)
# search('object')
#
# from flask import Flask, render_template, request
# from flask_bootstrap import Bootstrap
#
# app = Flask(__name__ ,template_folder='C:\\Users\\Hiba\\PycharmProjects\\pythonProject2')
# Bootstrap(app)
#
# # Define the list of strings to search through
# my_list = ['apple', 'banana', 'orange', 'pear', 'grape']
#
# # Define the search route
# @app.route('/', methods=['GET', 'POST'])
# def search():
#     if request.method == 'POST':
#         results = ['m']
#         return render_template('results.html', results=results)
#     else:
#         # Render the search template
#         return render_template('search.html')
# @app.route('/searchresult/', methods=['GET', 'POST'])
# def searchresult(name=None):
#     userquery = request.args.get('userquery')
#     result= searchquery(userquery)
#     return  result
#
# @app.route('/selectDataset/', methods=['GET','POST'])
# def selectdataset(name=None):
#     buf1 = request.args.get('name')
#     selectedOption=request.args.get('selectedOption')
#     if selectedOption=='dataset1':
#         path_to_folder='C:\\Users\\Hiba\\PycharmProjects\\pythonProject2\\prosseccedCorpus1'
#     elif selectedOption=='dataset2':
#         path_to_folder='C:\\Users\\Hiba\\PycharmProjects\\pythonProject2\\corpus1'
#     str = {'key':'Hello World!', 'q':buf1}
#     #out = {'key':str}
#     res = json.dumps(str)
#     return res
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
