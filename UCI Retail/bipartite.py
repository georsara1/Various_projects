# based partly on Andrea Gigli's bipartite graphs (slideshare)
# Import libraries
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import matplotlib.pyplot as plt
import time

#Import data
df = pd.read_excel('data_UCI.xlsx', header = 0)
df_sample = df.iloc[:200]

#Delete rows with no Customer ID (if there is such a case)
cleaned_retail = df_sample.loc[pd.isnull(df_sample.CustomerID) == False]

#Create lookup table
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease
#Keep in the lookup table only the items that appear in df_sample
# stockcode_in_cleanedRetail = list(cleaned_retail.StockCode.unique())
# item_lookup = item_lookup[item_lookup['StockCode'].isin(stockcode_in_cleanedRetail)]

#Create 'ratings' matrix
cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive

#Count number of products and number of customers
no_products = len(grouped_purchased.StockCode.unique())
no_customers = len(grouped_purchased.CustomerID.unique())
print('Number of customers in dataset:', no_customers)
print('Number of products in dataset:', no_products)

#Turn raw data to pivot ('ratings' matrix)
ratings = grouped_purchased.pivot(index = 'CustomerID', columns='StockCode', values='Quantity').fillna(0).astype('int')
#Binarize the ratings matrix (indicate only if a customer has purchased a product or not)
ratings_binary = ratings.copy()
ratings_binary[ratings_binary != 0] = 1

#Initialize zeros dataframe for product interactions
products_integer = np.zeros((no_products,no_products))


#Count how many times each product pair has been purchased
print('Counting how many times each pair of products has been purchased...')
for i in range(no_products):

    for j in range(no_products):
        if i != j:
            df_ij = ratings_binary.iloc[:,[i,j]]
            s1_ij = df_ij.sum(axis=1)
            pairings_ij = len(s1_ij[s1_ij == 2]) #if s1_ij == 2 it means that both products were purchased by the same customer
            products_integer[i,j] = pairings_ij
            products_integer[j,i] = pairings_ij


#Count how many customers have purchased each item
print('Counting how many times each individual product has been purchased...')
times_purchased = products_integer.sum(axis = 1)

#Construct final weighted matrix of item interactions
print('Building weighted product matrix...')
products_weighted = np.zeros((no_products,no_products))
for i in range(no_products):
    for j in range(no_products):
        if (times_purchased[i]+times_purchased[j]) != 0:
            products_weighted[i,j] = (products_integer[i,j])/(times_purchased[i]+times_purchased[j])

#Replace any nan values (from division with zero) with 0
products_weighted[np.isnan(products_weighted)] = 0

#Get list of item labels (instead of Codes)
nodes_codes = np.array(ratings_binary.columns).astype('str')
item_lookup_dict = pd.Series(item_lookup.Description.values,index=item_lookup.StockCode).to_dict()
nodes_labels = [item_lookup_dict[code] for code in nodes_codes]

#Create Graph object using the weighted product matrix as adjacency matrix
G = nx.from_numpy_matrix(products_weighted)
pos=nx.random_layout(G)

labels = {}
for idx, node in enumerate(G.nodes()):
    labels[node] = nodes_labels[idx]

nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels, font_size=6)
plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display

#Export graph to Gephi
H=nx.relabel_nodes(G,labels)
nx.write_gexf(H, "products.gexf")

#Turn into dataframe
products_weighted_pd = pd.DataFrame(products_weighted, columns = nodes_labels)
products_weighted_pd.set_index(products_weighted_pd.columns, 'product', inplace=True)

#Lets now divide each element in products_weighted dataframe with the maximum of each row.
#This will normalize values in the row and we can perceive it as the possibility af a customer also buying
#product in column j after showing interest for the product in row i
products_prob = products_weighted_pd.divide(products_weighted_pd.max(axis = 1), axis = 0)

#Now lets select a hypothetical basket of goods (one or more products) that a customer has already purchased or
#shown an interest for by clicking on an add or something, and then suggest him relative ones
basket = ['HOME BUILDING BLOCK WORD']
#Also select the number of relevant items to suggest
no_of_suggestions = 3

all_of_basket = products_prob[basket]
all_of_basket = all_of_basket.sort_values(by = basket, ascending=False)
suggestions_to_customer = list(all_of_basket.index[:no_of_suggestions])

print('You may also consider buying:', suggestions_to_customer)