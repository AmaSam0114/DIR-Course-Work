from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# transaction dataset
data = [
    ['Milk', 'Cereal', 'Pasta', 'Rice', 'Fish'],
    ['Milk', 'Cereal', 'Pasta', 'Bread', 'Fruits', 'Vegetables'],
    ['Cereal', 'Pasta', 'Bread', 'Fruits', 'Vegetables', 'Yoghurt'],
    ['Milk', 'Fruits', 'Vegetables', 'Yoghurt'],
    ['Milk', 'Fruits', 'Vegetables', 'Rice', 'Sugar', 'Fish'],
    ['Vegetables', 'Rice', 'Sugar', 'Fish'],
    ['Cereal', 'Pasta', 'Bread', 'Fruits', 'Vegetables', 'Yoghurt'],
    ['Rice', 'Pasta', 'Bread', 'Fruits', 'Vegetables', 'Sugar'],
    ['Bread', 'Fruits', 'Vegetables', 'Sugar'],
    ['Milk', 'Bread', 'Fruits']
]
# Convert the dataset into binary format for Apriori
all_items = set(item for transaction in data for item in transaction)
encoded_data = pd.DataFrame([{item: (item in transaction) for item in all_items} for transaction in data])

# Apriori algorithm to generate frequent itemsets
min_support = 0.5  # Support threshold of 5 out of 10 transactions
frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

# Calculate the total number of itemsets
num_itemsets = len(frequent_itemsets)

# association rules with confidence > 75%
rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="confidence", min_threshold=0.75)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules with Confidence > 75%:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
