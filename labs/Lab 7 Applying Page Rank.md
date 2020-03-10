---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**PUT YOUR FULL NAME(S) HERE**


## Should we grade this notebook? (Answer yes or no)


???YES OR NO???


**Instructions:** Pair programming assignment. Submit only a single notebook unless you deviate significantly after lab on Thursday. If you submit individually, make sure you indicate who you worked with originally. Make sure to include your first and last names. For those students who push to individual repos but still work in groups, please indicate which notebook should be graded.


# Applying page rank

## Lab Assignment

This is a pair programming assignment. I strongly
discourage individual work for this (and other team/pair programming) lab(s), even if you think you can do it
all by yourself. Also, this is a pair programming assignment, not a ”work in teams of two” assignment. Pair
programming requires joint work on all aspects of the project without delegating portions of the work to individual
1
team members. For this lab, I want all your work — discussion, software development, analysis of the results,
report writing — to be products of joint work.
Students enrolled in the class can pair with other students enrolled in the class. Students on the waitlist can
pair with other students on the waitlists. In the cases of ”odd person out” situations, a team of three people can
be formed, but that team must (a) ask and answer one additional question, and (b) work as a pair would, without
delegation of any work off-line.


## At the end of this lab, I should be able to
* Explain how page rank can be applied to different types of graph datasets
* Explain an experiment that times how long it takes a PageRank implementation to run on datasets of different sizes

```python
# We need a better version
!pip install -U scikit-learn
```

## Our implementation

```python
%matplotlib inline 
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
import networkx as nx
import pandas as pd

def create_graph(N,names="abcdefghijklmnopqrstuvwxyz"):
    G = nx.DiGraph()

    for i in range(N):
        G.add_node(names[i], score=1/N)
    return G

def one_iteration(G,d=0.85):
    counts={} # number of unique links going out of each node
    for n in G.nodes():
        counts[n] = 0
    for u,v in G.edges():
        if u not in counts:
            counts[u] = 0
        counts[u] += 1

    counts_series = pd.Series(counts)
    sinks = list(counts_series.index[counts_series == 0])

    edges = {} # keep track of the links that are coming into a node
    for n in G.nodes():
        edges[n] = []
    for u,v in G.edges():
        edges[v].append(u) # reverse the order (edge is from u->v but our data structure is from v->u)

    scores=dict((n,d['score']) for n,d in G.nodes(data=True))
    attrs=dict((n,{"score":d['score']}) for n,d in G.nodes(data=True))

    for v in counts.keys():
        attrs[v]['score'] = (1-d)/len(G.nodes())
        for u in edges[v]:
            attrs[v]['score'] += d*scores[u]/counts[u]

    for u in sinks:
        for n in G.nodes():
            attrs[n]['score'] += d*scores[u]/len(G.nodes())
    nx.set_node_attributes(G, attrs)

def run_page_rank(G,iterations=20,d=0.85):        
    for iteration in range(iterations):
        one_iteration(G,d=d)
    return dict((n,d['score']) for n,d in G.nodes(data=True))

def plot(G):
    pos = nx.circular_layout(G)  # positions for all nodes

    node_size=4500

    # labels
    labels=dict((n,"%s:%.2f"%(n,d['score'])) for n,d in G.nodes(data=True))
    #nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_family='sans-serif')

    nx.draw_networkx_edges(G, pos=pos, with_labels=True, node_size=node_size, alpha=0.3, arrows=True,
            arrowsize=20, width=2)
    # draw white circles over the lines
    nx.draw_networkx_nodes(G, pos=pos, with_labels=True, node_size=node_size, alpha=1, arrows=True,
            arrowsize=20, width=2, node_color='w')
    # draw the nodes as desired
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size, alpha=.3, arrows=True,
            arrowsize=20, width=2)
    nx.draw_networkx_labels(G, labels=labels,pos=pos)

    plt.axis('off')
    plt.show()
```

```python
G = create_graph(4)

G.add_edge('a', 'b')
G.add_edge('b', 'a')
G.add_edge('a', 'c')
G.add_edge('c', 'a')
G.add_edge('a', 'd')
G.add_edge('d', 'a')

plot(G)
```

```python
ranks = run_page_rank(G)
display(ranks)
ranks = pd.Series(ranks).sort_values(ascending=False)
ranks
```

```python
sum(ranks)
```

```python
plot(G)
```

### What if you want to time it?

```python
%%timeit
G = create_graph(4)

G.add_edge('a', 'b')
G.add_edge('b', 'a')
G.add_edge('a', 'c')
G.add_edge('c', 'a')
G.add_edge('a', 'd')
G.add_edge('d', 'a')

ranks = run_page_rank(G)
```

## Exercise 1 (Worth 8 points)
Apply page rank to two graphs that you create with 6 nodes each. 
* Graph 1: Approximately every page should be of relatively equal importance
* Graph 2: Contain some structure (or important nodes). 
Write up a brief discussion about what page rank is showing about your graphs. I want to see graphs and discussions that reference your graphs. 

```python
## CODE HERE
```

## Exercise 2 (Worth 3 points)
Now let's load up a real dataset that contains information about web pages. This dataset is a webcrawl using links that go through epa.gov (warning... it's a bit old). Run and study the code below. Then answer the questions at the end.

```python
!head ~/csc-466-student/data/epa-webcrawl/gr0.epa.txt
```

```python
!tail ~/csc-466-student/data/epa-webcrawl/gr0.epa.txt
```

```python
nodes = {}
edges = {}
for line in open("/home/jovyan/csc-466-student/data/epa-webcrawl/gr0.epa.txt").readlines():
    fields = line.strip().split(" ")
    if line[0] == "n":
        nodes[fields[1]] = fields[2]
    elif line[0] == "e":
        if fields[1] not in edges:
            edges[fields[1]] = []
        edges[fields[1]].append(fields[2])
```

```python
len(list(nodes.keys()))
```

```python
pd.Series(nodes)
```

```python
pd.Series(edges)
```

```python
node_names = list(nodes.keys())
G = create_graph(len(node_names),names=node_names)

for u in edges.keys():
    for v in edges[u]:
        G.add_edge(u,v)
```

### Example to show you how long it took me to run it

```python
%%timeit -n 1 -r 1
run_page_rank(G)
```

### Now run it for real and get the ranks

```python
ranks = run_page_rank(G)
```

```python
ranks = pd.Series(ranks).sort_values(ascending=False)
```

```python
ranks
```

```python
ranks.iloc[:20].plot.bar()
```

```python
df = pd.Series(nodes).to_frame()
df.columns = ["URL"]
ranks.name = "PageRank"
df = df.join(ranks)
```

```python
df
```

```python
df.sort_values(by="PageRank",ascending=False)
```

```python
df.sort_values(by="PageRank",ascending=False).set_index("URL").iloc[:20].plot.bar()
```

**Questions**

1. Do you think we converged? How would you show me you had either converged or not converged?
2. Run your experiment to see if we converged.
2. Assuming we've converged, what does the bar chart above tell you about the webpages? 



```python

```
