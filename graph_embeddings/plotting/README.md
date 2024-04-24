## The `PaperStylePlotter` class
Use PaperStylePlotter to create publication-ready plots with a consistent style. See example below:

```python
from graph_embeddings.plotting import PaperStylePlotter

with PaperStylePlotter().apply():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_title('Title')
    plt.show()
```