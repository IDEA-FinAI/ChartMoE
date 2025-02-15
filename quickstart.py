"""
    FEATURE: QuickTour of ChartMoE
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
"""
from chartmoe import ChartMoE_Robot
import torch

robot = ChartMoE_Robot()
image_path = "examples/bar2.png"
question = "Redraw the chart with python matplotlib, giving the code to highlight the column corresponding to the year in which the student got the highest score (painting it red). Please keep the same colors and legend as the input chart."

history = ""
with torch.cuda.amp.autocast():
    response, history = robot.chat(image_path=image_path, question=question, history=history)

print(response)

'''Response:
```python
import matplotlib.pyplot as plt

data = [3.3, 3.5, 3.6, 3.8, 3.7, 3.6, 3.8]
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
labels = ['Student A Average GPA']
colors = ['blue']

plt.bar(years, data, color=colors)
plt.title('Student Performance')
plt.xlabel('Year')
plt.ylabel('Student A Average GPA')
plt.legend(labels)

# Highlight the year with the highest score
highest_score_index = data.index(max(data))
plt.bar(years[highest_score_index], data[highest_score_index], color='red')

plt.show()
```

'''