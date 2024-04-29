import plotly.express as px

import pandas as pd

# Read data from a csv
df1 = pd.read_csv('mean_reward_out.csv')#自己手动切换 mean_reward_out.csv steps_out.csv



fig = px.line(df1 ,title='MeanReward')
fig.update_xaxes(
    type='linear',
    side='bottom',
    showgrid=False,
    linecolor='black',
    linewidth=3,
    gridwidth=2,
    title={'font': {'size': 18}, 'text': 'Episode', 'standoff': 10},
    automargin=True,
)

fig.update_yaxes(
    showline=True,
    linecolor='black',
    linewidth=3,
    gridwidth=2,
    title={'font': {'size': 18}, 'text': 'MeanReward', 'standoff': 10},
    automargin=True,
)


fig.show()

