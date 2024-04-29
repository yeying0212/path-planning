import plotly.graph_objects as go

import pandas as pd

# Read data from a csv
z_data1 = pd.read_csv('land.csv')


fig = go.Figure(data=[go.Surface(z=z_data1.values,colorscale="algae")])


fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.update_layout(title='Simulate of ocean', autosize=True,
                  scene_camera_eye=dict(x=2, y=0, z=2),
                  width=1500, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90),

)


fig.show()