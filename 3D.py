import plotly.graph_objects as go

import pandas as pd

# Read data from a csv
z_data1 = pd.read_csv('land.csv')
z_data2 = pd.read_csv('sea3.csv')
dates = pd.read_csv("agentpos3.csv")
fig = go.Figure(data=[go.Surface(z=z_data1.values,colorscale="algae")])
fig.add_trace(go.Surface(z = z_data2.values, colorscale='Blues',opacity=0.9))

fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

z_tickvals = [-6,-4,-2,0,2,4]
z_ticktext = [-6,-4,-2,0,2,4]

fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=True, gridcolor='dimgray', gridwidth=2),
        yaxis=dict(
            showgrid=True, 
            gridcolor='dimgray', 
            gridwidth=2,
            ticks='outside',
        ),
        zaxis=dict(showgrid=True, gridcolor='dimgray', gridwidth=2,ticktext=z_ticktext, tickvals=z_tickvals)
    ),
    title='Simulate of ocean', 
    autosize=True,
    scene_camera_eye=dict(x=2, y=0, z=2),
    width=1500, 
    height=1000,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.add_trace(go.Scatter3d(
    x=dates.y,y=dates.x,z=dates.z,
    marker=dict(
        size=4,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=5
    )
))

fig.show()