# 定义地图大小和初始搜索矩形大小
import plotly.graph_objects as go
map_size = 20
initial_rectangle_size = 3

# 初始化搜索点列表
search_points = []

# 定义中心点
center_point = (map_size // 2, map_size // 2)
search_points.append(center_point)

# 逐步扩展搜索矩形并记录搜索点
rectangle_size = initial_rectangle_size
while rectangle_size <= map_size:
    # 计算当前搜索矩形的四个角点
    top_left = (center_point[0] - rectangle_size // 2, center_point[1] - rectangle_size // 2)
    top_right = (center_point[0] + rectangle_size // 2, center_point[1] - rectangle_size // 2)
    bottom_right = (center_point[0] + rectangle_size // 2, center_point[1] + rectangle_size // 2)
    bottom_left = (center_point[0] - rectangle_size // 2, center_point[1] + rectangle_size // 2)
    
    # 添加矩形边界上的点到搜索点列表
    for x in range(top_left[0], bottom_right[0] + 1):
        search_points.append((x, top_left[1]))
        search_points.append((x, bottom_right[1]))
    for y in range(top_left[1], bottom_right[1] + 1):
        search_points.append((top_left[0], y))
        search_points.append((bottom_right[0], y))
    
    # 扩展搜索矩形大小
    rectangle_size += 2

# 打印搜索点列表
    x1=[]
    y1=[]
for point in search_points: 
    x1.append(point[0])
    y1.append(point[1])
fig = go.Figure(data=[go.Scatter(x=x1, y=y1)])

fig.show()