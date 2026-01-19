import pandapower.networks as nw
import pandapower.plotting as plot
import matplotlib.pyplot as plt

# 加载 case9 网络
net = nw.case9()

# 创建坐标（如果已有可以加 overwrite=True）
plot.create_generic_coordinates(net, overwrite=True)

# 获取坐标
coords = net.bus_geodata

# 画出网络结构
plot.simple_plot(net, show_plot=False)

# 添加 bus 编号标注
for idx, (x, y) in coords.iterrows():
    plt.text(x, y + 0.02, str(idx), color='black', fontsize=12, ha='center')

# 展示图像
plt.title("IEEE 9-Bus System with Bus Numbers")
plt.show()
