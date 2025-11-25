# Matplotlib 库详细介绍

Matplotlib 是 Python 最著名的绘图库，提供了全面的绘图功能，可以创建高质量的静态、动态和交互式可视化。

## 一、Matplotlib 概述

### 核心特性：
- **多功能性**：支持线图、散点图、柱状图、直方图、饼图等
- **高质量输出**：支持多种格式（PNG, PDF, SVG, EPS）
- **高度可定制**：几乎可以控制图形的每个元素
- **交互式环境**：在 Jupyter Notebook 中可以直接显示图形
- **与其他库集成**：与 NumPy, Pandas 完美配合

### 安装与导入：
```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt
import numpy as np
```

---

## 二、Matplotlib 架构

### 三个主要层次：
1. **Backend**：底层，处理不同输出格式
2. **Artist**：中间层，控制图形元素（线条、文本、矩形等）
3. **Scripting**（pyplot）：顶层，提供类似 MATLAB 的简单接口

---

## 三、基本绘图流程

### 1. 最简单的绘图
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图形
plt.figure(figsize=(8, 4))  # 设置图形大小
plt.plot(x, y, label='sin(x)')  # 绘制线条
plt.xlabel('X轴')              # X轴标签
plt.ylabel('Y轴')              # Y轴标签
plt.title('正弦函数图像')       # 标题
plt.legend()                   # 显示图例
plt.grid(True)                 # 显示网格
plt.show()                     # 显示图形
```

### 2. 面向对象的方式（推荐）
```python
# 更可控的方式
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, label='sin(x)')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_title('正弦函数图像')
ax.legend()
ax.grid(True)
plt.show()
```

---

## 四、常用图表类型

### 1. 线图 (Line Plot)
```python
# 创建多个线条
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')  # 蓝色实线
ax.plot(x, y2, 'r--', linewidth=2, label='cos(x)') # 红色虚线
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('三角函数')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 2. 散点图 (Scatter Plot)
```python
# 生成随机数据
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('散点图示例')
plt.colorbar(scatter)  # 添加颜色条
plt.show()
```

### 3. 柱状图 (Bar Chart)
```python
# 数据
categories = ['苹果', '香蕉', '橙子', '梨', '草莓']
values = [23, 45, 56, 12, 67]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, values, color=['red', 'yellow', 'orange', 'green', 'pink'])
ax.set_xlabel('水果')
ax.set_ylabel('销量')
ax.set_title('水果销量柱状图')

# 在柱子上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4. 直方图 (Histogram)
```python
# 生成正态分布数据
data = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('数值')
ax.set_ylabel('频数')
ax.set_title('数据分布直方图')
ax.grid(True, alpha=0.3)
plt.show()
```

### 5. 饼图 (Pie Chart)
```python
labels = ['Python', 'Java', 'C++', 'JavaScript', '其他']
sizes = [35, 25, 20, 15, 5]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
explode = (0.1, 0, 0, 0, 0)  # 突出显示第一部分

fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
       autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # 确保饼图是圆形
ax.set_title('编程语言使用比例')
plt.show()
```

---

## 五、多子图绘制

### 1. 使用 `plt.subplots()`
```python
# 创建 2x2 的子图网格
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('多子图示例', fontsize=16)

# 数据
x = np.linspace(0, 10, 100)

# 左上角子图
axes[0, 0].plot(x, np.sin(x), 'r-')
axes[0, 0].set_title('正弦函数')
axes[0, 0].grid(True)

# 右上角子图
axes[0, 1].plot(x, np.cos(x), 'b-')
axes[0, 1].set_title('余弦函数')
axes[0, 1].grid(True)

# 左下角子图
axes[1, 0].plot(x, np.tan(x), 'g-')
axes[1, 0].set_title('正切函数')
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].grid(True)

# 右下角子图
axes[1, 1].plot(x, np.exp(x), 'm-')
axes[1, 1].set_title('指数函数')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

### 2. 复杂布局
```python
# 创建不均匀的子图布局
fig = plt.figure(figsize=(12, 8))

# 创建不同大小的子图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

# 在各个子图中绘图
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('主图')

ax2.plot(x, np.cos(x))
ax2.set_title('余弦图')

ax3.plot(x, np.tan(x))
ax3.set_title('正切图')

ax4.hist(np.random.randn(1000), bins=30)
ax4.set_title('直方图')

ax5.scatter(np.random.randn(100), np.random.randn(100))
ax5.set_title('散点图')

plt.tight_layout()
plt.show()
```

---

## 六、样式和自定义

### 1. 使用样式表
```python
# 查看可用样式
print(plt.style.available)

# 使用不同的样式
plt.style.use('ggplot')  # 或者 'seaborn', 'fivethirtyeight' 等

# 绘图
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 10, 100)
for i in range(5):
    ax.plot(x, np.sin(x + i * 0.5), label=f'sin(x + {i*0.5})')
ax.legend()
ax.set_title('不同样式下的图形')
plt.show()
```

### 2. 深度自定义
```python
# 创建高度自定义的图形
fig, ax = plt.subplots(figsize=(10, 6))

# 数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制线条
line1 = ax.plot(x, y1, color='blue', linewidth=2, linestyle='-', 
                marker='o', markersize=4, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=1, 
                label='正弦波')

line2 = ax.plot(x, y2, color='green', linewidth=2, linestyle='--', 
                alpha=0.7, label='余弦波')

# 自定义坐标轴
ax.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('振幅', fontsize=12, fontweight='bold')
ax.set_title('波形图', fontsize=14, fontweight='bold')

# 自定义网格
ax.grid(True, linestyle=':', alpha=0.6)

# 自定义图例
ax.legend(loc='upper right', frameon=True, fancybox=True, 
          shadow=True, fontsize=10)

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# 添加文本注释
ax.text(2, 0.8, '峰值区域', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 添加箭头注释
ax.annotate('最小值', xy=(4.7, -1), xytext=(6, -1.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
            fontsize=10)

plt.tight_layout()
plt.show()
```

---

## 七、实际应用示例

### 1. 机器学习模型评估可视化
```python
# 模拟模型训练过程
np.random.seed(42)
epochs = 100
train_loss = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs)
val_loss = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.03, epochs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(range(epochs), train_loss, 'b-', label='训练损失', alpha=0.7)
ax1.plot(range(epochs), val_loss, 'r-', label='验证损失', alpha=0.7)
ax1.set_xlabel('训练轮次')
ax1.set_ylabel('损失值')
ax1.set_title('训练过程')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率
train_acc = 1 - train_loss + np.random.normal(0, 0.01, epochs)
val_acc = 1 - val_loss + np.random.normal(0, 0.015, epochs)
ax2.plot(range(epochs), train_acc, 'b-', label='训练准确率', alpha=0.7)
ax2.plot(range(epochs), val_acc, 'r-', label='验证准确率', alpha=0.7)
ax2.set_xlabel('训练轮次')
ax2.set_ylabel('准确率')
ax2.set_title('准确率变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. 数据分布分析
```python
# 创建综合数据可视化
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 800)

fig = plt.figure(figsize=(12, 8))

# 主散点图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
ax1.scatter(data1, np.random.randn(len(data1)), alpha=0.6, label='数据集1')
ax1.scatter(data2, np.random.randn(len(data2)), alpha=0.6, label='数据集2')
ax1.set_xlabel('数值')
ax1.set_ylabel('随机分布')
ax1.legend()
ax1.set_title('数据分布散点图')

# 直方图
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
ax2.hist(data1, bins=30, alpha=0.7, label='数据集1', density=True)
ax2.hist(data2, bins=30, alpha=0.7, label='数据集2', density=True)
ax2.set_xlabel('数值')
ax2.set_ylabel('密度')
ax2.legend()

# 箱线图
ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax3.boxplot([data1, data2], labels=['数据集1', '数据集2'])
ax3.set_ylabel('数值')
ax3.set_title('数据分布箱线图')

# 饼图（数据比例）
ax4 = plt.subplot2grid((3, 3), (2, 2))
sizes = [len(data1), len(data2)]
labels = [f'数据集1\n{len(data1)}个', f'数据集2\n{len(data2)}个']
ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax4.set_title('数据量比例')

plt.tight_layout()
plt.show()
```

---

## 八、保存图形

```python
# 创建图形
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.legend()
ax.set_title('三角函数')
ax.grid(True)

# 保存为不同格式
fig.savefig('my_plot.png', dpi=300, bbox_inches='tight')        # PNG格式
fig.savefig('my_plot.pdf', bbox_inches='tight')                 # PDF格式
fig.savefig('my_plot.svg', bbox_inches='tight')                 # SVG格式
fig.savefig('my_plot.jpg', dpi=300, bbox_inches='tight', 
            quality=95)                                         # JPG格式

plt.show()
```

---

## 九、实用技巧

### 1. 在 Jupyter Notebook 中显示图形
```python
%matplotlib inline
# 或者用于交互式图形
# %matplotlib notebook
```

### 2. 中文显示问题解决
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
```

### 3. 颜色和标记速查
```python
# 颜色: 'b'蓝色, 'g'绿色, 'r'红色, 'c'青色, 'm'洋红, 'y'黄色, 'k'黑色, 'w'白色
# 线型: '-'实线, '--'虚线, '-.'点划线, ':'点线
# 标记: '.'点, 'o'圆圈, 's'正方形, '+'加号, 'x'叉号, '^'三角形
```

---

## 十、总结

Matplotlib 是 Python 数据可视化的基石，提供了：
- **丰富的图表类型**：满足各种可视化需求
- **高度可定制性**：可以精细控制每个图形元素
- **多种输出格式**：适合出版、网页、演示等不同场景
- **良好的集成性**：与 NumPy、Pandas 等科学计算库完美配合

掌握 Matplotlib 是进行数据分析和机器学习可视化的重要技能，它让数据变得直观易懂，是数据科学工作流中不可或缺的工具。